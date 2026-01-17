import json
import os
import pickle
import random
import sys
import torch
import torch.nn.functional as F

from lime import lime_tabular
from tqdm.contrib import tzip

from train_mlp import MLP
from utils import set_seed, load_data, get_args


device = torch.device("cpu")


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args = vars(args)
    with open("../config/phishing.json", 'r') as cfg:
        config = json.load(cfg)

    set_seed(config["seed"])
    num_seeds = 2
    seed = random.sample(range(1, 10000), num_seeds)[1 if args["alternate"] else 0]
    print(f"Running seed {seed}")
    set_seed(seed)

    x_train, y_train, x_test, y_test = load_data(config["data_path"])
    x_test = torch.from_numpy(x_test)

    with open(f"../results/{config['dataset']}/importance_scores/lime_seed{seed}.pkl", "rb") as infile:
        contents = pickle.load(infile)
    assert contents["seed"] == seed, "Incorrect seed found"
    importance_scores = contents["importance_scores"]
    importance_scores = torch.abs(importance_scores)

    model_path = f"../checkpoint/{config['dataset']}/model_mlp.pt"
    print(f"Loading model {model_path}")
    checkpoint = torch.load(model_path)
    model = MLP(config["feature_dim"], config["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train(False)

    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=x_train,
        training_labels=y_train,
        mode="classification",
    )

    with torch.no_grad():
        logits_test, _, _, _ = model(x_test)
        probs_test = F.softmax(logits_test, dim=1)
        pred_test = probs_test.argmax(dim=1)

    lr = 1e-2
    num_iter = 200
    factors = [1e11, 1e6]

    adv_samples = []
    adv_importances = []

    for orig_sample, orig_prob, orig_pred, orig_expl in tzip(x_test, probs_test, pred_test, importance_scores):
        orig_sample = orig_sample.unsqueeze(0)
        orig_prob = orig_prob.unsqueeze(0)
        orig_pred = orig_pred.item()
        orig_expl = orig_expl.detach().unsqueeze(0)
        orig_sample.detach().requires_grad_(False)
        orig_prob.detach().requires_grad_(False)
        orig_expl.detach().requires_grad_(False)

        # initialize adversarial sample as a copy of the original
        adv_sample = orig_sample.clone().detach()
        adv_sample.requires_grad_(True)
        optimizer = torch.optim.Adam([adv_sample], lr=lr)

        for it in range(num_iter):
            optimizer.zero_grad()

            # current logits and prediction
            adv_logits, _, _, _ = model(adv_sample)
            adv_prob = F.softmax(adv_logits, dim=1)
            adv_pred = adv_prob.argmax(dim=1).item()

            # explanation for adversarial sample
            exp = lime_explainer.explain_instance(
                data_row=adv_sample.squeeze().detach().numpy(),
                predict_fn=model.predict_proba,
                labels=(adv_pred,),
                num_features=config["feature_dim"],
                num_samples=150,
            )
            exp = sorted(exp.as_map()[adv_pred], key=lambda x: x[0])
            adv_expl = torch.tensor([j[1] for j in exp], requires_grad=False).unsqueeze(0)

            loss_expl = F.mse_loss(adv_expl, orig_expl)
            loss_pred = F.mse_loss(adv_prob, orig_prob)

            # total loss: minimize output diff, maximize explanation diff
            total_loss = factors[1] * loss_pred - factors[0] * loss_expl
            total_loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            adv_samples.append(adv_sample.squeeze(0).cpu())
            adv_importances.append(adv_expl.squeeze(0).cpu())

    adv_samples = torch.stack(adv_samples)
    adv_importances = torch.stack(adv_importances)

    # recompute predictions for original and adversarial samples
    with torch.no_grad():
        logits_orig, _, _, _ = model(x_test)
        logits_adv, _, _, _ = model(adv_samples)

        preds_orig = logits_orig.argmax(dim=1)
        preds_adv = logits_adv.argmax(dim=1)

    # mask of samples where prediction stayed the same
    same_pred_mask = preds_orig == preds_adv

    num_total = preds_orig.shape[0]
    num_same = same_pred_mask.sum().item()
    asr = num_same / num_total  # Attack Success Rate: label unchanged

    print(f"Number of samples: {num_total}")
    print(f"Samples with unchanged prediction: {num_same}")
    print(f"ASR (unchanged prediction fraction): {asr:.5f}")

    # cosine similarity between original and adversarial explanations
    # only for samples where prediction did not change
    if num_same > 0:
        cos_sim = F.cosine_similarity(
            importance_scores[same_pred_mask],
            adv_importances[same_pred_mask],
            dim=1,
            eps=1e-8,
        )
        mean_cos = cos_sim.mean().item()
        median_cos = cos_sim.median().item()

        print(f"Mean cosine similarity (explanations, same-label samples): {mean_cos:.5f}")
        print(f"Median cosine similarity (explanations, same-label samples): {median_cos:.5f}")
    else:
        print("No samples with unchanged prediction; cannot compute cosine similarity.")