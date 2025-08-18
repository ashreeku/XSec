import json
import pickle
import random
import sys
import torch

import numpy as np

from tqdm import tqdm

from fidelity_tests import *
from train_lstm import LSTMNN
from train_mlp import MLP
from utils import *


device = torch.device("cpu")


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args = vars(args)
    with open(args["config"], 'r') as cfg:
        config = json.load(cfg)

    set_seed(config["seed"])
    num_seeds = 1
    seed = random.sample(range(1, 10000), num_seeds)[0]
    print(f"Running seed {seed}")
    set_seed(seed)

    if config["dataset"] not in ["nsl_kdd_multi"]:
        model_path = f"../checkpoint/{config['dataset']}/model_mlp.pt"
    else:
        model_path = f"../checkpoint/{config['dataset']}/model_lstm.pt"
    print(f"Loading model {model_path}")
    checkpoint = torch.load(model_path)
    if config["dataset"] not in ["nsl_kdd_multi"]:
        model = MLP(config["feature_dim"], config["num_classes"]).to(device)
    else:
        model = LSTMNN(config["feature_dim"], config["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train(False)

    _, _, x_test, _ = load_data(config["data_path"])
    x_test = torch.from_numpy(x_test)
    num_samples, num_features = x_test.shape

    logits, _, _, _ = model(x_test)
    pred_orig = logits.argmax(dim=1).int()

    with open(f"../results/{config['dataset']}/importance_scores/ig_seed{seed}.pkl", "rb") as infile:
        contents = pickle.load(infile)
    assert contents["seed"] == seed, "Incorrect seed found"
    importance_scores = contents["importance_scores"]
    importance_scores = torch.abs(importance_scores)

    step = config["evaluation"]["step"]
    x_axis = range(step, config["feature_dim"] + 1, step)

    synth_test = []
    for m in tqdm(x_axis, leave=False, desc="Synthetic Test"):
        synthetic_samples = synthetic_test(x_test, importance_scores, m)
        logits, _, _, _ = model(synthetic_samples)
        pred_new = torch.argmax(logits, dim=1).int()
        count = (pred_new == pred_orig).sum()
        synth_test.append(count / num_samples * 100.0)

    feat_aug_test = []
    not_class_loader = NotClassLoader(x_test, pred_orig, config)
    for m in tqdm(x_axis, leave=False, desc="Feature Augmentation Test"):
        # it is possible that the model predicts all samples to a single class
        # thus leading to errors in the not_class_loader when samples
        try:
            augmented_samples = feature_augmentation_test(x_test, pred_orig, importance_scores, m, not_class_loader)
            logits, _, _, _ = model(augmented_samples)
            pred_new = torch.argmax(logits, dim=1).int()
            count = (pred_new == pred_orig).sum()
            feat_aug_test.append(count / num_samples * 100.0)
        except:
            feat_aug_test.append(None)

    feat_deduc_test = []
    for m in tqdm(x_axis, leave=False, desc="Feature Deduction Test"):
        deducted_samples = feature_deduction_test(x_test, importance_scores, m)
        logits, _, _, _ = model(deducted_samples)
        pred_new = torch.argmax(logits, dim=1).int()
        count = (pred_new == pred_orig).sum()
        feat_deduc_test.append(count / num_samples * 100.0)

    data = {
        "seed": seed,
        "synth_test": np.array(synth_test),
        "feat_aug_test": np.array(feat_aug_test),
        "feat_deduc_test": np.array(feat_deduc_test),
    }
    
    with open(f"../results/{config['dataset']}/ig_fidelity_values_seed{seed}.pkl", "wb") as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)