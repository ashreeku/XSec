import json
import pickle
import random
import sys
import torch

from captum.attr import Occlusion
from timeit import default_timer
from tqdm.contrib import tzip

from train_mlp import MLP
from train_lstm import LSTMNN
from utils import set_seed, load_data, get_args


device = torch.device("cpu")


def occl_generate_importance_scores(x, y, occl_explainer):
    num_samples, num_features = x.shape
    importance_scores = torch.zeros((num_samples, num_features,))
    for i, (s, p) in enumerate(tzip(x, y)):
        sample = torch.from_numpy(s).unsqueeze(0)
        exp = occl_explainer.attribute(sample, target=p.item(), sliding_window_shapes=(3,)).detach()
        w = exp.squeeze()
        importance_scores[i] = w
    return importance_scores


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args = vars(args)
    with open(args["config"], 'r') as cfg:
        config = json.load(cfg)

    set_seed(config["seed"])
    num_seeds = 2
    seed = random.sample(range(1, 10000), num_seeds)[1 if args["alternate"] else 0]
    print(f"Running seed {seed}")
    set_seed(seed)
    
    x_train, y_train, x_test, _ = load_data(config["data_path"])

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

    if args["train"]:
        p, _, _, _ = model(torch.from_numpy(x_train))
        pred_train = p.argmax(dim=1).numpy().astype(int)
    else:
        p, _, _, _ = model(torch.from_numpy(x_test))
        pred_test = p.argmax(dim=1).numpy().astype(int)

    occl_explainer = Occlusion(model.wrapped_forward)

    start_time = default_timer()
    if args["train"]:
        importance_scores = occl_generate_importance_scores(x_train, pred_train, occl_explainer)
    else:
        importance_scores = occl_generate_importance_scores(x_test, pred_test, occl_explainer)
    end_time = default_timer()
    print(f"Latency {end_time - start_time} seconds")

    data = {
        "seed": seed,
        "importance_scores": importance_scores,
    }

    if args["train"]:
        with open(f"../results/{config['dataset']}/importance_scores/occl_train_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f"../results/{config['dataset']}/importance_scores/occl_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)