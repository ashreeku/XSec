import json
import os
import pickle
import random
import sys
import torch

from timeit import default_timer
from tqdm import tqdm

from train_protopnet import ProtoPNet
from utils import set_seed, load_data, get_args


device = torch.device("cpu")


def protopnet_generate_importance_scores(x, model):
    importance_scores = torch.zeros_like(x)
    for i, s in enumerate(tqdm(x)):
        w = model.explain(s.unsqueeze(0))
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

    x_train, _, x_test, _ = load_data(config["data_path"])
    x_train = torch.from_numpy(x_train).unsqueeze(1)
    x_test = torch.from_numpy(x_test).unsqueeze(1)

    model_path = f"../checkpoint/{config['dataset']}/model_protopnet.pt"
    print(f"Loading model {model_path}")
    checkpoint = torch.load(model_path)
    model = ProtoPNet(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train(False)

    start_time = default_timer()
    if args["train"]:
        importance_scores = protopnet_generate_importance_scores(x_train, model).squeeze()
    else:
        importance_scores = protopnet_generate_importance_scores(x_test, model).squeeze()
    end_time = default_timer()
    print(f"Latency {end_time - start_time} seconds")

    data = {
        "seed": seed,
        "importance_scores": importance_scores,
    }

    if args["train"]:
        with open(f"../results/{config['dataset']}/importance_scores/protopnet_train_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f"../results/{config['dataset']}/importance_scores/protopnet_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)