import json
import os
import pickle
import random
import sys
import torch

from timeit import default_timer
from tqdm import tqdm

from xsec import XSec
from utils import set_seed, load_data, get_args


device = torch.device("cpu")


def get_model_path(args, config):
    model_name = f"model"
    if args['num_prototypes_per_class'] != 10:
        model_name += f"_k{args['num_prototypes_per_class']}"
        config["num_prototypes_per_class"] = args["num_prototypes_per_class"]
    if args["ablation"]:
        model_name += f"_no{args['ablation']}"
    model_name += ".pt"
    model_path = os.path.join("../checkpoint", config["dataset"], model_name)
    return model_path, config


def xsec_generate_importance_scores(x, model, n):
    importance_scores = torch.zeros_like(x)
    for i, s in enumerate(tqdm(x)):
        w = model.explain(s.unsqueeze(0), n=n)
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
    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)

    n = args["num_similarity_scores"]
    model_path, config = get_model_path(args, config)
    print(f"Loading model {model_path}")
    checkpoint = torch.load(model_path)
    model = XSec(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train(False)

    start_time = default_timer()
    if args["train"]:
        importance_scores = xsec_generate_importance_scores(x_train, model, n)
    else:
        importance_scores = xsec_generate_importance_scores(x_test, model, n)
    end_time = default_timer()
    print(f"Latency {end_time - start_time} seconds")

    data = {
        "seed": seed,
        "num_prototypes_per_class": args["num_prototypes_per_class"],
        "num_similarity_scores": args["num_similarity_scores"],
        "ablation": args["ablation"],
        "importance_scores": importance_scores,
    }

    if args["train"]:
        with open(f"../results/{config['dataset']}/importance_scores/xsec_train_seed{seed}_k{config['num_prototypes_per_class']}_n{args['num_similarity_scores']}_ablation{args['ablation']}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f"../results/{config['dataset']}/importance_scores/xsec_seed{seed}_k{config['num_prototypes_per_class']}_n{args['num_similarity_scores']}_ablation{args['ablation']}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)