import json
import pickle
import random
import sys
import torch

import numpy as np

from rpy2 import robjects
from tqdm import trange

from utils import set_seed, load_data, get_args


def lemna_generate_importance_scores(x, Z, Beta):
    num_samples, num_features = x.shape
    importance_scores = torch.zeros((num_samples, num_features,))
    for i in trange(num_samples):
        w = Beta[:, (Z[i] - 1)]
        importance_scores[i] = w
    return importance_scores


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args = vars(args)
    with open(args["config"], 'r') as cfg:
        config = json.load(cfg)

    assert config["dataset"] in ["pdf", "phishing", "netflow"], "LEMNA is run only on binary datasets"

    set_seed(config["seed"])
    num_seeds = 2
    seed = random.sample(range(1, 10000), num_seeds)[1 if args["alternate"] else 0]
    print(f"Running seed {seed}")
    set_seed(seed)

    x_train, y_train, x_test, _ = load_data(config["data_path"])

    if args["train"]:
        param_file = f"../results/{config['dataset']}/lemna_parameters/train_{seed}.RData"
        robjects.r['load'](param_file)
        Z = torch.from_numpy(np.asarray((robjects.r['final_params'][0])))
        Beta = torch.from_numpy(np.asarray((robjects.r['final_params'][1])))
        assert len(x_train) == len(Z), "Inconsistency between data and explanations found"
    else:
        param_file = f"../results/{config['dataset']}/lemna_parameters/test_{seed}.RData"
        robjects.r['load'](param_file)
        Z = torch.from_numpy(np.asarray((robjects.r['final_params'][0])))
        Beta = torch.from_numpy(np.asarray((robjects.r['final_params'][1])))
        assert len(x_test) == len(Z), "Inconsistency between data and explanations found"

    if args["train"]:
        importance_scores = lemna_generate_importance_scores(x_train, Z, Beta)
    else:
        importance_scores = lemna_generate_importance_scores(x_test, Z, Beta)

    data = {
        "seed": seed,
        "importance_scores": importance_scores,
    }

    if args["train"]:
        with open(f"../results/{config['dataset']}/importance_scores/lemna_train_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f"../results/{config['dataset']}/importance_scores/lemna_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)