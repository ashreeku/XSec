import json
import pickle
import random
import torch

import numpy as np

from utils import set_seed


def calculate_stability(importance_scores_1, importance_scores_2, N):
    num_samples, num_features = importance_scores_1.shape
    sorted_indices_1 = torch.argsort(importance_scores_1, dim=1, descending=True, stable=True)[:, :N].numpy()
    sorted_indices_2 = torch.argsort(importance_scores_2, dim=1, descending=True, stable=True)[:, :N].numpy()
    intersects = np.zeros(shape=(num_samples,))
    for i, (idx1, idx2) in enumerate(zip(sorted_indices_1, sorted_indices_2)):
        intersects[i] = len(np.intersect1d(idx1, idx2, assume_unique=True))
    return np.mean(intersects).item() / N


if __name__ == '__main__':
    for dataset in ["pdf", "phishing", "netflow", "bodmas", "nsl_kdd_multi"]:
        config_path = f"../config/{dataset}.json"
        with open(config_path, 'r') as cfg:
            config = json.load(cfg)
        
        print(f"\nDataset: {dataset}")
        set_seed(config["seed"])
        num_seeds = 2
        seeds = random.sample(range(1, 10000), num_seeds)

        method_to_path = {
            # "XSec": (f"xsec_seed{seeds[0]}_k10_n3_ablationNone.pkl", f"xsec_seed{seeds[1]}_k10_n3_ablationNone.pkl"),
            # "LIME": (f"lime_seed{seeds[0]}.pkl", f"lime_seed{seeds[1]}.pkl"),
            # "SHAP": (f"shap_seed{seeds[0]}.pkl", f"shap_seed{seeds[1]}.pkl"),
            # "IG": (f"ig_seed{seeds[0]}.pkl", f"ig_seed{seeds[1]}.pkl"),
            # "GGC": (f"ggc_seed{seeds[0]}.pkl", f"ggc_seed{seeds[1]}.pkl"),
            # "Occl": (f"occl_seed{seeds[0]}.pkl", f"occl_seed{seeds[1]}.pkl"),
            "SM": (f"sm_seed{seeds[0]}.pkl", f"sm_seed{seeds[1]}.pkl"),
        }

        # if config["dataset"] in ["pdf", "phishing", "netflow"]:
        #     method_to_path["LEMNA"] = (f"lemna_seed{seeds[0]}.pkl", f"lemna_seed{seeds[1]}.pkl")
        
        # if config["dataset"] in ["netflow"]:
        #     method_to_path["xNIDS"] = (f"xnids_seed{seeds[0]}.pkl", f"xnids_seed{seeds[1]}.pkl")

        N = 10

        for method, (path1, path2) in method_to_path.items():
            with open(f"../results/{config['dataset']}/importance_scores/{path1}", "rb") as infile:
                importance_scores_1 = pickle.load(infile)["importance_scores"]
            importance_scores_1 = torch.abs(importance_scores_1)

            with open(f"../results/{config['dataset']}/importance_scores/{path2}", "rb") as infile:
                importance_scores_2 = pickle.load(infile)["importance_scores"]
            importance_scores_2 = torch.abs(importance_scores_2)
            
            print(f"{method}: {calculate_stability(importance_scores_1, importance_scores_2, N)}")