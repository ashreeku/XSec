import json
import pickle
import random
import torch

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import numpy as np

from sklearn.metrics import auc
from utils import set_seed, normalize_scores


if __name__ == '__main__':
    fig, axs = plt.subplots(nrows=1, ncols=5)
    fig.set_size_inches(25, 3)
    pad = 13

    colors = ["blue", "darkgreen", "red", "orange", "pink", "brown"]
    linestyles = ["solid", "dashed", "dotted", "dashdot", (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1))]

    num_bins = 1000
    bins = np.linspace(0.0, 1.0, num=num_bins + 1)

    for i, dataset in enumerate(["pdf", "phishing", "netflow", "bodmas", "nsl_kdd_multi"]):
        config_path = f"../config/{dataset}.json"
        with open(config_path, 'r') as cfg:
            config = json.load(cfg)
        
        print(f"\nDataset: {dataset}")
        set_seed(config["seed"])
        num_seeds = 1
        seed = random.sample(range(1, 10000), num_seeds)[0]

        method_to_path = {
            "No ablation": f"xsec_seed{seed}_k10_n3_ablationNone.pkl",
            "$\mathcal{L}_{bin}$": f"xsec_seed{seed}_k10_n3_ablationbinary.pkl",
            "$\mathcal{L}_{sim}$": f"xsec_seed{seed}_k10_n3_ablationdiversity.pkl",
            "$\mathcal{L}_{xe}$": f"xsec_seed{seed}_k10_n3_ablationcrossentropy.pkl",
            "$\mathcal{L}_{cls}$": f"xsec_seed{seed}_k10_n3_ablationcluster.pkl",
            "$\mathcal{L}_{spar}$": f"xsec_seed{seed}_k10_n3_ablationsparsity.pkl",
        }

        for j, (method, path) in enumerate(method_to_path.items()):
            with open(f"../results/{config['dataset']}/importance_scores/{path}", "rb") as infile:
                importance_scores = pickle.load(infile)["importance_scores"]
            importance_scores = torch.abs(importance_scores)
            importance_scores = normalize_scores(importance_scores)
            importance_scores = importance_scores.detach().numpy()

            densities, _ = np.histogram(importance_scores.flatten(), bins=bins, density=True)
            densities = np.cumsum(densities / num_bins)
            print(f"{method}: {auc(bins[:-1], densities)}")

            axs[i].plot(bins[:-1], densities, color=colors[j], linestyle=linestyles[j], label=method)
        axs[i].set_xlabel("Interval Size")
        axs[i].set_xlim([0.0, 1.0])
        axs[i].set_ylim([0.0, 1.0])
        axs[i].grid()

    axs[0].set_title(f"PDF Malware Identification", pad=pad, fontsize=18)
    axs[1].set_title(f"Phishing Website Detection", pad=pad, fontsize=18)
    axs[2].set_title(f"Network Intrusion Detection", pad=pad, fontsize=18)
    axs[3].set_title(f"PE Malware Classification", pad=pad, fontsize=18)
    axs[4].set_title(f"Network Attack Classification", pad=pad, fontsize=18)

    axs[0].annotate("MAZ", xy=(0, 0.5), xytext=(-axs[0].yaxis.labelpad - pad, 0),
                size=18, xycoords=axs[0].yaxis.label, textcoords='offset points',
                ha='right', va='center', rotation=90)

    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center", bbox_to_anchor=(0.517, 1.15), frameon=True, edgecolor="black")
    
    plt.tight_layout()
    plt.savefig(f"../results/figures/sparsity_ablation_study.pdf", bbox_inches="tight")