import json
import pickle
import random
import torch

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
from matplotlib.ticker import FuncFormatter
import numpy as np

from sklearn.metrics import auc
from utils import set_seed, normalize_scores

def compact_tick(x, pos):
    if np.isclose(x, 0):
        return "0"
    if np.isclose(x, 1):
        return "1"
    return f"{x:g}"


if __name__ == '__main__':
    fig, axs = plt.subplots(nrows=1, ncols=5, sharey=True)
    fig.set_size_inches(25, 3)
    pad = 13

    colors = [
        "blue",
        "darkgreen",
        "red",
        # "orange",
        "pink",
        "brown",
    ]
    linestyles = [
        "solid",
        "dashed",
        "dotted",
        # "dashdot",
        (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 1, 1, 1, 1, 1)),
    ]

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
            "w/o $\mathcal{L}_{bin}$": f"xsec_seed{seed}_k10_n3_ablationbinary.pkl",
            "w/o $\mathcal{L}_{sim}$": f"xsec_seed{seed}_k10_n3_ablationdiversity.pkl",
            # "w/o $\mathcal{L}_{xe}$": f"xsec_seed{seed}_k10_n3_ablationcrossentropy.pkl",
            "w/o $\mathcal{L}_{cls}$": f"xsec_seed{seed}_k10_n3_ablationcluster.pkl",
            "w/o $\mathcal{L}_{spar}$": f"xsec_seed{seed}_k10_n3_ablationsparsity.pkl",
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
        axs[i].set_xticks([0, 0.25, 0.5, 0.75, 1])
        axs[i].xaxis.set_major_formatter(FuncFormatter(compact_tick))
        axs[i].set_yticks([0, 0.5, 1])
        axs[i].yaxis.set_major_formatter(FuncFormatter(compact_tick))
        axs[i].grid()

    axs[0].set_title(f"PDF Malware", pad=pad, fontsize=24)
    axs[1].set_title(f"Phishing Website", pad=pad, fontsize=24)
    axs[2].set_title(f"Network Intrusion", pad=pad, fontsize=24)
    axs[3].set_title(f"PE Malware", pad=pad, fontsize=24)
    axs[4].set_title(f"Network Attack", pad=pad, fontsize=24)

    axs[0].annotate("MAZ", xy=(0, 0.5), xytext=(-axs[0].yaxis.labelpad - pad, 0),
                size=24, xycoords=axs[0].yaxis.label, textcoords='offset points',
                ha='right', va='center', rotation=90)
    
    # Share y-axis ticks/labels: show them only on the leftmost subplot
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)
        ax.set_ylabel("")

    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=True, edgecolor="black", fontsize=22)
    
    plt.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.45)
    fig.subplots_adjust(wspace=0.05, hspace=0.10)
    plt.savefig(f"../results/figures/sparsity_ablation_study.pdf", bbox_inches="tight")