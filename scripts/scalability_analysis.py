import pickle
import torch

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import numpy as np

from sklearn.metrics import auc
from utils import normalize_scores


if __name__ == "__main__":
    fig, axs = plt.subplots(nrows=1, ncols=4)
    fig.set_size_inches(20, 3)
    pad = 13
    
    with open(f"../results/bodmas/xsec_fidelity_values_seed8815_k10_n3_ablationNone.pkl", "rb") as infile:
        bodmas_100 = pickle.load(infile)
    with open(f"../results/bodmas200/xsec_fidelity_values_seed8815_k20_n3_ablationNone.pkl", "rb") as infile:
        bodmas_200 = pickle.load(infile)
    with open(f"../results/bodmas500/xsec_fidelity_values_seed8815_k50_n3_ablationNone.pkl", "rb") as infile:
        bodmas_500 = pickle.load(infile)

    x_axis = np.linspace(0.0, 1.0, 10)

    synth_100 = bodmas_100["synth_test"]
    synth_200 = bodmas_200["synth_test"]
    synth_500 = bodmas_500["synth_test"]
    
    axs[0].plot(x_axis, synth_100, color="blue", marker='x', linestyle="solid", label=f"100")
    axs[0].plot(x_axis, synth_200, color="darkgreen", marker='o', linestyle="dashed", label="200")
    axs[0].plot(x_axis, synth_500, color="red", marker='*', linestyle="dotted", label="500")
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 100])
    axs[0].set_xlabel(r"$\frac{|\mathbf{F_{x}}|}{d}$")
    axs[0].set_ylabel("PCR (%)")
    axs[0].set_title(f"Synthetic Test", pad=pad, fontsize=18)
    axs[0].grid()

    feat_aug_100 = bodmas_100["feat_aug_test"]
    mask_100 = ~np.isnan(feat_aug_100)
    feat_aug_200 = bodmas_200["feat_aug_test"]
    mask_200 = ~np.isnan(feat_aug_200)
    feat_aug_500 = bodmas_500["feat_aug_test"]
    mask_500 = ~np.isnan(feat_aug_500)
    
    if np.any(mask_100):
        axs[1].plot(x_axis[mask_100], feat_aug_100[mask_100], color="blue", marker='x', linestyle="solid", label=f"100")
    if np.any(mask_200):
        axs[1].plot(x_axis[mask_200], feat_aug_200[mask_200], color="darkgreen", marker='o', linestyle="dashed", label="200")
    if np.any(mask_500):
        axs[1].plot(x_axis[mask_500], feat_aug_500[mask_500], color="red", marker='*', linestyle="dotted", label="500")
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 100])
    axs[1].set_xlabel(r"$\frac{|\mathbf{F_{x}}|}{d}$")
    axs[1].set_ylabel("PCR (%)")
    axs[1].set_title(f"Feature Augmentation Test", pad=pad, fontsize=18)
    axs[1].grid()

    feat_deduc_100 = bodmas_100["feat_deduc_test"]
    feat_deduc_200 = bodmas_200["feat_deduc_test"]
    feat_deduc_500 = bodmas_500["feat_deduc_test"]
    
    axs[2].plot(x_axis, feat_deduc_100, color="blue", marker='x', linestyle="solid", label=f"100")
    axs[2].plot(x_axis, feat_deduc_200, color="darkgreen", marker='o', linestyle="dashed", label="200")
    axs[2].plot(x_axis, feat_deduc_500, color="red", marker='*', linestyle="dotted", label="500")
    axs[2].set_xlim([0, 1])
    axs[2].set_ylim([0, 100])
    axs[2].set_xlabel(r"$\frac{|\mathbf{F_{x}}|}{d}$")
    axs[2].set_ylabel("PCR (%)")
    axs[2].set_title(f"Feature Deduction Test", pad=pad, fontsize=18)
    axs[2].grid()

    colors = ["blue", "darkgreen", "red"]
    linestyles = ["solid", "dashed", "dotted"]

    num_bins = 1000
    bins = np.linspace(0.0, 1.0, num=num_bins + 1)

    method_to_path = {
        "100": f"../results/bodmas/importance_scores/xsec_seed8815_k10_n3_ablationNone.pkl",
        "200": f"../results/bodmas200/importance_scores/xsec_seed8815_k20_n3_ablationNone.pkl",
        "500": f"../results/bodmas500/importance_scores/xsec_seed8815_k50_n3_ablationNone.pkl",
    }

    for j, (method, path) in enumerate(method_to_path.items()):
        with open(path, "rb") as infile:
            importance_scores = pickle.load(infile)["importance_scores"]
        importance_scores = torch.abs(importance_scores)
        importance_scores = normalize_scores(importance_scores)
        importance_scores = importance_scores.detach().numpy()

        densities, _ = np.histogram(importance_scores.flatten(), bins=bins, density=True)
        densities = np.cumsum(densities / num_bins)
        print(f"{method}: {auc(bins[:-1], densities)}")

        axs[3].plot(bins[:-1], densities, color=colors[j], linestyle=linestyles[j], label=method)
    axs[3].set_title(f"Sparsity", pad=pad, fontsize=18)
    axs[3].set_xlabel("Interval Size")
    axs[3].set_xlim([0.0, 1.0])
    axs[3].set_ylabel("MAZ")
    axs[3].set_ylim([0.0, 1.0])
    axs[3].grid()

    fig.text(0.135, 0.01, "(a)", fontsize=18)
    fig.text(0.385, 0.01, "(b)", fontsize=18)
    fig.text(0.63, 0.01, "(c)", fontsize=18)
    fig.text(0.88, 0.01, "(d)", fontsize=18)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.517, 1.15), frameon=True, edgecolor="black")
    
    plt.tight_layout()
    plt.savefig(f"../results/figures/scalability_analysis.pdf", bbox_inches="tight")