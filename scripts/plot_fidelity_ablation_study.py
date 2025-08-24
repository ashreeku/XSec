import json
import os
import pickle
import random

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import numpy as np

from utils import set_seed


if __name__ == "__main__":
    fig, axs = plt.subplots(nrows=3, ncols=5)
    fig.set_size_inches(25, 9)
    pad = 13

    for i, config_name in enumerate(["pdf.json", "phishing.json", "netflow.json", "bodmas.json", "nsl_kdd_multi.json"]):
        with open(os.path.join("../config", config_name), 'r') as cfg:
            config = json.load(cfg)

        set_seed(config["seed"])
        num_seeds = 1
        seed = random.sample(range(1, 10000), num_seeds)[0]
        
        with open(f"../results/{config['dataset']}/xsec_fidelity_values_seed{seed}_k10_n3_ablationNone.pkl", "rb") as infile:
            noablation = pickle.load(infile)
        with open(f"../results/{config['dataset']}/xsec_fidelity_values_seed{seed}_k10_n3_ablationbinary.pkl", "rb") as infile:
            binary = pickle.load(infile)
        with open(f"../results/{config['dataset']}/xsec_fidelity_values_seed{seed}_k10_n3_ablationdiversity.pkl", "rb") as infile:
            diversity = pickle.load(infile)
        with open(f"../results/{config['dataset']}/xsec_fidelity_values_seed{seed}_k10_n3_ablationcrossentropy.pkl", "rb") as infile:
            crossentropy = pickle.load(infile)
        with open(f"../results/{config['dataset']}/xsec_fidelity_values_seed{seed}_k10_n3_ablationcluster.pkl", "rb") as infile:
            cluster = pickle.load(infile)
        with open(f"../results/{config['dataset']}/xsec_fidelity_values_seed{seed}_k10_n3_ablationsparsity.pkl", "rb") as infile:
            sparsity = pickle.load(infile)

        step = config["evaluation"]["step"]
        x_axis = np.arange(step, config["feature_dim"] + 1, step)

        synth_test_noablation = noablation["synth_test"]
        synth_test_binary = binary["synth_test"]
        synth_test_diversity = diversity["synth_test"]
        synth_test_crossentropy = crossentropy["synth_test"]
        synth_test_cluster = cluster["synth_test"]
        synth_test_sparsity = sparsity["synth_test"]
        
        axs[0][i].plot(x_axis, synth_test_noablation, color="blue", marker='x', linestyle="solid", label="No ablation")
        axs[0][i].plot(x_axis, synth_test_binary, color="darkgreen", marker='o', linestyle="dashed", label="$\mathcal{L}_{bin}$")
        axs[0][i].plot(x_axis, synth_test_diversity, color="red", marker='*', linestyle="dotted", label="$\mathcal{L}_{sim}$")
        axs[0][i].plot(x_axis, synth_test_crossentropy, color="orange", marker='|', linestyle="dashdot", label="$\mathcal{L}_{xe}$")
        axs[0][i].plot(x_axis, synth_test_cluster, color="pink", marker='^', linestyle=(0, (3, 5, 1, 5, 1, 5)), label="$\mathcal{L}_{cls}$")
        axs[0][i].plot(x_axis, synth_test_sparsity, color="brown", marker='s', linestyle=(0, (3, 1, 1, 1, 1, 1)), label="$\mathcal{L}_{spar}$")
        axs[0][i].set_xlim([0, config["feature_dim"]])
        axs[0][i].set_ylim([0, 100])
        axs[0][i].set_xticks(x_axis[::2])
        axs[0][i].grid()

        feat_aug_test_noablation = noablation["feat_aug_test"]
        mask_noablation = ~np.isnan(feat_aug_test_noablation.astype(np.double))
        feat_aug_test_binary = binary["feat_aug_test"]
        mask_binary = ~np.isnan(feat_aug_test_binary.astype(np.double))
        feat_aug_test_diversity = diversity["feat_aug_test"]
        mask_diversity = ~np.isnan(feat_aug_test_diversity.astype(np.double))
        feat_aug_test_crossentropy = crossentropy["feat_aug_test"]
        mask_crossentropy = ~np.isnan(feat_aug_test_crossentropy.astype(np.double))
        feat_aug_test_cluster = cluster["feat_aug_test"]
        mask_cluster = ~np.isnan(feat_aug_test_cluster.astype(np.double))
        feat_aug_test_sparsity = sparsity["feat_aug_test"]
        mask_sparsity = ~np.isnan(feat_aug_test_sparsity.astype(np.double))
        
        if np.any(mask_noablation):
            axs[1][i].plot(x_axis[mask_noablation], feat_aug_test_noablation[mask_noablation], color="blue", marker='x', linestyle="solid", label="No ablation")
        if np.any(mask_binary):
            axs[1][i].plot(x_axis[mask_binary], feat_aug_test_binary[mask_binary], color="darkgreen", marker='o', linestyle="dashed", label="$\mathcal{L}_{bin}$")
        if np.any(mask_diversity):
            axs[1][i].plot(x_axis[mask_diversity], feat_aug_test_diversity[mask_diversity], color="red", marker='*', linestyle="dotted", label="$\mathcal{L}_{sim}$")
        if np.any(mask_crossentropy):
            axs[1][i].plot(x_axis[mask_crossentropy], feat_aug_test_crossentropy[mask_crossentropy], color="orange", marker='|', linestyle="dashdot", label="$\mathcal{L}_{xe}$")
        if np.any(mask_cluster):
            axs[1][i].plot(x_axis[mask_cluster], feat_aug_test_cluster[mask_cluster], color="pink", marker='^', linestyle=(0, (3, 5, 1, 5, 1, 5)), label="$\mathcal{L}_{cls}$")
        if np.any(mask_sparsity):
            axs[1][i].plot(x_axis[mask_sparsity], feat_aug_test_sparsity[mask_sparsity], color="brown", marker='s', linestyle=(0, (3, 1, 1, 1, 1, 1)), label="$\mathcal{L}_{spar}$")
        axs[1][i].set_xlim([0, config["feature_dim"]])
        axs[1][i].set_ylim([0, 100])
        axs[1][i].set_xticks(x_axis[::2])
        axs[1][i].grid()

        feat_deduc_test_noablation = noablation["feat_deduc_test"]
        feat_deduc_test_binary = binary["feat_deduc_test"]
        feat_deduc_test_diversity = diversity["feat_deduc_test"]
        feat_deduc_test_crossentropy = crossentropy["feat_deduc_test"]
        feat_deduc_test_cluster = cluster["feat_deduc_test"]
        feat_deduc_test_sparsity = sparsity["feat_deduc_test"]
        
        axs[2][i].plot(x_axis, feat_deduc_test_noablation, color="blue", marker='x', linestyle="solid", label="No ablation")
        axs[2][i].plot(x_axis, feat_deduc_test_binary, color="darkgreen", marker='o', linestyle="dashed", label="$\mathcal{L}_{bin}$")
        axs[2][i].plot(x_axis, feat_deduc_test_diversity, color="red", marker='*', linestyle="dotted", label="$\mathcal{L}_{sim}$")
        axs[2][i].plot(x_axis, feat_deduc_test_crossentropy, color="orange", marker='|', linestyle="dashdot", label="$\mathcal{L}_{xe}$")
        axs[2][i].plot(x_axis, feat_deduc_test_cluster, color="pink", marker='^', linestyle=(0, (3, 5, 1, 5, 1, 5)), label="$\mathcal{L}_{cls}$")
        axs[2][i].plot(x_axis, feat_deduc_test_sparsity, color="brown", marker='s', linestyle=(0, (3, 1, 1, 1, 1, 1)), label="$\mathcal{L}_{spar}$")
        axs[2][i].set_xlim([0, config["feature_dim"]])
        axs[2][i].set_ylim([0, 100])
        axs[2][i].set_xticks(x_axis[::2])
        axs[2][i].grid()

    axs[0][0].set_ylabel("PCR (%)")
    axs[0][0].set_title(f"PDF Malware Identification", pad=pad, fontsize=18)
    axs[1][0].set_ylabel("PCR (%)")
    axs[2][0].set_xlabel("$|\mathbf{F_{x}}|$")
    axs[2][0].set_ylabel("PCR (%)")
    axs[0][1].set_title(f"Website Phishing Detection", pad=pad, fontsize=18)
    axs[2][1].set_xlabel("$|\mathbf{F_{x}}|$")
    axs[0][2].set_title(f"Network Intrusion Detection", pad=pad, fontsize=18)
    axs[2][2].set_xlabel("$|\mathbf{F_{x}}|$")
    axs[0][3].set_title(f"PE Malware Classification", pad=pad, fontsize=18)
    axs[2][3].set_xlabel("$|\mathbf{F_{x}}|$")
    axs[0][4].set_title(f"Network Attack Classification", pad=pad, fontsize=18)
    axs[2][4].set_xlabel("$|\mathbf{F_{x}}|$")

    rows = ["Synthetic", "Feature Augmentation", "Feature Deduction"]
    for ax, row in zip(axs[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    size=18, xycoords=ax.yaxis.label, textcoords='offset points',
                    ha='right', va='center', rotation=90)

    handles, labels = axs[0][2].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc="upper center", bbox_to_anchor=(0.517, 1.05), frameon=True, edgecolor="black")
    
    plt.tight_layout()
    plt.savefig(f"../results/figures/fidelity_ablation_study.pdf", bbox_inches="tight")