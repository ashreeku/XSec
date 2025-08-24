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
            xsec_3 = pickle.load(infile)
        with open(f"../results/{config['dataset']}/xsec_fidelity_values_seed{seed}_k10_n1_ablationNone.pkl", "rb") as infile:
            xsec_1 = pickle.load(infile)
        with open(f"../results/{config['dataset']}/xsec_fidelity_values_seed{seed}_k10_n5_ablationNone.pkl", "rb") as infile:
            xsec_5 = pickle.load(infile)
        with open(f"../results/{config['dataset']}/xsec_fidelity_values_seed{seed}_k10_n7_ablationNone.pkl", "rb") as infile:
            xsec_7 = pickle.load(infile)
        with open(f"../results/{config['dataset']}/xsec_fidelity_values_seed{seed}_k10_n10_ablationNone.pkl", "rb") as infile:
            xsec_10 = pickle.load(infile)

        step = config["evaluation"]["step"]
        x_axis = np.arange(step, config["feature_dim"] + 1, step)

        synth_test_xsec_3 = xsec_3["synth_test"]
        synth_test_xsec_1 = xsec_1["synth_test"]
        synth_test_xsec_5 = xsec_5["synth_test"]
        synth_test_xsec_7 = xsec_7["synth_test"]
        synth_test_xsec_10 = xsec_10["synth_test"]
        
        axs[0][i].plot(x_axis, synth_test_xsec_3, color="blue", marker='x', linestyle="solid", label="n = 3")
        axs[0][i].plot(x_axis, synth_test_xsec_1, color="darkgreen", marker='o', linestyle="dashed", label="n = 1")
        axs[0][i].plot(x_axis, synth_test_xsec_5, color="red", marker='*', linestyle="dotted", label="n = 5")
        axs[0][i].plot(x_axis, synth_test_xsec_7, color="orange", marker='|', linestyle="dashdot", label="n = 7")
        axs[0][i].plot(x_axis, synth_test_xsec_10, color="cadetblue", marker='s', linestyle=(0, (3, 5, 1, 5, 1, 5)), label="n = 10")
        axs[0][i].set_xlim([0, config["feature_dim"]])
        axs[0][i].set_xticks(x_axis[::2])
        axs[0][i].grid()

        feat_aug_test_xsec_3 = xsec_3["feat_aug_test"]
        mask_xsec_3 = ~np.isnan(feat_aug_test_xsec_3)
        feat_aug_test_xsec_1 = xsec_1["feat_aug_test"]
        mask_xsec_1 = ~np.isnan(feat_aug_test_xsec_1)
        feat_aug_test_xsec_5 = xsec_5["feat_aug_test"]
        mask_xsec_5 = ~np.isnan(feat_aug_test_xsec_5)
        feat_aug_test_xsec_7 = xsec_7["feat_aug_test"]
        mask_xsec_7 = ~np.isnan(feat_aug_test_xsec_7)
        feat_aug_test_xsec_10 = xsec_10["feat_aug_test"]
        mask_xsec_10 = ~np.isnan(feat_aug_test_xsec_10)
        
        if np.any(mask_xsec_3):
            axs[1][i].plot(x_axis[mask_xsec_3], feat_aug_test_xsec_3[mask_xsec_3], color="blue", marker='x', linestyle="solid", label="n = 3")
        if np.any(mask_xsec_1):
            axs[1][i].plot(x_axis[mask_xsec_1], feat_aug_test_xsec_1[mask_xsec_1], color="darkgreen", marker='o', linestyle="dashed", label="n = 1")
        if np.any(mask_xsec_5):
            axs[1][i].plot(x_axis[mask_xsec_5], feat_aug_test_xsec_5[mask_xsec_5], color="red", marker='*', linestyle="dotted", label="n = 5")
        if np.any(mask_xsec_7):
            axs[1][i].plot(x_axis[mask_xsec_7], feat_aug_test_xsec_7[mask_xsec_7], color="orange", marker='|', linestyle="dashdot", label="n = 7")
        if np.any(mask_xsec_10):
            axs[1][i].plot(x_axis[mask_xsec_10], feat_aug_test_xsec_10[mask_xsec_10], color="cadetblue", marker='s', linestyle=(0, (3, 5, 1, 5, 1, 5)), label="n = 10")
        axs[1][i].set_xlim([0, config["feature_dim"]])
        axs[1][i].set_xticks(x_axis[::2])
        axs[1][i].grid()

        feat_deduc_test_xsec_3 = xsec_3["feat_deduc_test"]
        feat_deduc_test_xsec_1 = xsec_1["feat_deduc_test"]
        feat_deduc_test_xsec_5 = xsec_5["feat_deduc_test"]
        feat_deduc_test_xsec_7 = xsec_7["feat_deduc_test"]
        feat_deduc_test_xsec_10 = xsec_10["feat_deduc_test"]
        
        axs[2][i].plot(x_axis, feat_deduc_test_xsec_3, color="blue", marker='x', linestyle="solid", label="n = 3")
        axs[2][i].plot(x_axis, feat_deduc_test_xsec_1, color="darkgreen", marker='o', linestyle="dashed", label="n = 1")
        axs[2][i].plot(x_axis, feat_deduc_test_xsec_5, color="red", marker='*', linestyle="dotted", label="n = 5")
        axs[2][i].plot(x_axis, feat_deduc_test_xsec_7, color="orange", marker='|', linestyle="dashdot", label="n = 7")
        axs[2][i].plot(x_axis, feat_deduc_test_xsec_10, color="cadetblue", marker='s', linestyle=(0, (3, 5, 1, 5, 1, 5)), label="n = 10")
        axs[2][i].set_xlim([0, config["feature_dim"]])
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
    plt.savefig(f"../results/figures/fidelity_hyperparameter_sensitivity_n.pdf", bbox_inches="tight")