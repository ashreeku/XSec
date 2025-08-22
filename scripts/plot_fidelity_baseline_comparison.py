import json
import os
import pickle
import random

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import numpy as np

from utils import set_seed


if __name__ == "__main__":
    n = 3
    
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
            xsec = pickle.load(infile)
        with open(f"../results/{config['dataset']}/lime_fidelity_values_seed{seed}.pkl", "rb") as infile:
            lime = pickle.load(infile)
        with open(f"../results/{config['dataset']}/shap_fidelity_values_seed{seed}.pkl", "rb") as infile:
            shap = pickle.load(infile)
        with open(f"../results/{config['dataset']}/ig_fidelity_values_seed{seed}.pkl", "rb") as infile:
            ig = pickle.load(infile)
        with open(f"../results/{config['dataset']}/ggc_fidelity_values_seed{seed}.pkl", "rb") as infile:
            ggc = pickle.load(infile)

        step = config["evaluation"]["step"]
        x_axis = np.arange(step, config["feature_dim"] + 1, step)

        synth_test_xsec = xsec["synth_test"]
        synth_test_lime = lime["synth_test"]
        synth_test_shap = shap["synth_test"]
        synth_test_ig = ig["synth_test"]
        synth_test_ggc = ggc["synth_test"]
        
        axs[0][i].plot(x_axis, synth_test_xsec, color="blue", marker='x', linestyle="solid", label=f"XSec ($n={n}$)")
        axs[0][i].plot(x_axis, synth_test_lime, color="darkgreen", marker='o', linestyle="dashed", label="LIME")
        axs[0][i].plot(x_axis, synth_test_shap, color="red", marker='*', linestyle="dotted", label="SHAP")
        axs[0][i].plot(x_axis, synth_test_ig, color="orange", marker='|', linestyle="dashdot", label="IG")
        axs[0][i].plot(x_axis, synth_test_ggc, color="pink", marker='|', linestyle=(0, (3, 1, 1, 1, 1, 1)), label="GGC")
        axs[0][i].set_xlim([0, config["feature_dim"]])
        axs[0][i].set_ylim([0, 100])
        axs[0][i].set_xticks(x_axis[::2])
        axs[0][i].grid()

        feat_aug_test_xsec = xsec["feat_aug_test"]
        mask_xsec = ~np.isnan(feat_aug_test_xsec)
        feat_aug_test_lime = lime["feat_aug_test"]
        mask_lime = ~np.isnan(feat_aug_test_lime)
        feat_aug_test_shap = shap["feat_aug_test"]
        mask_shap = ~np.isnan(feat_aug_test_shap)
        feat_aug_test_ig = ig["feat_aug_test"]
        mask_ig = ~np.isnan(feat_aug_test_ig)
        feat_aug_test_ggc = ggc["feat_aug_test"]
        mask_ggc = ~np.isnan(feat_aug_test_ggc)
        
        if np.any(mask_xsec):
            axs[1][i].plot(x_axis[mask_xsec], feat_aug_test_xsec[mask_xsec], color="blue", marker='x', linestyle="solid", label=f"XSec (n={n})")
        if np.any(mask_lime):
            axs[1][i].plot(x_axis[mask_lime], feat_aug_test_lime[mask_lime], color="darkgreen", marker='o', linestyle="dashed", label="LIME")
        if np.any(mask_shap):
            axs[1][i].plot(x_axis[mask_shap], feat_aug_test_shap[mask_shap], color="red", marker='*', linestyle="dotted", label="SHAP")
        if np.any(mask_ig):
            axs[1][i].plot(x_axis[mask_ig], feat_aug_test_ig[mask_ig], color="orange", marker='|', linestyle="dashdot", label="IG")
        if np.any(mask_ggc):
            axs[1][i].plot(x_axis[mask_ggc], feat_aug_test_ggc[mask_ggc], color="pink", marker='|', linestyle=(0, (3, 1, 1, 1, 1, 1)), label="GGC")
        axs[1][i].set_xlim([0, config["feature_dim"]])
        axs[1][i].set_ylim([0, 100])
        axs[1][i].set_xticks(x_axis[::2])
        axs[1][i].grid()

        feat_deduc_test_xsec = xsec["feat_deduc_test"]
        feat_deduc_test_lime = lime["feat_deduc_test"]
        feat_deduc_test_shap = shap["feat_deduc_test"]
        feat_deduc_test_ig = ig["feat_deduc_test"]
        feat_deduc_test_ggc = ggc["feat_deduc_test"]
        
        axs[2][i].plot(x_axis, feat_deduc_test_xsec, color="blue", marker='x', linestyle="solid", label=f"XSec (n={n})")
        axs[2][i].plot(x_axis, feat_deduc_test_lime, color="darkgreen", marker='o', linestyle="dashed", label="LIME")
        axs[2][i].plot(x_axis, feat_deduc_test_shap, color="red", marker='*', linestyle="dotted", label="SHAP")
        axs[2][i].plot(x_axis, feat_deduc_test_ig, color="orange", marker='|', linestyle="dashdot", label="IG")
        axs[2][i].plot(x_axis, feat_deduc_test_ggc, color="pink", marker='|', linestyle=(0, (3, 1, 1, 1, 1, 1)), label="GGC")
        axs[2][i].set_xlim([0, config["feature_dim"]])
        axs[2][i].set_ylim([0, 100])
        axs[2][i].set_xticks(x_axis[::2])
        axs[2][i].grid()

    for i, config_name in [(0, "pdf.json"), (1, "phishing.json"), (2, "netflow.json")]:
        with open(os.path.join("../config", config_name), 'r') as cfg:
            config = json.load(cfg)

        set_seed(config["seed"])
        num_seeds = 1
        seed = random.sample(range(1, 10000), num_seeds)[0]
        
        with open(f"../results/{config['dataset']}/lemna_fidelity_values_seed{seed}.pkl", "rb") as infile:
            lemna = pickle.load(infile)

        step = config["evaluation"]["step"]
        x_axis = np.arange(step, config["feature_dim"] + 1, step)

        synth_test_lemna = lemna["synth_test"]
        
        axs[0][i].plot(x_axis, synth_test_lemna, color="brown", marker='|', linestyle=(0, (3, 5, 1, 5, 1, 5)), label="LEMNA")

        feat_aug_test_lemna = lemna["feat_aug_test"]
        mask_lemna = ~np.isnan(feat_aug_test_lemna)
        
        if np.any(mask_lemna):
            axs[1][i].plot(x_axis[mask_lemna], feat_aug_test_lemna[mask_lemna], color="brown", marker='|', linestyle=(0, (3, 5, 1, 5, 1, 5)), label="LEMNA")

        feat_deduc_test_lemna = lemna["feat_deduc_test"]
        
        axs[2][i].plot(x_axis, feat_deduc_test_lemna, color="brown", marker='|', linestyle=(0, (3, 5, 1, 5, 1, 5)), label="LEMNA")

    for i, config_name in [(2, "netflow.json")]:
        with open(os.path.join("../config", config_name), 'r') as cfg:
            config = json.load(cfg)

        set_seed(config["seed"])
        num_seeds = 1
        seed = random.sample(range(1, 10000), num_seeds)[0]
        
        with open(f"../results/{config['dataset']}/xnids_fidelity_values_seed{seed}.pkl", "rb") as infile:
            xnids = pickle.load(infile)

        step = config["evaluation"]["step"]
        x_axis = np.arange(step, config["feature_dim"] + 1, step)

        synth_test_xnids = xnids["synth_test"]
        
        axs[0][i].plot(x_axis, synth_test_xnids, color="cadetblue", marker='*', linestyle=(0, (3, 10, 1, 10, 1, 10)), label="xNIDS")

        feat_aug_test_xnids = xnids["feat_aug_test"]
        mask_xnids = ~np.isnan(feat_aug_test_xnids)
        
        if np.any(mask_xnids):
            axs[1][i].plot(x_axis[mask_xnids], feat_aug_test_xnids[mask_xnids], color="cadetblue", marker='*', linestyle=(0, (3, 10, 1, 10, 1, 10)), label="xNIDS")

        feat_deduc_test_xnids = xnids["feat_deduc_test"]
        
        axs[2][i].plot(x_axis, feat_deduc_test_xnids, color="cadetblue", marker='*', linestyle=(0, (3, 10, 1, 10, 1, 10)), label="xNIDS")

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
    plt.savefig(f"../results/figures/fidelity_baseline_comparison.pdf", bbox_inches="tight")