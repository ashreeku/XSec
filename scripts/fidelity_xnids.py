"""
This file must be run with an environment created using requirements_xnids.txt
"""

import json
import pickle
import random
import sys
import torch

import numpy as np

from tensorflow import keras
from tqdm import tqdm

from fidelity_tests import *
from utils import *


device = torch.device("cpu")


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args = vars(args)
    with open(args["config"], 'r') as cfg:
        config = json.load(cfg)

    assert config["dataset"] in ["netflow"], "xNIDS is run only on network datasets"

    set_seed(config["seed"])
    num_seeds = 1
    seed = random.sample(range(1, 10000), num_seeds)[0]
    print(f"Running seed {seed}")
    set_seed(seed)

    model = keras.models.load_model(f"{config['save_path']}/model_xnids.h5")

    _, _, x_test, _ = load_data(config["data_path"])
    num_samples, num_features = x_test.shape

    probabilities = model.predict(np.expand_dims(x_test, axis=1))
    pred_orig = np.squeeze(probabilities >= 0.5).astype(int)

    with open(f"../results/{config['dataset']}/importance_scores/xnids_seed{seed}.pkl", "rb") as infile:
        contents = pickle.load(infile)
    assert contents["seed"] == seed, "Incorrect seed found"
    importance_scores = contents["importance_scores"]
    importance_scores = torch.abs(importance_scores)

    step = config["evaluation"]["step"]
    x_axis = range(step, config["feature_dim"] + 1, step)

    synth_test = []
    for m in tqdm(x_axis, leave=False, desc="Synthetic Test"):
        synthetic_samples = synthetic_test(torch.from_numpy(x_test), importance_scores, m)
        synthetic_samples = np.expand_dims(synthetic_samples.detach().numpy(), axis=1)
        probabilities = model.predict(synthetic_samples)
        pred_new = np.squeeze(probabilities >= 0.5).astype(int)
        count = (pred_new == pred_orig).sum()
        synth_test.append(count / num_samples * 100.0)

    feat_aug_test = []
    not_class_loader = NotClassLoader(torch.from_numpy(x_test), torch.from_numpy(pred_orig), config)
    for m in tqdm(x_axis, leave=False, desc="Feature Augmentation Test"):
        # it is possible that the model predicts all samples to a single class
        # thus leading to errors in the not_class_loader when samples
        try:
            augmented_samples = feature_augmentation_test(torch.from_numpy(x_test), torch.from_numpy(pred_orig), importance_scores, m, not_class_loader)
            augmented_samples = np.expand_dims(augmented_samples.detach().numpy(), axis=1)
            probabilities = model.predict(augmented_samples)
            pred_new = np.squeeze(probabilities >= 0.5).astype(int)
            count = (pred_new == pred_orig).sum()
            feat_aug_test.append(count / num_samples * 100.0)
        except:
            feat_aug_test.append(None)

    feat_deduc_test = []
    for m in tqdm(x_axis, leave=False, desc="Feature Deduction Test"):
        deducted_samples = feature_deduction_test(torch.from_numpy(x_test), importance_scores, m)
        deducted_samples = np.expand_dims(deducted_samples.detach().numpy(), axis=1)
        probabilities = model.predict(deducted_samples)
        pred_new = np.squeeze(probabilities >= 0.5).astype(int)
        count = (pred_new == pred_orig).sum()
        feat_deduc_test.append(count / num_samples * 100.0)

    data = {
        "seed": seed,
        "synth_test": np.array(synth_test),
        "feat_aug_test": np.array(feat_aug_test),
        "feat_deduc_test": np.array(feat_deduc_test),
    }
    
    with open(f"../results/{config['dataset']}/xnids_fidelity_values_seed{seed}.pkl", "wb") as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)