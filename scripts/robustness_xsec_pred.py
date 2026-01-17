import json
import os
import pickle
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timeit import default_timer
from tqdm import tqdm
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from sklearn.metrics import accuracy_score

from utils import set_seed, load_data, get_args


device = torch.device("cpu")


class PhishingEncoder(nn.Module):
    def __init__(self, config):
        super(PhishingEncoder, self).__init__()
        feature_dim = config["feature_dim"]
        embedding_dim = config["embedding_dim"]
        self.linear1 = nn.Linear(feature_dim, 256)
        self.dropout1 = nn.Dropout(0.7)
        self.linear2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        return x


class PhishingMaskGenerator(nn.Module):
    def __init__(self, config) -> None:
        super(PhishingMaskGenerator, self).__init__()
        self.config = config
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * config["num_prototypes_per_class"]
        self.linear1 = nn.Linear(num_prototypes, 256)
        self.linear2 = nn.Linear(256 , 512)
        self.linear3 = nn.Linear(512 , 1024)
        self.linear4 = nn.Linear(1024, feature_dim * num_prototypes)

    def forward(self, x):
        feature_dim = self.config["feature_dim"]
        num_prototypes = self.config["num_classes"] * self.config["num_prototypes_per_class"]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.sigmoid(x)
        return x.reshape((num_prototypes, feature_dim))


class LastLayer(nn.Module):
    def __init__(self, config):
        super(LastLayer, self).__init__()
        num_classes = config["num_classes"]
        num_prototypes_per_class = config["num_prototypes_per_class"]
        num_prototypes = num_classes * num_prototypes_per_class
        self.linear = nn.Linear(num_prototypes, num_classes, bias=False)
        pos_weight_loc = torch.zeros((num_classes, num_prototypes))
        for c in range(num_classes):
            for i in range(num_prototypes_per_class):
                pos_weight_loc[c, c * num_prototypes_per_class + i] = 1
        neg_weight_loc = 1 - pos_weight_loc
        self.linear.weight.data.copy_(1 * pos_weight_loc - 0.5 * neg_weight_loc)
    
    def forward(self, x):
        return self.linear(x)
    

"""
Slightly modified implementation of the forward pass to be compatible with IBM ART
"""
class XSec(nn.Module):
    def __init__(self, config, device=None):
        super(XSec, self).__init__()
        self.device = device if device else torch.device('cpu')

        self.num_classes = config["num_classes"]
        self.num_prototypes_per_class = config["num_prototypes_per_class"]
        num_prototypes = self.num_classes * self.num_prototypes_per_class
        embedding_dim = config["embedding_dim"]

        # define all models
        self.mask_generator = PhishingMaskGenerator(config)
        self.encoder = PhishingEncoder(config)

        # define masks
        self.maskgen_input = nn.Parameter(
            torch.rand((num_prototypes,)),
            requires_grad=True
        )

        # define prototypes
        self.prototypes = nn.Parameter(
            torch.rand((num_prototypes, embedding_dim)),
            requires_grad=True
        )

        # define and set last layer weights
        self.last_layer = LastLayer(config)

    def calculate_similarity(self, x):
        distances = torch.linalg.vector_norm(x - self.prototypes, ord=2, dim=2) ** 2
        similarities = torch.log((distances + 1) / (distances + 1e-4))
        return distances, similarities

    def forward(self, x):
        masks = self.mask_generator(self.maskgen_input)
        sub_features = x.unsqueeze(1) * masks
        encoded = self.encoder(sub_features)

        distances, similarities = self.calculate_similarity(encoded)
        predictions = self.last_layer(similarities)
        return predictions

    def explain(self, x, n):
        k = self.num_prototypes_per_class
        prob, _, sim, masks = self.forward(x)
        pred = torch.argmax(prob).item()
        rel_masks = masks[pred * k: (pred + 1) * k]
        sim = sim.squeeze()
        rel_sim = sim[pred * k: (pred + 1) * k]
        sorted_sim_idx = rel_sim.sort(descending=True).indices[:n]
        rel_sim = rel_sim[sorted_sim_idx].unsqueeze(dim=1)
        rel_masks = rel_masks[sorted_sim_idx, :]
        weighted_masks = rel_masks * rel_sim
        importance_scores = weighted_masks.sum(dim=0)
        return importance_scores


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


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args = vars(args)
    with open("../config/phishing.json", 'r') as cfg:
        config = json.load(cfg)

    set_seed(config["seed"])
    num_seeds = 2
    seed = random.sample(range(1, 10000), num_seeds)[1 if args["alternate"] else 0]
    print(f"Running seed {seed}")
    set_seed(seed)

    x_train_np, y_train_np, x_test_np, y_test_np = load_data(config["data_path"])

    x_train_np = x_train_np.astype(np.float32)
    x_test_np  = x_test_np.astype(np.float32)
    y_train_np = y_train_np.astype(np.int64)
    y_test_np  = y_test_np.astype(np.int64)

    n = args["num_similarity_scores"]
    model_path, config = get_model_path(args, config)
    print(f"Loading model {model_path}")
    checkpoint = torch.load(model_path)
    model = XSec(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train(False)

    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=(config["feature_dim"]),
        nb_classes=config["num_classes"],
        clip_values=(x_train_np.min(), x_train_np.max()),
    )

    # Clean accuracy
    y_pred_clean = np.argmax(classifier.predict(x_test_np), axis=1)
    acc_clean = accuracy_score(y_test_np, y_pred_clean)
    print(f"Clean accuracy: {acc_clean:.5f}")

    attacks = {"FGSM": FastGradientMethod, "PGD": ProjectedGradientDescent}

    for attack, attack_func in attacks.items():
        print(attack)
        for eps in [0.01, 0.1, 0.3, 0.5, 1.0]:
            # FGSM attack
            attack = attack_func(estimator=classifier, eps=eps)

            x_test_adv = attack.generate(x=x_test_np)

            y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            asr = (y_pred_adv != y_test_np).mean()   # attack success rate

            print(f"Attack success rate, eps={eps}: {asr:.5f}")