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


class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        feature_dim = config["feature_dim"]
        self.prototype_dim = config["embedding_dim"]
        self.num_prototypes = config["num_prototypes_per_class"] * config["num_classes"]
        num_flattened_nodes = 32 * ((feature_dim - 4) // 2)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=(3,))
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(3,))
        self.fc1 = nn.Linear(num_flattened_nodes, self.num_prototypes * self.prototype_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, (2,))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = x.reshape((batch_size, self.num_prototypes, self.prototype_dim))
        return x


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
class ProtoPNet(nn.Module):
    def __init__(self, config):
        super(ProtoPNet, self).__init__()
        self.num_classes = config["num_classes"]
        self.num_prototypes_per_class = config["num_prototypes_per_class"]
        num_prototypes = self.num_classes * self.num_prototypes_per_class
        prototype_dim = config["embedding_dim"]

        # define CNN encoder
        self.encoder = ConvNet(config)

        # define prototypes
        self.prototypes = nn.Parameter(
            torch.rand((num_prototypes, prototype_dim)),
            requires_grad=True
        )

        # define and set last layer weights
        self.last_layer = LastLayer(config)

    def calculate_similarity(self, x):
        distances = torch.linalg.vector_norm(x - self.prototypes, ord=2, dim=2) ** 2
        similarities = torch.log((distances + 1) / (distances + 1e-4))
        return distances, similarities

    def forward(self, x):
        encoded = self.encoder(x)

        distances, similarities = self.calculate_similarity(encoded)
        predictions = self.last_layer(similarities)
        return predictions


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


def xsec_generate_importance_scores(x, model, n):
    importance_scores = torch.zeros_like(x)
    for i, s in enumerate(tqdm(x)):
        w = model.explain(s.unsqueeze(0), n=n)
        importance_scores[i] = w
    return importance_scores


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
    x_train_np = np.expand_dims(x_train_np, 1)
    x_test_np = np.expand_dims(x_test_np, 1)

    x_train_np = x_train_np.astype(np.float32)
    x_test_np  = x_test_np.astype(np.float32)
    y_train_np = y_train_np.astype(np.int64)
    y_test_np  = y_test_np.astype(np.int64)

    model_path = f"../checkpoint/{config['dataset']}/model_protopnet.pt"
    print(f"Loading model {model_path}")
    checkpoint = torch.load(model_path)
    model = ProtoPNet(config)
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
        for eps in [0.01, 0.1, 0.3]:
            # FGSM attack
            attack = attack_func(estimator=classifier, eps=eps)

            x_test_adv = attack.generate(x=x_test_np)

            y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            asr = (y_pred_adv != y_test_np).mean()   # attack success rate

            print(f"Attack success rate, eps={eps}: {asr:.5f}")