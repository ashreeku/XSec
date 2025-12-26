import json
import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from utils import *


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
        return predictions, distances, similarities, None

    def explain(self, x):
        self.eval()
        self.zero_grad()
        x.requires_grad_(True)

        # Forward through encoder and prototypes
        encoded = self.encoder(x)  # (1, num_prototypes, prototype_dim)
        distance, similarity = self.calculate_similarity(encoded)  # (1, num_prototypes), (1, num_prototypes)
        similarity = similarity.sum()
        similarity.backward()

        feature_importance = x.grad.squeeze().detach().cpu()  # (feature_dim,)

        return feature_importance


def train_one_epoch(epoch, optimizer):
    model.train(True)
    for i, (samples, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        samples = samples.to(device)
        labels = labels.to(device)
        predictions, distances, _, _ = model(samples)
        criterion = nn.CrossEntropyLoss()
        cross_entropy_loss = criterion(predictions, labels)

        prototypes_of_required_class = torch.t(prototype_class_identifier[:, labels])

        # cluster loss
        cluster_loss = torch.mean(torch.min(distances[prototypes_of_required_class].reshape(-1, num_prototypes_per_class), dim=1).values)

        # separation loss
        separation_loss = torch.mean(torch.min(distances[~prototypes_of_required_class].reshape(-1, (num_classes - 1) * num_prototypes_per_class), dim=1).values)
        
        # l1 loss used for last layer finetuning
        l1_mask = torch.t(~prototype_class_identifier).to(device)
        l1_loss = (model.last_layer.linear.weight * l1_mask).norm(p=1)

        loss = coefs["cross_entropy"] * cross_entropy_loss \
            + coefs["cluster"] * cluster_loss \
            + coefs["l1"] * l1_loss
            # - coefs["separation"] * separation_loss \

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch}/{max_epochs}], latest loss: {loss.item():.4f}")


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    args = vars(args)
    with open(args["config"], 'r') as cfg:
        config = json.load(cfg)

    set_seed(config["seed"])
    num_seeds = 1
    seed = random.sample(range(1, 10000), num_seeds)[0]
    print(f"Running seed {seed}")
    set_seed(seed)

    num_classes = config["num_classes"]
    batch_size = config["batch_size"]
    num_prototypes_per_class = config["num_prototypes_per_class"]
    num_prototypes = num_classes * num_prototypes_per_class
    max_epochs = config["joint_epochs"] + config["last_layer_epochs"]
    lrs = config["learning_rates"]
    coefs = config["coefficients"]

    x_train, y_train, x_test, y_test = load_data(config["data_path"])
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    train_dataset = CustomDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = CustomDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    epoch = 1

    model = ProtoPNet(config)
    for params in model.parameters():
        params.register_hook(lambda grad: torch.clamp(grad, -2.0, 2.0))
    
    model = model.to(device)

    # indicates which class each prototype belongs to
    prototype_class_identifier = torch.zeros((num_prototypes, num_classes)).bool().to(device)
    for i in range(num_prototypes_per_class * num_classes):
        prototype_class_identifier[i, i // num_prototypes_per_class] = True

    # joint training phase to learn a meaningful space in the encoder
    l1_coef = coefs["l1"]
    coefs["l1"] = 0
    for param in model.encoder.parameters():
        param.requires_grad = True
    model.prototypes.requires_grad = True
    for param in model.last_layer.parameters():
        param.requires_grad = False

    joint_opt_specs = [
        {'params': model.encoder.parameters(), "lr": lrs["encoder"]},
        {'params': model.prototypes, "lr": lrs["prototypes"]},
    ]
    joint_opt = torch.optim.Adam(joint_opt_specs)

    print(f"Running joint phase")
    
    while epoch <= config["joint_epochs"]:
        train_one_epoch(epoch, joint_opt)
        epoch += 1

    # last layer finetuning phase
    coefs["mask_diversity"] = 0
    coefs["binary_mask"] = 0
    coefs["sparsity"] = 0
    coefs["l1"] = l1_coef
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.prototypes.requires_grad = False
    for param in model.last_layer.parameters():
        param.requires_grad = True

    last_layer_opt_specs = [
        {'params': model.last_layer.parameters(), "lr": lrs["last_layer"]}
    ]
    last_layer_opt = torch.optim.Adam(last_layer_opt_specs)

    print(f"Running last layer finetuning phase")
    
    while epoch <= max_epochs:
        train_one_epoch(epoch, last_layer_opt)
        epoch += 1
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(config["save_path"], "model_protopnet.pt"))

    gt, pred = evaluate(test_loader, model, device)

    performance = EvaluationScores(gt, pred, config["num_classes"])
    acc = performance.accuracy()
    pre = performance.precision()
    rec = performance.recall()
    fpr = performance.fpr()
    print("========================================================")
    print(f"Test accuracy: {acc}")
    print(f"Test precision: {pre}")
    print(f"Test recall: {rec}")
    print(f"Test FPR: {fpr}")
    print("========================================================\n\n")