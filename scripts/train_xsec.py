import json
import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from xsec.models import *
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class WeightConstraint(object):
    def __init__(self, max_k):
        self.constraint_type = "L2"
        self.max_k = max_k
        self.iterations = 2

    def __call__(self, module):
        if hasattr(module, 'weight'):
            clipped_w = self.L2Lipschitz(module.weight.data)
            module.weight.data = clipped_w

    def L2Lipschitz(self, w):
        norm = self.max_k
        x = torch.randn(size=(int(w.shape[1]), 1)).to(device)
        for i in range(0, self.iterations): 
            x_p = torch.matmul(w, x)
            x = torch.matmul(torch.t(w), x_p)

        norm = torch.sqrt(torch.sum(torch.matmul(w, x) ** 2) / torch.sum(x ** 2))

        return w * (1.0 / torch.clamp(norm / self.max_k, min=1))


def train_one_epoch(epoch, optimizer):
    model.train(True)
    for i, (samples, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        samples = samples.to(device)
        labels = labels.to(device)
        predictions, distances, _, masks = model(samples)
        criterion = nn.CrossEntropyLoss()
        cross_entropy_loss = criterion(predictions, labels)

        # loss for values in masks not close to either 0 or 1
        non_binary_loss = (torch.norm(masks * (1 - masks), p=2, dim=1) ** 2).sum()

        # pairwise similarity loss between masks to encourage diverse masks
        mask_diversity_loss = F.cosine_similarity(masks[None, :, :], masks[:, None, :], dim=-1).sum()

        # sparsity loss to minimize number of features used in each subfeature
        sparsity_loss = torch.linalg.vector_norm(masks, dim=1).mean()

        prototypes_of_required_class = torch.t(prototype_class_identifier[:, labels])

        # cluster loss
        cluster_loss = torch.mean(torch.min(distances[prototypes_of_required_class].reshape(-1, num_prototypes_per_class), dim=1).values)

        # separation loss
        separation_loss = torch.mean(torch.min(distances[~prototypes_of_required_class].reshape(-1, (num_classes - 1) * num_prototypes_per_class), dim=1).values)
        
        # l1 loss used for last layer finetuning
        l1_mask = torch.t(~prototype_class_identifier).cuda()
        l1_loss = (model.last_layer.linear.weight * l1_mask).norm(p=1)

        loss = coefs["cross_entropy"] * cross_entropy_loss \
            + coefs["cluster"] * cluster_loss \
            + coefs["mask_diversity"] * mask_diversity_loss \
            + coefs["binary_mask"] * non_binary_loss \
            + coefs["sparsity"] * sparsity_loss \
            + coefs["l1"] * l1_loss
            # - coefs["separation"] * separation_loss \

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.encoder.apply(constraint)


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

    train_dataset = CustomDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = CustomDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    constraint = WeightConstraint(max_k=2)

    epoch_pbar = tqdm(total=max_epochs, leave=False)
    epoch = 1

    model = XSec(config, device)
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
    model.maskgen_input.requires_grad = True
    for param in model.mask_generator.parameters():
        param.requires_grad = True
    for param in model.last_layer.parameters():
        param.requires_grad = False

    joint_opt_specs = [
        {'params': model.encoder.parameters(), "lr": lrs["encoder"]},
        {'params': model.prototypes, "lr": lrs["prototypes"]},
        {'params': model.maskgen_input, "lr": lrs["masks"]},
        {'params': model.mask_generator.parameters(), "lr": lrs["masks"]},
    ]
    joint_opt = torch.optim.Adam(joint_opt_specs)

    epoch_pbar.set_description(f"Running joint phase")
    
    while epoch <= config["joint_epochs"]:
        train_one_epoch(epoch, joint_opt)
        epoch += 1
        epoch_pbar.update(1)

    # last layer finetuning phase
    coefs["mask_diversity"] = 0
    coefs["binary_mask"] = 0
    coefs["sparsity"] = 0
    coefs["l1"] = l1_coef
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.prototypes.requires_grad = False
    model.maskgen_input.requires_grad = False
    for param in model.mask_generator.parameters():
        param.requires_grad = False
    for param in model.last_layer.parameters():
        param.requires_grad = True

    last_layer_opt_specs = [
        {'params': model.last_layer.parameters(), "lr": lrs["last_layer"]}
    ]
    last_layer_opt = torch.optim.Adam(last_layer_opt_specs)

    epoch_pbar.set_description(f"Running last layer finetuning phase")
    
    while epoch <= max_epochs:
        train_one_epoch(epoch, last_layer_opt)
        epoch += 1
        epoch_pbar.update(1)

    epoch_pbar.close()

    # testing

    model.train(False)
    gt, pred = evaluate(train_loader, model, device)
    performance = EvaluationScores(gt, pred, config["num_classes"])
    acc = performance.accuracy()
    pre = performance.precision()
    rec = performance.recall()
    fpr = performance.fpr()
    print(f"Train accuracy: {acc}")
    print(f"Train Precision: {pre}")
    print(f"Train Recall: {rec}")
    print(f"Train FPR: {fpr}")

    # testing
    gt, pred = evaluate(test_loader, model, device)
    performance = EvaluationScores(gt, pred, config["num_classes"])
    acc = performance.accuracy()
    pre = performance.precision()
    rec = performance.recall()
    fpr = performance.fpr()
    print(f"Test accuracy: {acc}")
    print(f"Test Precision: {pre}")
    print(f"Test Recall: {rec}")
    print(f"Test FPR: {fpr}")
    
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    # }, os.path.join(config["save_path"], f"model_{seed}_nocrossentropy.pt"))