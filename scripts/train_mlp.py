import json
import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, feature_dim, num_classes, device=None):
        super(MLP, self).__init__()
        self.device = device if device else torch.device("cpu")

        self.linear1 = nn.Linear(feature_dim, 512)
        self.dropout1 = nn.Dropout(0.7)
        self.linear2 = nn.Linear(512 , 256)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256 , 128)
        self.dropout3 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(128, num_classes)

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
        return x, None, None, None

    def wrapped_forward(self, x):
        p, _, _, _ = self.forward(x)
        return p

    def predict_proba(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        pred, _, _, _ = self.forward(x)
        pred = F.softmax(pred, dim=1).cpu().detach().numpy()
        return pred


def train_one_epoch(epoch, optimizer):
    model.train(True)
    for i, (samples, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        samples = samples.to(device)
        labels = labels.to(device)
        predictions, _, _, _ = model(samples)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch}/{max_epochs}], latest loss: {loss.item():.4f}")


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args = vars(args)
    with open(args["config"], 'r') as cfg:
        config = json.load(cfg)

    set_seed(config["seed"])
    num_seeds = 1
    seed = random.sample(range(1, 10000), num_seeds)[0]
    print(f"Running seed {seed}")
    set_seed(seed)

    x_train, y_train, x_test, y_test = load_data(config["data_path"])

    train_dataset = CustomDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    test_dataset = CustomDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    model = MLP(config["feature_dim"], config["num_classes"]).to(device)

    opt_specs = [
        {"params": model.parameters(), "lr": 1e-3}
    ]
    opt = torch.optim.Adam(opt_specs)

    epoch = 1
    max_epochs = 50

    while epoch <= max_epochs:
        train_one_epoch(epoch, opt)
        epoch += 1

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
    # }, os.path.join(config["save_path"], f"model_mlp.pt"))