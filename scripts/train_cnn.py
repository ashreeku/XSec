import json
import os
import sys
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, feature_dim, num_classes, device=None):
        super(CNN, self).__init__()
        self.device = device if device else torch.device("cpu")
        num_flattened_nodes = 32 * ((feature_dim - 4) // 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1))
        self.fc1 = nn.Linear(num_flattened_nodes, 128)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 1))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x, None, None, None

    def wrapped_forward(self, x):
        p, _, _, _ = self.forward(x)
        return p

    def predict_proba(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        logits, _, _, _ = self.forward(x)
        prob = F.softmax(logits, dim=1).cpu().detach().numpy()
        return prob


def train_one_epoch(epoch, optimizer):
    model.train(True)
    for i, (samples, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        samples = samples.to(device)
        labels = labels.to(device)
        logits, _, _, _ = model(samples)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

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
    x_train = np.expand_dims(x_train, axis=1)
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=1)
    x_test = np.expand_dims(x_test, axis=3)

    train_dataset = CustomDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    test_dataset = CustomDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    model = CNN(config["feature_dim"], config["num_classes"]).to(device)

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
    # }, os.path.join(config["save_path"], f"model_cnn.pt"))