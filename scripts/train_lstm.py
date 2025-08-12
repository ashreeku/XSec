import json
import os
import sys
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMNN(nn.Module):
    def __init__(self, feature_dim, num_classes, device=None):
        super(LSTMNN, self).__init__()
        self.device = device if device else torch.device("cpu")
        self.lstm1 = nn.LSTM(input_size=feature_dim, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=8, num_layers=1, batch_first=True)
        self.linear = nn.Linear(8, num_classes)

    def forward(self, x):
        batch_size, features = x.shape
        x_reshaped = x.unsqueeze(1)
        output, (_, _) = self.lstm1(x_reshaped)
        _, (hidden, _) = self.lstm2(output)
        output = hidden[-1]
        output = self.linear(output)
        return output, None, None, None

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

    model = LSTMNN(config["feature_dim"], config["num_classes"]).to(device)

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
    # }, os.path.join(config["save_path"], f"model_lstm.pt"))