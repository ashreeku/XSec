import argparse
import os
import random
import torch

import numpy as np

from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
from torch.utils.data import Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)


def get_args(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, help='path to configuration file')
        parser.add_argument('-n', '--num_similarity_scores', type=int, default=3, help='number of similarity scores to consider while generating explanations')
        parser.add_argument('-k', '--num_prototypes_per_class', type=int, default=10, help='number of prototypes per class')
        parser.add_argument(
            '-a', '--ablation', type=str, default=None,
            choices = {"binary", "cluster", "diversity", "sparsity", "crossentropy"},
            help='ablation used to train the model'
        )
        return parser.parse_args(argv)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def load_data(path):
    data = np.load(os.path.join(path,"data.npz"))
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


class NotClassLoader(object):
    def __init__(self, x, y, config):
        self.x = x
        self.y = y
        self.num_classes = config["num_classes"]
        self.class_indices = [np.where(y != c)[0] for c in range(self.num_classes)]
    
    def get_sample_not_class(self, c):
        indices = self.class_indices[c]
        idx = random.choice(indices)
        return self.x[idx].copy(), self.y[idx].copy()


def evaluate(test_loader, model, device):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            probabilities, _, _, _ = model(x)
            predictions = torch.argmax(probabilities, dim=1)
            y_true.append(y)
            y_pred.append(predictions)
    y_true = torch.cat((y_true)).numpy()
    y_pred = torch.cat((y_pred)).cpu().numpy()
    return y_true, y_pred


class EvaluationScores():
    def __init__(self, gt, pred, num_classes) -> None:
        self.gt = gt
        self.pred = pred
        self.num_classes = num_classes

    def precision(self):
        if self.num_classes == 2:
            return precision_score(self.gt, self.pred, average=None)[1] * 100.0
        else:
            return precision_score(self.gt, self.pred, average="macro") * 100.0

    def recall(self):
        if self.num_classes == 2:
            return recall_score(self.gt, self.pred, average=None)[1] * 100.0
        else:
            return recall_score(self.gt, self.pred, average="macro") * 100.0

    def accuracy(self):
        return accuracy_score(self.gt, self.pred) * 100.0
    
    def fpr(self):
        if self.num_classes == 2:
            # binary datasets
            cm = confusion_matrix(self.gt, self.pred)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn)
            return fpr * 100.0
        else:
            # multi class datasets
            fpr_per_class = []
            for l in range(self.num_classes):
                y_true = np.array(self.gt) == l
                y_pred = np.array(self.pred) == l
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn)
                fpr_per_class.append(fpr)
            return np.mean(fpr_per_class) * 100.0