"""
This file must be run with an environment created using requirements_xnids.txt
"""

import argparse
import json
import os
import random
import sys

import numpy as np

from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)


def get_args(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, help='path to configuration file')
        return parser.parse_args(argv)


def load_data(path):
    data = np.load(os.path.join(path,"data.npz"))
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def get_model(feature_dim):
    model = Sequential()
    model.add(LSTM(50, input_dim=feature_dim))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


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

    assert config["dataset"] in ["netflow"], "xNIDS is run only on network datasets"

    x_train, y_train, x_test, y_test = load_data(config["data_path"])
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    model = get_model(config["feature_dim"])

    history = model.fit(x_train, y_train, epochs=50, batch_size=256, validation_split=0.0)
    # model.save("../checkpoint/netflow/model_xnids.h5")

    model = keras.models.load_model(f"{config['save_path']}/model_xnids.h5")

    y_pred = model.predict(x_test)
    y_pred = (y_pred >= 0.5).astype(int)
    y_pred = np.squeeze(y_pred)

    performance = EvaluationScores(y_test, y_pred, config["num_classes"])
    acc = performance.accuracy()
    pre = performance.precision()
    rec = performance.recall()
    fpr = performance.fpr()
    print("xNIDS")
    print("========================================================")
    print(f"Test accuracy: {acc}")
    print(f"Test precision: {pre}")
    print(f"Test recall: {rec}")
    print(f"Test FPR: {fpr}")
    print("========================================================\n\n")