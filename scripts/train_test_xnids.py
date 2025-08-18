"""
This file must be run with an environment created using requirements_xnids.txt
"""

import json
import random
import sys

import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from utils import *


def get_model(feature_dim):
    model = Sequential()
    model.add(LSTM(50, input_dim=feature_dim))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


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

    if args["train"]:
        history = model.fit(x_train, y_train, epochs=50, batch_size=256, validation_split=0.0)
        model.save(f"../checkpoint/{config['dataset']}/model_xnids.h5")
    else:
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