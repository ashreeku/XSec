# ===============================================
# DO NOT UPLOAD
# ===============================================

import json
import random

import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import *


if __name__ == "__main__":
    with open("../configs/nsl_kdd_multi.json", 'r') as cfg:
        config = json.load(cfg)

    set_seed(config["seed"])
    num_seeds = 1
    seed = random.sample(range(1, 10000), num_seeds)[0]
    print(f"Running seed {seed}")
    set_seed(seed)
    
    x_all, y_all, headers = data_map[config["dataset"]](config["data_path"])

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, shuffle=True, stratify=y_all)

    if config["normalize"]:
        ss = StandardScaler().fit(x_train)
        x_train = ss.transform(x_train)
        x_test = ss.transform(x_test)
    
    np.savez(f"../data/{config['dataset']}/data.npz", x_train=x_train, y_train = y_train, x_test=x_test, y_test=y_test)
    
    x_train_new, y_train_new, x_test_new, y_test_new = load_data(config["data_path"])

    print(x_train[0], x_train_new[0])

    assert np.all(x_train == x_train_new)
    assert np.all(y_train == y_train_new)
    assert np.all(x_test == x_test_new)
    assert np.all(y_test == y_test_new)