import json
import sys
import torch

from torch.utils.data import DataLoader

from xsec import XSec
from train_mlp import MLP
from train_lstm import LSTMNN
from train_cnn import CNN
from utils import *


device = torch.device("cpu")


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

    models = {}
    checkpoint_path = f"{config['save_path']}/model_{seed}.pt"
    checkpoint = torch.load(checkpoint_path)
    model_xsec = XSec(config, device)
    model_xsec.load_state_dict(checkpoint["model_state_dict"])
    model_xsec.train(False)
    models["XSec"] = model_xsec

    if config["dataset"] not in ["bodmas200", "bodmas500"]:
        if config["dataset"] not in ["nsl_kdd_multi", "nsl_kdd"]:
            checkpoint = torch.load(f"{config['save_path']}/model_mlp.pt")
            model_mlp = MLP(config["feature_dim"], config["num_classes"])
            model_mlp.load_state_dict(checkpoint["model_state_dict"])
            model_mlp.train(False)
            models["MLP"] = model_mlp
        else:
            checkpoint = torch.load(f"{config['save_path']}/model_lstm.pt")
            model_lstm = LSTMNN(config["feature_dim"], config["num_classes"])
            model_lstm.load_state_dict(checkpoint["model_state_dict"])
            model_lstm.train(False)
            models["LSTM"] = model_lstm

        checkpoint = torch.load(f"{config['save_path']}/model_cnn.pt")
        model_cnn = CNN(config["feature_dim"], config["num_classes"])
        model_cnn.load_state_dict(checkpoint["model_state_dict"])
        model_cnn.train(False)
        models["CNN"] = model_cnn

    for name, model in models.items():
        x_train, y_train, x_test, y_test = load_data(config["data_path"])

        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

        gt, pred = evaluate(test_loader, model, device)

        performance = EvaluationScores(gt, pred, config["num_classes"])
        acc = performance.accuracy()
        pre = performance.precision()
        rec = performance.recall()
        fpr = performance.fpr()
        print(name)
        print("========================================================")
        print(f"Test accuracy: {acc}")
        print(f"Test precision: {pre}")
        print(f"Test recall: {rec}")
        print(f"Test FPR: {fpr}")
        print("========================================================\n\n")