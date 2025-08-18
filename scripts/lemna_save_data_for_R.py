import json
import random
import sys
import torch

from scipy import io

from train_mlp import MLP
from train_lstm import LSTMNN
from utils import set_seed, load_data, get_args


device = torch.device("cpu")


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
    
    x_train, y_train, x_test, _ = load_data(config["data_path"])

    if config["dataset"] not in ["nsl_kdd_multi"]:
        model_path = f"../checkpoint/{config['dataset']}/model_mlp.pt"
    else:
        model_path = f"../checkpoint/{config['dataset']}/model_lstm.pt"
    print(f"Loading model {model_path}")
    checkpoint = torch.load(model_path)
    if config["dataset"] not in ["nsl_kdd_multi"]:
        model = MLP(config["feature_dim"], config["num_classes"]).to(device)
    else:
        model = LSTMNN(config["feature_dim"], config["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train(False)

    x_train = torch.from_numpy(x_train)

    softmax = torch.nn.Softmax(dim=1)
    logits, _, _, _ = model(x_train)
    probabilities = softmax(logits).detach().numpy()
    positive_probabilities = probabilities[:, 1]
    positive_probabilities = positive_probabilities.reshape(x_train.shape[0], 1)
    assert positive_probabilities.shape == (x_train.shape[0], 1,)
    io.savemat(f"../data/{config['dataset']}/train.mat", {'X': x_train, 'y': positive_probabilities})

    x_test = torch.from_numpy(x_test)

    softmax = torch.nn.Softmax(dim=1)
    logits, _, _, _ = model(x_test)
    probabilities = softmax(logits).detach().numpy()
    positive_probabilities = probabilities[:, 1]
    positive_probabilities = positive_probabilities.reshape(x_test.shape[0], 1)
    assert positive_probabilities.shape == (x_test.shape[0], 1,)
    io.savemat(f"../data/{config['dataset']}/test.mat", {'X': x_test, 'y': positive_probabilities})
