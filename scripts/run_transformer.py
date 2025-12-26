import json
import pickle
import random
import sys
import torch

from timeit import default_timer
from tqdm.contrib import tzip

from train_transformer import Transformer
from utils import set_seed, load_data, get_args


device = torch.device("cpu")


def transformer_generate_importance_scores(x, y, model):
    num_samples, num_features = x.shape
    importance_scores = torch.zeros((num_samples, num_features,))
    for i, (s, p) in enumerate(tzip(x, y)):
        s = s.unsqueeze(dim=0)
        _ = model(s)
        w = model.explain()
        w = w.squeeze()
        importance_scores[i] = w
    return importance_scores


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args = vars(args)
    with open(args["config"], 'r') as cfg:
        config = json.load(cfg)

    set_seed(config["seed"])
    num_seeds = 2
    seed = random.sample(range(1, 10000), num_seeds)[1 if args["alternate"] else 0]
    print(f"Running seed {seed}")
    set_seed(seed)
    
    x_train, y_train, x_test, y_test = load_data(config["data_path"])
    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    x_test = torch.from_numpy(x_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    model_path = f"../checkpoint/{config['dataset']}/model_transformer.pt"
    print(f"Loading model {model_path}")
    checkpoint = torch.load(model_path)
    model = Transformer(config["embedding_dim"], config["feature_dim"], config["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train(False)

    if args["train"]:
        p, _, _, _ = model(x_train)
        pred_train = p.argmax(dim=1).int()
    else:
        p, _, _, _ = model(x_test)
        pred_test = p.argmax(dim=1).int()

    start_time = default_timer()
    if args["train"]:
        importance_scores = transformer_generate_importance_scores(x_train, pred_train, model)
    else:
        importance_scores = transformer_generate_importance_scores(x_test, pred_test, model)
    end_time = default_timer()
    print(f"Latency {end_time - start_time} seconds")

    data = {
        "seed": seed,
        "importance_scores": importance_scores,
    }

    if args["train"]:
        with open(f"../results/{config['dataset']}/importance_scores/transformer_train_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f"../results/{config['dataset']}/importance_scores/transformer_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)