import json
import pickle
import random
import sys
import torch

import numpy as np

from captum.attr import GuidedGradCam
from timeit import default_timer
from tqdm.contrib import tzip

from train_cnn import CNN
from utils import set_seed, load_data, get_args


device = torch.device("cpu")


def ggc_generate_importance_scores(x, y, ggc_explainer):
    num_samples, num_features = x.shape[0], x.shape[2]
    importance_scores = torch.zeros((num_samples, num_features,))
    for i, (s, p) in enumerate(tzip(x, y)):
        sample = torch.from_numpy(s).unsqueeze(0)
        sample.requires_grad_()
        exp = ggc_explainer.attribute(sample, target=p.item()).detach()
        w = exp.squeeze()
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
    
    x_train, y_train, x_test, _ = load_data(config["data_path"])
    x_train = np.expand_dims(x_train, axis=1)
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=1)
    x_test = np.expand_dims(x_test, axis=3)

    model_path = f"../checkpoint/{config['dataset']}/model_cnn.pt"
    print(f"Loading model {model_path}")
    checkpoint = torch.load(model_path)
    model = CNN(config["feature_dim"], config["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train(False)

    if args["train"]:
        p, _, _, _ = model(torch.from_numpy(x_train))
        pred_train = p.argmax(dim=1).numpy().astype(int)
    else:
        p, _, _, _ = model(torch.from_numpy(x_test))
        pred_test = p.argmax(dim=1).numpy().astype(int)

    ggc_explainer = GuidedGradCam(model, model.conv2)
    ggc_explainer.grad_cam.forward_func = model.wrapped_forward
    ggc_explainer.guided_backprop.forward_func = model.wrapped_forward

    start_time = default_timer()
    if args["train"]:
        importance_scores = ggc_generate_importance_scores(x_train, pred_train, ggc_explainer)
    else:
        importance_scores = ggc_generate_importance_scores(x_test, pred_test, ggc_explainer)
    end_time = default_timer()
    print(f"Latency {end_time - start_time} seconds")

    data = {
        "seed": seed,
        "importance_scores": importance_scores,
    }

    if args["train"]:
        with open(f"../results/{config['dataset']}/importance_scores/ggc_train_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    else:
        with open(f"../results/{config['dataset']}/importance_scores/ggc_seed{seed}.pkl", "wb") as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)