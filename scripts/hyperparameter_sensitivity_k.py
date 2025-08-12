import json
import sys
import torch


from torch.utils.data import DataLoader

from xsec import XSec
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

    k_to_model = {
        10: f"model_{seed}.pt",
        3: f"model_{seed}_k3.pt",
        5: f"model_{seed}_k5.pt",
        20: f"model_{seed}_k20.pt",
        30: f"model_{seed}_k30.pt",
    }

    for k, model_name in k_to_model.items():
        checkpoint_path = f"{config['save_path']}/{model_name}"
        checkpoint = torch.load(checkpoint_path)
        config["num_prototypes_per_class"] = k
        model = XSec(config, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.train(False)
        _, _, x_test, y_test = load_data(config["data_path"])

        test_dataset = CustomDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

        gt, pred = evaluate(test_loader, model, device)

        performance = EvaluationScores(gt, pred, config["num_classes"])
        acc = performance.accuracy()
        pre = performance.precision()
        rec = performance.recall()
        fpr = performance.fpr()
        print(f"k = {k}")
        print("========================================================")
        print(f"Test accuracy: {acc}")
        print(f"Test precision: {pre}")
        print(f"Test recall: {rec}")
        print(f"Test FPR: {fpr}")
        print("========================================================\n\n")