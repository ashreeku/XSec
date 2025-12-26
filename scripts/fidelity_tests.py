import random
import torch


class NotClassLoader(object):
    def __init__(self, x, y, config):
        self.x = x
        self.y = y
        self.class_indices = [torch.where(y != c)[0] for c in range(config["num_classes"])]
    
    def get_sample_not_class(self, c):
        indices = self.class_indices[c]
        idx = random.choice(indices)
        return self.x[idx].clone().detach(), self.y[idx].clone().detach()


def synthetic_test(x, importance_scores, m):
    assert x.shape == importance_scores.shape, "shape mismatch"
    _, num_features = x.shape
    synthetic_samples = []
    for s, w in zip(x, importance_scores):
        sorted_idx = torch.argsort(w, descending=True)
        sorted_idx = sorted_idx[:m]
        synth_s = torch.randn(size=(num_features,))
        synth_s[sorted_idx] = s[sorted_idx]
        synthetic_samples.append(synth_s)
    synthetic_samples = torch.stack((synthetic_samples))
    return synthetic_samples


def feature_augmentation_test(x, y, importance_scores, m, not_class_loader):
    assert x.shape == importance_scores.shape, "shape mismatch"
    augmented_samples = []
    for s, p, w in zip(x, y, importance_scores):
        sorted_idx = torch.argsort(w, descending=True)
        sorted_idx = sorted_idx[:m]
        s_not_y, _ = not_class_loader.get_sample_not_class(p)
        s_not_y[sorted_idx] = s[sorted_idx]
        augmented_samples.append(s_not_y)
    augmented_samples = torch.stack((augmented_samples))
    return augmented_samples


def feature_deduction_test(x, importance_scores, m):
    assert x.shape == importance_scores.shape, "shape mismatch"
    _, num_features = x.shape
    deducted_samples = []
    for s, w in zip(x, importance_scores):
        sorted_idx = torch.argsort(w, descending=True)
        sorted_idx = sorted_idx[:m]
        deducted_s = s.clone().detach()
        random_values = torch.randn(size=(num_features,))
        deducted_s[sorted_idx] = random_values[sorted_idx]
        deducted_samples.append(deducted_s)
    deducted_samples = torch.stack((deducted_samples))
    return deducted_samples