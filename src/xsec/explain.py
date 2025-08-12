import torch


def explain(sample, model, n, config):
    prob, _, sim, masks = model(sample)
    pred = torch.argmax(prob).item()
    num_p_per_c = config["num_prototypes_per_class"]
    rel_masks = masks[pred * num_p_per_c: (pred + 1) * num_p_per_c]
    sim = sim.squeeze()
    rel_sim = sim[pred * num_p_per_c: (pred + 1) * num_p_per_c]
    sorted_sim_idx = rel_sim.sort(descending=True).indices[:n]
    rel_sim = rel_sim[sorted_sim_idx].unsqueeze(dim=1)
    rel_masks = rel_masks[sorted_sim_idx, :]
    weighted_masks = rel_masks * rel_sim
    importance_scores = weighted_masks.sum(dim=0)
    return importance_scores