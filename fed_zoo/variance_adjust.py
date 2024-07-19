import numpy as np
import torch


def weight_adjust(init_weight, loss_before_fedavg, loss_after_fedavg, scale=0.01):
    gap_list = []
    for idx in loss_before_fedavg.keys():
        gap_list.append(loss_after_fedavg[idx] - loss_before_fedavg[idx])
    gap_list = torch.from_numpy(np.array(gap_list))
    new_gap = torch.zeros_like(gap_list)
    for idx, gap in enumerate(gap_list):
        if gap <= 0:
            new_gap[idx] = 0
        else:
            new_gap[idx] = gap_list[idx] / torch.max(gap_list)
    for idx, weight in enumerate(init_weight):
        weight = weight + scale * new_gap[idx]
        init_weight[idx] = weight
    new_weight = init_weight / torch.sum(init_weight)
    return new_weight