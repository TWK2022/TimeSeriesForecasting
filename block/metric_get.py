import torch


def metric(pred, true):
    mae = torch.mean(abs(pred - true))
    mse = torch.mean(torch.square(pred - true))
    return mae, mse
