import torch


def metric(pred, true):
    mae = torch.mean(abs(pred - true))
    rmse = torch.sqrt(torch.mean(torch.square(pred - true)))
    return mae, rmse
