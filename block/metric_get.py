import torch


def metric(pred, true):  # 0类别为非目标
    mae = torch.mean(abs(pred - true))
    mse = torch.mean(torch.square(pred - true))
    return mae, mse
