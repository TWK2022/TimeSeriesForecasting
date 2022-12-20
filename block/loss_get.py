import torch


def loss_get(args):
    choice_dict = {'mae': 'torch.nn.L1Loss()',
                   'mse': 'torch.nn.MSELoss()'}
    loss = eval(choice_dict[args.loss])
    return loss
