import torch


def loss_get(args):
    choice_dict = {'mae': 'torch.nn.L1Loss()',
                   'mse': 'torch.nn.MSELoss()',
                   'mse_decay': 'mse_decay(args)'}
    loss = eval(choice_dict[args.loss])
    return loss


class mse_decay:  # 使越靠近前面的数值的准确率越重要
    def __init__(self, args):
        start = 1.3
        end = 0.7
        self.decay = torch.arange(start, end, -1 / args.output_size)
        self.mse = torch.nn.MSELoss()

    def __call__(self, pred, true):
        device = pred.device
        pred_decay = pred * self.decay.to(device)
        true_decay = true * self.decay.to(device)
        loss = self.mse(pred_decay, true_decay)
        return loss
