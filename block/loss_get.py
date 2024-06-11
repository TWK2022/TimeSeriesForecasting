import torch


def loss_get(args):
    choice_dict = {'mae': 'torch.nn.L1Loss()',
                   'mse': 'torch.nn.MSELoss()',
                   'mse_decay': 'mse_decay(args)'}
    loss = eval(choice_dict[args.loss])
    return loss


class mse_decay:  # 使越靠近前面的数值的准确率越重要
    def __init__(self, args):
        self.decay = torch.linspace(1.5, 0.5, args.output_size)
        self.mse = torch.nn.MSELoss()

    def __call__(self, pred, true):
        self.decay = self.decay.to(pred.device)
        pred_decay = pred * self.decay
        true_decay = true * self.decay
        loss = self.mse(pred_decay, true_decay)
        return loss
