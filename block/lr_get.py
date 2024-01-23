import math
import torch


def adam(regularization, r_value, param, lr, betas):
    if regularization == 'L2':
        optimizer = torch.optim.Adam(param, lr=lr, betas=betas, weight_decay=r_value)
    else:
        optimizer = torch.optim.Adam(param, lr=lr, betas=betas)
    return optimizer


class lr_adjust:
    def __init__(self, args, step_epoch, epoch_finished):
        self.lr_start = args.lr_start  # 初始学习率
        self.lr_end = args.lr_end_ratio * args.lr_start  # 最终学习率
        self.lr_end_epoch = args.lr_end_epoch  # 最终学习率达到的轮数
        self.step_all = self.lr_end_epoch * step_epoch  # 总调整步数
        self.step_finished = epoch_finished * step_epoch  # 已调整步数
        self.warmup_step = max(5, int(args.warmup_ratio * self.step_all))  # 预热训练步数

    def __call__(self, optimizer):
        self.step_finished += 1
        step_now = self.step_finished
        if step_now <= self.warmup_step:
            lr = self.lr_start * (0.1 + 0.9 * step_now / self.warmup_step)
        else:
            decay = step_now / self.step_all
            lr = self.lr_end + (self.lr_start - self.lr_end) * math.cos(math.pi / 2 * decay)
        lr = max(lr, 0.000001)
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
        return optimizer
