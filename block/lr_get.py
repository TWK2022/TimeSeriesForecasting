import math
import torch


def adam(regularization, r_value, param, lr, betas):
    if regularization == 'L2':
        optimizer = torch.optim.Adam(param, lr=lr, betas=betas, weight_decay=r_value)
    else:
        optimizer = torch.optim.Adam(param, lr=lr, betas=betas)
    return optimizer


class lr_adjust:
    def __init__(self, args, lr_adjust_index):
        self.loss_last = 0  # 记录上一轮的学习率，初始化为0
        self.lr_adjust_index = lr_adjust_index  # 当前学习率调整次数
        self.lr_adjust_num = args.lr_adjust_num  # 最大学习率调整次数
        self.lr_adjust_threshold = args.lr_adjust_threshold  # 学习率调整阈值
        self.lr_start = args.lr_start  # 初始学习率
        self.lr_end = args.lr_end_ratio * args.lr_start  # 最终学习率

    def __call__(self, optimizer, epoch, loss_now):
        threshold = self.lr_adjust_threshold * self.loss_last  # 当前要大于阈值才会调整
        if epoch <= 3:  # 预热阶段学习率减少为0.1,0.3,0.5
            lr = max(self.lr_start * (0.2 * epoch - 0.1), 0.00001)
        elif epoch == 4:  # 正式训练时第1轮学习率不变
            lr = self.lr_start
        elif loss_now > threshold and self.lr_adjust_index < self.lr_adjust_num:  # 调整学习率
            self.lr_adjust_index += 1
            decay = self.lr_adjust_index / self.lr_adjust_num  # 0-1
            lr = self.lr_end + (self.lr_start - self.lr_end) * math.cos(math.pi / 2 * decay)
        else:  # 当损失下降幅度比较大或调整次数用完时不调整学习率
            lr = optimizer.param_groups[0]['lr']
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
        self.loss_last = loss_now  # 记录上一轮的学习率
        return optimizer
