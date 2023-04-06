import math


class lr_adjust:
    def __init__(self, lr_adjust_item):
        self.loss_last = 0
        self.lr_adjust_range = [_ for _ in range(50)]  # 学习率调整次数
        self.lr_adjust_item = lr_adjust_item

    def _decay(self, x):  # 学习率最终下降为0.1*lr
        x = 0.1 + 0.9 * math.cos(math.pi / 2 * x)  # x=0-1
        return x

    def __call__(self, optimizer, lr_base, epoch, loss_now):
        if epoch < 3:  # 预热阶段学习率减少为0.2,0.5,0.8
            lr = lr_base * (0.2 + 0.3 * epoch)
        elif epoch < 5:  # 前5-warmup轮学习率不变
            lr = lr_base
        elif loss_now > 0.9 * self.loss_last and self.lr_adjust_item < len(self.lr_adjust_range) - 1:  # 调整学习率
            self.lr_adjust_item += 1
            x = (self.lr_adjust_range[self.lr_adjust_item] + 1) / len(self.lr_adjust_range)
            lr = lr_base * self._decay(x)
        else:  # 当损失下降幅度比较大时暂时不更新学习率
            lr = optimizer.param_groups[0]['lr']
        self.loss_last = loss_now
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
        return optimizer
