import math


class lr_adjust:
    def __init__(self, args, lr_adjust_item):
        self.loss_last = 0
        self.lr_adjust_item = lr_adjust_item  # 当前学习率调整次数
        self.lr_adjust_num = args.lr_adjust_num  # 最大学习率调整次数
        self.lr_start = args.lr_start  # 初始学习率
        self.lr_end = args.lr_end  # 最终学习率

    def __call__(self, optimizer, epoch, loss_now):
        if epoch < 3:  # 预热阶段学习率减少为0.1,0.3,0.5
            lr = self.lr_start * (0.1 + 0.2 * epoch)
        elif epoch == 3:  # 正式训练时第1轮学习率不变
            lr = self.lr_start
        elif loss_now > 0.9 * self.loss_last and self.lr_adjust_item < self.lr_adjust_num:  # 调整学习率
            self.lr_adjust_item += 1
            decay = self.lr_adjust_item / self.lr_adjust_num  # 0-1
            lr = self.lr_end + (self.lr_start - self.lr_end) * math.cos(math.pi / 2 * decay)
        else:  # 当损失下降幅度比较大时暂时不调整学习率
            lr = optimizer.param_groups[0]['lr']
        self.loss_last = loss_now
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
        return optimizer
