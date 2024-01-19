import os
import torch


def model_get(args):
    if os.path.exists(args.weight):
        model_dict = torch.load(args.weight, map_location='cpu')
    else:
        choice_dict = {'linear': 'model_prepare(args).linear()',
                       'linear_conv': 'model_prepare(args).linear_conv()',
                       'lstm': 'model_prepare(args).lstm()',
                       'nlinear': 'model_prepare(args).nlinear()',
                       }
        model = eval(choice_dict[args.model])
        model_dict = {}
        model_dict['model'] = model
        model_dict['epoch'] = 0  # 已训练的轮次
        model_dict['optimizer_state_dict'] = None  # 学习率参数
        model_dict['lr_adjust_index'] = 0  # 学习率调整次数
        model_dict['ema_updates'] = 0  # ema参数
        model_dict['standard'] = 999  # 评价指标
    model_dict['model'](torch.rand(args.batch, len(args.input_column), args.input_size))  # 检查
    return model_dict


class model_prepare:
    def __init__(self, args):
        self.args = args

    def linear(self):
        from model.linear import linear
        model = linear(self.args)
        return model

    def linear_conv(self):
        from model.linear_conv import linear_conv
        model = linear_conv(self.args)
        return model

    def lstm(self):
        from model.lstm import lstm
        model = lstm(self.args)
        return model

    def nlinear(self):
        from model.nlinear import nlinear
        model = nlinear(self.args)
        return model
