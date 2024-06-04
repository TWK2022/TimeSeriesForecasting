import os
import torch


def model_get(args):
    if os.path.exists(args.weight):
        model_dict = torch.load(args.weight, map_location='cpu')
        if args.weight_again:
            model_dict['epoch_finished'] = 0  # 已训练的轮数
            model_dict['optimizer_state_dict'] = None  # 学习率参数
            model_dict['ema_updates'] = 0  # ema参数
            model_dict['standard'] = 999  # 评价指标
    else:
        choice_dict = {'crossformer': 'model_prepare(args).crossformer()',
                       'diffusion_ts': 'model_prepare(args).diffusion_ts()',
                       'itransformer': 'model_prepare(args).itransformer()',
                       'linear': 'model_prepare(args).linear()',
                       'linear_conv': 'model_prepare(args).linear_conv()',
                       'lstm': 'model_prepare(args).lstm()',
                       'nlinear': 'model_prepare(args).nlinear()',
                       'pathformer': 'model_prepare(args).pathformer()',
                       'special_add': 'model_prepare(args).special_add()'}
        model = eval(choice_dict[args.model])
        model_dict = {}
        model_dict['model'] = model
        model_dict['epoch_finished'] = 0  # 已训练的轮数
        model_dict['optimizer_state_dict'] = None  # 学习率参数
        model_dict['ema_updates'] = 0  # ema参数
        model_dict['standard'] = 999  # 评价指标
    return model_dict


class model_prepare:
    def __init__(self, args):
        self.args = args

    def crossformer(self):
        from model.crossformer import crossformer
        model = crossformer(self.args)
        return model

    def diffusion_ts(self):
        from model.diffusion_ts import diffusion_ts
        model = diffusion_ts(self.args)
        return model

    def itransformer(self):
        from model.itransformer import itransformer
        model = itransformer(self.args)
        return model

    def linear(self):
        from model.linear import linear
        model = linear(self.args)
        return model

    def lstm(self):
        from model.lstm import lstm
        model = lstm(self.args)
        return model

    def nlinear(self):
        from model.nlinear import nlinear
        model = nlinear(self.args)
        return model

    def pathformer(self):
        from model.pathformer import pathformer
        model = pathformer(self.args)
        return model

    def special_add(self):
        from model.special_add import special_add
        model = special_add(self.args)
        return model
