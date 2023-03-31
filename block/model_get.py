import os
import torch


def model_get(args):
    if os.path.exists(args.weight):
        model_dict = torch.load(args.weight, map_location='cpu')
    else:
        choice_dict = {'lstm': 'model_prepare(args)._lstm()',
                       'scinet': 'model_prepare(args)._scinet()'}
        model = eval(choice_dict[args.model])
        model_dict = {}
        model_dict['model'] = model
        model_dict['epoch'] = 0
        model_dict['optimizer_state_dict'] = None
        model_dict['ema_updates'] = 0
        model_dict['standard'] = 999
    model_dict['model'](torch.rand(args.batch, len(args.input_column), args.input_size))  # 检查
    return model_dict


class model_prepare(object):
    def __init__(self, args):
        self.args = args

    def _lstm(self):
        from model.lstm import lstm
        model = lstm(self.args)
        return model

    def _scinet(self):
        from model.scinet import scinet
        model = scinet(self.args)
        return model
