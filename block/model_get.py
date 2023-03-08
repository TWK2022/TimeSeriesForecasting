import os
import torch


def model_get(args, data_dict):
    if os.path.exists(args.weight):
        model_dict = torch.load(args.weight, map_location='cpu')
    else:
        choice_dict = {'tsf': 'model_prepare(args,data_dict)._tsf()',
                       'lstm': 'model_prepare(args,data_dict)._lstm()'}
        model = eval(choice_dict[args.model])
        model_dict = {}
        model_dict['model'] = model
        model_dict['val_mse'] = 999
    model_dict['model'](torch.rand(args.batch, len(args.input_column), args.input_size))  # 检查
    return model_dict


class model_prepare(object):
    def __init__(self, args, data_dict):
        self.args = args
        self.data_dict = data_dict

    def _tsf(self):
        from model.tsf import tsf
        model = tsf(self.args, self.data_dict)
        return model

    def _lstm(self):
        from model.lstm import lstm
        model = lstm(self.args, self.data_dict)
        return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--weight', default='', type=str)
    args = parser.parse_args()
    model_get(args)
