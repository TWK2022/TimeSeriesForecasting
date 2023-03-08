import torch


class lstm(torch.nn.Module):
    def __init__(self, args, data_dict):
        super().__init__()
        self.input_mean = data_dict['input_mean']
        self.input_std = data_dict['input_std']
        self.output_mean = data_dict['output_mean']
        self.output_std = data_dict['output_std']
        # 网络结构
        self.series_encode = series_encode(self.input_mean, self.input_std)
        self.lstm = torch.nn.LSTM(input_size=args.input_size, hidden_size=args.output_size, num_layers=2, dropout=0.2)
        self.series_decode = series_decode(self.output_mean, self.output_std)

    def forward(self, x):
        # 输入(batch,input_dim,input_size)
        x = self.series_encode(x)
        x, (h_n, c_n) = self.lstm(x)
        x = self.series_decode(x)
        return x


if __name__ == '__main__':
    import argparse
    import numpy as np
    from layer import series_encode, series_decode

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='l', type=str)
    parser.add_argument('--input_column', default='1,3', type=str)
    parser.add_argument('--output_column', default='1,3', type=str)
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--output_size', default=16, type=int)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    data_dict = {}
    data_dict['input_mean'] = np.ones(len(args.input_column))
    data_dict['input_std'] = np.ones(len(args.input_column))
    data_dict['output_mean'] = np.ones(len(args.output_column))
    data_dict['output_std'] = np.ones(len(args.output_column))
    model = lstm(args, data_dict).to('cuda')
    print(model)
    tensor = torch.zeros((4, len(args.input_column), args.input_size), dtype=torch.float32).to('cuda')
    print(model(tensor).shape)
