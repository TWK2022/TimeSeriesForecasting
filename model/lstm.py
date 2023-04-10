import torch


class lstm(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = len(args.input_column)
        self.output_dim = len(args.output_column)
        self.input_size = args.input_size
        self.output_size = args.output_size
        n_dict = {'s': 1, 'm': 2, 'l': 3}
        n = n_dict[args.model_type]
        # 网络结构
        self.lstm0 = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.output_size, num_layers=n, dropout=0.2)
        self.conv1 = torch.nn.Conv1d(self.input_dim, self.output_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # 输入(batch,input_dim,input_size)
        x, (h_n, c_n) = self.lstm0(x)
        x = self.conv1(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='m', type=str)
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1,3', type=str)
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--output_size', default=32, type=int)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = lstm(args).to('cuda')
    print(model)
    tensor = torch.zeros((4, len(args.input_column), args.input_size), dtype=torch.float32).to('cuda')
    print(model(tensor).shape)
