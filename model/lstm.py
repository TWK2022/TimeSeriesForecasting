# 单变量自标签
import torch


class lstm(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = len(args.input_column)
        output_dim = len(args.output_column)
        input_size = args.input_size
        output_size = args.output_size
        n_dict = {'s': 1, 'm': 2, 'l': 3}
        n = n_dict[args.model_type]
        assert input_dim == output_dim, f'! 输入变量要和预测变量一致 !'
        # 网络结构
        self.lstm0 = torch.nn.LSTM(input_size=input_size, hidden_size=output_size, num_layers=n, dropout=0.2)

    def forward(self, x):  # (batch,input_dim,input_size) -> (batch,output_dim,output_size)
        x, (h_n, c_n) = self.lstm0(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='m', type=str)
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1,2,3', type=str)
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--output_size', default=64, type=int)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = lstm(args)
    tensor = torch.zeros((4, len(args.input_column), args.input_size), dtype=torch.float32)
    print(model)
    print(model(tensor).shape)
