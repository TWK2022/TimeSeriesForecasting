# 多变量异标签
import torch
from model.layer import cbs, split_linear


class linear_conv(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = len(args.input_column)
        output_dim = len(args.output_column)
        input_size = args.input_size
        output_size = args.output_size
        # 网络结构
        self.split_linear0 = split_linear(input_dim, input_size)
        self.linear1 = torch.nn.Linear(input_size, input_size, bias=True)
        self.linear2 = torch.nn.Linear(input_size, output_size, bias=True)
        self.cbs3 = cbs(input_dim, 16 * input_dim, 1, 1)  # 推荐'4至16 * input_dim'
        self.conv4 = torch.nn.Conv1d(16 * input_dim, output_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # 输入(batch,input_dim,input_size)
        series_last = x[:, :, -1:]
        x = x - series_last
        x = self.split_linear0(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x + series_last
        x = self.cbs3(x)
        x = self.conv4(x)
        return x


if __name__ == '__main__':
    import argparse
    from layer import cbs

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1', type=str)
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--output_size', default=64, type=int)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = linear_conv(args).to('cpu')
    tensor = torch.zeros((4, len(args.input_column), args.input_size), dtype=torch.float32).to('cpu')
    print(model)
    print(model(tensor).shape)
