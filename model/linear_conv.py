# 多变量异标签
import torch
from model.layer import split_linear, llg, clg


class linear_conv(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = len(args.input_column)
        output_dim = len(args.output_column)
        input_size = args.input_size
        output_size = args.output_size
        # 网络结构
        self.l0 = split_linear(input_dim, input_size)
        self.l1 = llg(input_size, input_size)
        self.l2 = llg(input_size, output_size)
        self.l3 = clg(input_dim, 8 * input_dim, output_size, 1, 1)
        self.l4 = torch.nn.Conv1d(8 * input_dim, output_dim, kernel_size=1, stride=1)

    def forward(self, x):  # (batch,input_dim,input_size) -> (batch,output_dim,output_size)
        series_last = x[:, :, -1:]
        x = x - series_last
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = x + series_last
        x = self.l3(x)
        x = self.l4(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1', type=str)
    parser.add_argument('--input_size', default=512, type=int)
    parser.add_argument('--output_size', default=256, type=int)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = linear_conv(args)
    tensor = torch.zeros((4, len(args.input_column), args.input_size), dtype=torch.float32)
    print(model)
    print(model(tensor).shape)
