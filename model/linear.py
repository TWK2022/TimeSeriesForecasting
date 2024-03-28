# 单变量自标签
import torch


class linear(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_size = args.input_size
        output_size = args.output_size
        # 网络结构
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):  # (batch,input_dim,input_size) -> (batch,output_dim,output_size)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1,2,3', type=str)
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--output_size', default=64, type=int)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = linear(args)
    tensor = torch.zeros((4, len(args.input_column), args.input_size), dtype=torch.float32)
    print(model)
    print(model(tensor).shape)
