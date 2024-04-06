# 根据nlinear改编:https://github.com/cure-lab/LTSF-Linear
# 多变量异标签
import torch


class nlinear(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = len(args.input_column)
        output_dim = len(args.output_column)
        input_size = args.input_size
        output_size = args.output_size
        # 网络结构
        self.l0 = torch.nn.Linear(input_size, output_size)
        self.l1 = torch.nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=1)

    def forward(self, x):  # (batch,input_dim,input_size) -> (batch,output_dim,output_size)
        series_last = x[:, :, -1:]
        x = x - series_last
        x = self.l0(x)
        x = x + series_last
        x = self.l1(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1,2', type=str)
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--output_size', default=64, type=int)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = nlinear(args)
    tensor = torch.randn((4, len(args.input_column), args.input_size), dtype=torch.float32)
    print(model)
    print(model(tensor).shape)
