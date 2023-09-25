# 根据nlinear改编:https://github.com/cure-lab/LTSF-Linear
# 多变量异标签
import torch
from model.layer import cbs


class nlinear_conv(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = len(args.input_column)
        self.output_dim = len(args.output_column)
        self.input_size = args.input_size
        self.output_size = args.output_size
        # 网络结构
        self.linear0 = torch.nn.Linear(self.input_size, self.input_size, bias=False)
        self.linear1 = torch.nn.Linear(self.input_size, self.input_size, bias=False)
        self.linear2 = torch.nn.Linear(self.input_size, self.input_size, bias=False)
        self.linear3 = torch.nn.Linear(self.input_size, self.input_size, bias=False)
        self.linear4 = torch.nn.Linear(self.input_size, self.output_size, bias=False)
        self.linear5 = torch.nn.Linear(self.input_size, self.output_size, bias=False)
        self.linear6 = torch.nn.Linear(self.input_size, self.output_size, bias=False)
        self.linear7 = torch.nn.Linear(self.input_size, self.output_size, bias=False)
        self.linear8 = torch.nn.Linear(self.output_size, self.output_size, bias=True)
        self.conv9 = cbs(self.input_dim, 4 * self.input_dim, 1, 1)
        self.conv10 = torch.nn.Conv1d(4 * self.input_dim, self.output_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # 输入(batch,input_dim,input_size)
        series_last = x[:, :, -1:]
        x = x - series_last
        x0 = self.linear0(x)
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x3 = self.linear3(x)
        x0 = self.linear4(x0)
        x1 = self.linear5(x1)
        x2 = self.linear6(x2)
        x3 = self.linear7(x3)
        x = x0 + x1 + x2 + x3
        x = self.linear8(x)
        x = x + series_last
        x = self.conv9(x)
        x = self.conv10(x)
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
    model = nlinear_conv(args).to('cuda')
    print(model)
    tensor = torch.zeros((4, len(args.input_column), args.input_size), dtype=torch.float32).to('cuda')
    print(model(tensor).shape)
