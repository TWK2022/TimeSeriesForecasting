# 多变量
import torch


class linear_multi(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = len(args.input_column)
        self.output_dim = len(args.output_column)
        self.input_size = args.input_size
        self.output_size = args.output_size
        # 网络结构
        self.linear0 = torch.nn.Linear(self.input_size, self.output_size)
        self.conv1 = torch.nn.Conv1d(self.input_dim, self.output_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # 输入(batch,input_dim,input_size)
        series_last = x[:, :, -1:]
        x = x - series_last
        x = self.linear0(x)  # 各dim之间是分开运算的
        x = x + series_last
        x = self.conv1(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_column', default='HUFL,HULL,MUFL,MULL,LUFL,LULL,OT', type=str)
    parser.add_argument('--output_column', default='HUFL,HULL,MUFL,MULL,LUFL,LULL,OT', type=str)
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--output_size', default=64, type=int)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = linear_multi(args).to('cuda')
    print(model)
    tensor = torch.zeros((4, len(args.input_column), args.input_size), dtype=torch.float32).to('cuda')
    print(model(tensor).shape)
