# 根据SCINet奇偶分离的思想改编
# 预测长度要为输入长度的1/2以下
import torch
from model.layer import cbs


class Splitting(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, :, 1::2], x[:, :, ::2]


class baseblock(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        module = [cbs(input_dim, input_dim, 3, 1),
                  cbs(input_dim, input_dim, 3, 1)]
        self.split = Splitting()
        self.odd = torch.nn.Sequential(*module)
        self.even = torch.nn.Sequential(*module)

    def forward(self, x):
        x_odd, x_even = self.split(x)
        x_odd_mix = torch.mul(self.odd(x_odd), x_even)
        x_even_mix = torch.mul(self.even(x_even), x_odd)
        return x_odd_mix, x_even_mix


class SCINet_Tree(torch.nn.Module):
    def __init__(self, input_dim, input_size):
        super().__init__()
        self.Tree_main = baseblock(input_dim=input_dim)
        self.Tree_odd = baseblock(input_dim=input_dim)
        self.Tree_even = baseblock(input_dim=input_dim)
        self.Tree_odd_odd = baseblock(input_dim=input_dim)
        self.Tree_odd_even = baseblock(input_dim=input_dim)
        self.Tree_even_odd = baseblock(input_dim=input_dim)
        self.Tree_even_even = baseblock(input_dim=input_dim)
        self.stack = torch.stack

    def forward(self, x):
        # 第一层
        x_odd, x_even = self.Tree_main(x)
        # 第二层
        x_odd_odd, x_odd_even = self.Tree_odd(x_odd)
        x_even_odd, x_even_even = self.Tree_even(x_even)
        # 第三层
        x_odd_odd_odd, x_odd_odd_even = self.Tree_odd_odd(x_odd_odd)
        x_odd_even_odd, x_odd_even_even = self.Tree_odd_even(x_odd_even)
        x_even_odd_odd, x_even_odd_even = self.Tree_even_odd(x_even_odd)
        x_even_even_odd, x_even_even_even = self.Tree_even_even(x_even_even)
        # 交叉合并
        merge = self.stack([x_odd_odd_odd, x_odd_odd_even, x_odd_even_odd, x_odd_even_even, x_even_odd_odd,
                            x_even_odd_even, x_even_even_odd, x_even_even_even], dim=3).reshape(x.shape)
        return merge


class tsf(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.input_size % 8 == 0, '输入的长度要为8的倍数'
        dim_dict = {'s': 1, 'm': 2, 'l': 3}
        dim = dim_dict[args.model_type]
        input_dim = len(args.input_column)
        output_dim = len(args.output_column)
        self.cbs = cbs(input_dim, input_dim * dim, 3, 1)
        self.backbone = SCINet_Tree(input_dim=input_dim * dim, input_size=args.input_size)
        self.conv0 = torch.nn.Conv1d(input_dim * dim, output_dim, kernel_size=1, stride=1)
        self.conv1 = torch.nn.Conv1d(args.input_size, args.output_size, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.cbs(x)
        x = self.backbone(x)
        x = self.conv0(x).permute(0, 2, 1)
        x = self.conv1(x).permute(0, 2, 1)
        return x


if __name__ == '__main__':
    import argparse
    from layer import cbs

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='s', type=str)
    parser.add_argument('--input_column', default='1,3', type=str)
    parser.add_argument('--output_column', default='1,3', type=str)
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--output_size', default=16, type=int)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = tsf(args).to('cuda')
    print(model)
    tensor = torch.zeros((4, len(args.input_column), args.input_size), dtype=torch.float32).to('cuda')
    print(model(tensor).shape)
