# 根据SCINet改编:https://github.com/cure-lab/SCINet
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


class SCINet_Tree3(torch.nn.Module):
    def __init__(self, input_dim):
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
        x = merge + x
        return x


class SCINet_Tree4(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.Tree_main = baseblock(input_dim=input_dim)
        self.Tree_odd = baseblock(input_dim=input_dim)
        self.Tree_even = baseblock(input_dim=input_dim)
        self.Tree_odd_odd = baseblock(input_dim=input_dim)
        self.Tree_odd_even = baseblock(input_dim=input_dim)
        self.Tree_even_odd = baseblock(input_dim=input_dim)
        self.Tree_even_even = baseblock(input_dim=input_dim)
        self.Tree0 = baseblock(input_dim=input_dim)
        self.Tree1 = baseblock(input_dim=input_dim)
        self.Tree2 = baseblock(input_dim=input_dim)
        self.Tree3 = baseblock(input_dim=input_dim)
        self.Tree4 = baseblock(input_dim=input_dim)
        self.Tree5 = baseblock(input_dim=input_dim)
        self.Tree6 = baseblock(input_dim=input_dim)
        self.Tree7 = baseblock(input_dim=input_dim)
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
        # 第四层
        x_0, x_1 = self.Tree0(x_odd_odd_odd)
        x_2, x_3 = self.Tree1(x_odd_odd_even)
        x_4, x_5 = self.Tree2(x_odd_even_odd)
        x_6, x_7 = self.Tree3(x_odd_even_even)
        x_8, x_9 = self.Tree4(x_even_odd_odd)
        x_10, x_11 = self.Tree5(x_even_odd_even)
        x_12, x_13 = self.Tree6(x_even_even_odd)
        x_14, x_15 = self.Tree7(x_even_even_even)
        # 交叉合并
        merge = self.stack([x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15],
                           dim=3).reshape(x.shape)
        x = merge + x
        return x


class scinet(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = len(args.input_column)
        self.output_dim = len(args.output_column)
        self.input_size = args.input_size
        self.output_size = args.output_size
        assert self.input_size % 8 == 0, '输入的长度要为8的倍数'
        n_dict = {'s': 3, 'm': 4}
        n = n_dict[args.model_type]
        # 网络结构
        if n == 3:
            self.backbone0 = SCINet_Tree3(input_dim=self.input_dim)
        elif n == 4:
            self.backbone0 = SCINet_Tree4(input_dim=self.input_dim)
        self.conv1 = torch.nn.Conv1d(self.input_dim, self.output_dim, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv1d(self.input_size, self.output_size, kernel_size=1, stride=1)

    def forward(self, x):
        # 输入(batch,input_dim,input_size)
        x = self.backbone0(x)
        x = self.conv1(x).permute(0, 2, 1)
        x = self.conv2(x).permute(0, 2, 1)
        return x


if __name__ == '__main__':
    import argparse
    from layer import cbs

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='m', type=str)
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1,3', type=str)
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--output_size', default=32, type=int)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = scinet(args).to('cuda')
    print(model)
    tensor = torch.zeros((4, len(args.input_column), args.input_size), dtype=torch.float32).to('cuda')
    print(model(tensor).shape)
