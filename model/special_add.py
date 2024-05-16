# 特殊模型
# 多变量单标签
import torch
from model.itransformer import itransformer


class special_add(torch.nn.Module):
    def __init__(self, args, model=None):
        super().__init__()
        output_size = args.output_size
        # 网络结构
        self.model = model(args) if model else itransformer(args)
        self.l0 = torch.nn.Linear(output_size, output_size, bias=False)
        self.l1 = torch.nn.GELU()
        self.l2 = torch.nn.Linear(output_size, output_size)

    def forward(self, x, special=None):  # (batch,input_dim,input_size) -> (batch,output_dim,output_size)
        x = self.model(x)
        if special == None:
            return x
        else:
            special = special.unsqueeze(1).unsqueeze(2)
            x = x - special
            x = self.l0(x)
            x = self.l1(x)
            x = self.l2(x)
            x = x + special
            return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1', type=str)
    parser.add_argument('--input_size', default=96, type=int)
    parser.add_argument('--output_size', default=24, type=int)
    parser.add_argument('--model_type', default='m', type=str)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = special_add(args)
    tensor = torch.randn((4, len(args.input_column), args.input_size), dtype=torch.float32)
    special = torch.randn(4, dtype=torch.float32)
    print(model)
    print(model(tensor, special).shape)
