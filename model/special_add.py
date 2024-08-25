# 特殊模型
import torch
from model.tsf import tsf


class special_add(torch.nn.Module):
    def __init__(self, args, model=None):
        super().__init__()
        # 网络结构
        self.model = model(args) if model else tsf(args)

    def forward(self, x, special=None):  # (batch,input_dim,input_size) -> (batch,output_dim,output_size)
        series_last = torch.mean(x[:, [1, 2], -5:], dim=2, keepdim=True)
        x = self.model(x)
        x = x + series_last
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1,2', type=str)
    parser.add_argument('--input_size', default=96, type=int)
    parser.add_argument('--output_size', default=24, type=int)
    parser.add_argument('--model_type', default='m', type=str)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = special_add(args)
    tensor = torch.randn((4, len(args.input_column), args.input_size), dtype=torch.float32)
    special = torch.randn((4, 1), dtype=torch.float32)
    print(model)
    print(model(tensor, special).shape)
