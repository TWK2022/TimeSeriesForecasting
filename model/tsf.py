# 多变量异标签
import torch
from model.layer import rms_normalization, rotary_position, group_query_attention, split_linear


class encode_block(torch.nn.Module):
    def __init__(self, dim, feature, head):
        super().__init__()
        position = rotary_position(dim, feature // head)
        self.attention = group_query_attention(feature, head, dropout=0.2, position=position)
        self.conv1d1 = torch.nn.Conv1d(in_channels=feature, out_channels=feature, kernel_size=1)
        self.conv1d2 = torch.nn.Conv1d(in_channels=feature, out_channels=feature, kernel_size=1)
        self.activation = torch.nn.GELU()
        self.normalization1 = rms_normalization(feature)
        self.normalization2 = rms_normalization(feature)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):  # (batch,dim,feature) -> (batch,dim,feature)
        x = self.normalization1(x + self.dropout(self.attention(x, x, x)))
        x1 = x.permute(0, 2, 1)
        x1 = self.conv1d1(x1)
        x1 = self.activation(x1)
        x1 = self.dropout(x1)
        x1 = self.conv1d2(x1)
        x1 = x1.permute(0, 2, 1)
        x = self.normalization2(x + x1)
        return x


class tsf(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = len(args.input_column)
        output_dim = len(args.output_column)
        input_size = args.input_size
        output_size = args.output_size
        n_dict = {'s': 128, 'm': 256, 'l': 512}
        feature = n_dict[args.model_type]
        head = 8
        # 网络结构
        self.l0 = torch.nn.Linear(input_size, feature)
        self.l1 = encode_block(input_dim, feature, head)
        self.l2 = encode_block(input_dim, feature, head)
        self.l3 = torch.nn.Linear(feature, output_size)
        self.l4 = torch.nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.l5 = split_linear(output_dim, output_size)

    def forward(self, x):  # (batch,input_dim,input_size) -> (batch,output_dim,output_size)
        x = self.l0(x)  # (batch,input_dim,feature)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)  # (batch,input_dim,output_size)
        x = self.l5(x)  # (batch,output_dim,output_size)
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
    model = tsf(args)
    tensor = torch.randn((4, len(args.input_column), args.input_size), dtype=torch.float32)
    print(model)
    print(model(tensor).shape)
