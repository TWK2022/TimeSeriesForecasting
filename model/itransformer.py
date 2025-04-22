# 根据itransformer改编:https://github.com/thuml/iTransformer
import torch
from model.layer import multihead_attention, split_linear


class encode_block(torch.nn.Module):
    def __init__(self, feature, head):
        super().__init__()
        self.attention = multihead_attention(feature, head, dropout=0.2)
        self.conv1d0 = torch.nn.Conv1d(in_channels=feature, out_channels=feature, kernel_size=1)
        self.conv1d1 = torch.nn.Conv1d(in_channels=feature, out_channels=feature, kernel_size=1)
        self.activation = torch.nn.GELU()
        self.normalization0 = torch.nn.LayerNorm(feature)
        self.normalization1 = torch.nn.LayerNorm(feature)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):  # (batch,dim,feature) -> (batch,dim,feature)
        x = self.normalization0(x + self.dropout(self.attention(x, x, x)))
        x1 = x.permute(0, 2, 1)
        x1 = self.conv1d0(x1)
        x1 = self.activation(x1)
        x1 = self.dropout(x1)
        x1 = self.conv1d1(x1)
        x1 = x1.permute(0, 2, 1)
        x = self.normalization1(x + x1)
        return x


class itransformer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_len = args.input_len
        output_len = args.output_len
        input_dim = len(args.input_column)
        output_dim = len(args.output_column)
        n_dict = {'s': 32, 'm': 64, 'l': 128}
        feature = n_dict[args.model_type]
        head = 8
        # 网络结构
        self.l0 = torch.nn.Linear(input_len, feature)
        self.l1 = encode_block(feature, head)
        self.l2 = encode_block(feature, head)
        self.l3 = torch.nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.l4 = split_linear(output_dim, feature, output_len)

    def forward(self, x):  # (batch,input_len,input_feature) -> (batch,output_len,output_feature)
        x = x.permute(0, 2, 1)  # (batch,input_feature,input_len)
        x = self.l0(x)  # (batch,input_feature,feature)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)  # (batch,output_feature,input_len)
        x = self.l4(x)  # (batch,output_feature,output_len)
        x = x.permute(0, 2, 1)  # (batch,output_len,output_feature)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', default=96, type=int)
    parser.add_argument('--output_len', default=24, type=int)
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1,2', type=str)
    parser.add_argument('--model_type', default='m', type=str)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = itransformer(args)
    tensor = torch.randn((4, args.input_len, len(args.input_column)), dtype=torch.float32)
    print(model)
    print(model(tensor).shape)
