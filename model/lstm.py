import torch
from model.layer import split_linear


class lstm(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_len = args.input_len
        output_len = args.output_len
        input_feature = len(args.input_column)
        output_feature = len(args.output_column)
        n_dict = {'s': 1, 'm': 2, 'l': 3}
        n = n_dict[args.model_type]
        # 网络结构
        self.l0 = torch.nn.LSTM(input_size=input_len, hidden_size=input_len, num_layers=n, dropout=0.2)
        self.l1 = torch.nn.Linear(input_len, input_len)
        self.l2 = torch.nn.Conv1d(input_feature, output_feature, kernel_size=1)
        self.l3 = split_linear(output_feature, input_len, output_len)

    def forward(self, x):  # (batch,input_len,input_feature) -> (batch,output_len,output_feature)
        x = x.permute(0, 2, 1)  # (batch,input_feature,input_len)
        x, (h_n, c_n) = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)  # (batch,output_feature,input_len)
        x = self.l3(x)  # (batch,output_feature,output_len)
        x = x.permute(0, 2, 1)  # (batch,output_len,output_feature)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='m', type=str)
    parser.add_argument('--input_len', default=96, type=int)
    parser.add_argument('--output_len', default=24, type=int)
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1,2', type=str)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = lstm(args)
    tensor = torch.randn((4, args.input_len, len(args.input_column)), dtype=torch.float32)
    print(model)
    print(model(tensor).shape)
