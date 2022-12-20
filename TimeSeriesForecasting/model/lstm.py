import torch

class lstm(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=args.input_size, hidden_size=args.output_size, num_layers=2, dropout=0.2)

    def forward(self, x):
        x, (h_n, c_n) = self.lstm(x)
        return x


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='l', type=str)
    parser.add_argument('--input_column', default='1,3', type=str)
    parser.add_argument('--output_column', default='1,3', type=str)
    parser.add_argument('--input_size', default=128, type=int)
    parser.add_argument('--output_size', default=16, type=int)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    model = lstm(args).to('cuda')
    print(model)
    tensor = torch.zeros((4, len(args.input_column), args.input_size), dtype=torch.float32).to('cuda')
    print(model(tensor).shape)