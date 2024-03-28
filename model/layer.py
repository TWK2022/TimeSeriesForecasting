import torch


class llg(torch.nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.linear = torch.nn.Linear(in_, out_, bias=False)
        self.ln = torch.nn.LayerNorm(out_)
        self.gelu = torch.nn.GELU()

    def forward(self, x):  # (batch,dim,in_) -> (batch,dim,out_)
        x = self.linear(x)
        x = self.ln(x)
        x = self.gelu(x)
        return x


class clg(torch.nn.Module):
    def __init__(self, in_, out_, feature, kernel_size, stride):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                                      bias=False)
        self.ln = torch.nn.LayerNorm(feature)
        self.gelu = torch.nn.GELU()

    def forward(self, x):  # (batch,in_,feature) -> (batch,out_,feature)
        x = self.conv1d(x)
        x = self.ln(x)
        x = self.gelu(x)
        return x


class concat(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.concat = torch.concat
        self.dim = dim

    def forward(self, x):
        x = self.concat(x, dim=self.dim)
        return x


class split_linear(torch.nn.Module):
    def __init__(self, in_, feature):
        super().__init__()
        self.input_dim = in_
        for i in range(self.input_dim):
            exec(f'self.linear{i} = torch.nn.Linear(feature, feature, bias=False)')

    def forward(self, x):  # (batch,in_,feature) -> (batch,in_,feature)
        x_list = []
        for i in range(self.input_dim):
            x_list.append(eval(f'self.linear{i}')(x[:, i, :]))
        x = torch.stack(x_list, dim=1)
        return x


class series_encode(torch.nn.Module):  # 归一化
    def __init__(self, mean_input, std_input):
        super().__init__()
        self.mean_input = mean_input
        self.std_input = std_input

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = (x - self.mean_input) / self.std_input
        x = x.permute(0, 2, 1)
        return x


class series_decode(torch.nn.Module):  # 反归一化
    def __init__(self, mean_output, std_output):
        super().__init__()
        self.mean_output = mean_output
        self.std_output = std_output

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x * self.std_output + self.mean_output
        x = x.permute(0, 2, 1)
        return x


class deploy(torch.nn.Module):  # 对输入进行归一化，对输出进行反归一化
    def __init__(self, model, mean_input, mean_output, std_input, std_output):
        super().__init__()
        self.series_encode = series_encode(mean_input, std_input)
        self.model = model
        self.series_decode = series_decode(mean_output, std_output)

    def forward(self, x):
        x = self.series_encode(x)
        x = self.model(x)
        x = self.series_decode(x)
        return x
