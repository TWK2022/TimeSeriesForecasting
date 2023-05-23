import torch


class cbs(torch.nn.Module):
    def __init__(self, in_, out_, kernel_size, stride):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                                    bias=False)
        self.bn = torch.nn.BatchNorm1d(out_)
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x


class residual(torch.nn.Module):  # in_->in_，len->len
    def __init__(self, in_):
        super().__init__()
        self.cbs0 = cbs(in_, in_, kernel_size=3, stride=1)
        self.cbs1 = cbs(in_, in_, kernel_size=3, stride=1)
        self.cbs2 = cbs(in_, in_, kernel_size=3, stride=1)

    def forward(self, x):
        x0 = self.cbs0(x)
        x0 = self.cbs1(x0)
        x0 = self.cbs2(x0)
        return x + x0


class elan(torch.nn.Module):  # in_->out_，len->len
    def __init__(self, in_, n):
        super().__init__()
        self.cbs0 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
        self.cbs1 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
        self.sequential2 = torch.nn.Sequential(*(cbs(in_ // 2, in_ // 2, kernel_size=3, stride=1) for _ in range(n)))
        self.sequential3 = torch.nn.Sequential(*(cbs(in_ // 2, in_ // 2, kernel_size=3, stride=1) for _ in range(n)))
        self.concat4 = torch.concat
        self.cbs5 = cbs(2 * in_, 2 * in_, kernel_size=1, stride=1)

    def forward(self, x):
        x0 = self.cbs0(x)
        x1 = self.cbs1(x)
        x2 = self.sequential2(x1)
        x3 = self.sequential3(x2)
        x = self.concat4([x0, x1, x2, x3], dim=1)
        x = self.cbs5(x)
        return x


class series_encode(torch.nn.Module):
    def __init__(self, mean_input, std_input):
        super().__init__()
        self.mean_input = mean_input
        self.std_input = std_input

    def forward(self, x):
        for i in range(len(self.mean_input)):
            x[:, i] = (x[:, i] - self.mean_input[i]) / self.std_input[i]
        return x


class series_decode(torch.nn.Module):
    def __init__(self, mean_output, std_output):
        super().__init__()
        self.mean_output = mean_output
        self.std_output = std_output

    def forward(self, x):
        for i in range(len(self.mean_output)):
            x[:, i] = x[:, i] * self.std_output[i] + self.mean_output[i]
        return x


class deploy(torch.nn.Module):
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
