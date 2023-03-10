import torch


class series_encode(torch.nn.Module):
    def __init__(self, input_mean, input_std):
        super().__init__()
        self.input_mean = input_mean
        self.input_std = input_std

    def forward(self, x):
        for i in range(len(self.input_mean)):
            x[:, i] = (x[:, i] - self.input_mean[i]) / self.input_std[i]
        return x


class series_decode(torch.nn.Module):
    def __init__(self, output_mean, output_std):
        super().__init__()
        self.output_mean = output_mean
        self.output_std = output_std

    def forward(self, x):
        for i in range(len(self.output_mean)):
            x[:, i] = x[:, i] * self.output_std[i] + self.output_mean[i]
        return x


class cbs(torch.nn.Module):
    def __init__(self, in_, out_, kernel_size, stride):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        self.bn = torch.nn.BatchNorm1d(out_)
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return x


class residual(torch.nn.Module):
    def __init__(self, in_):
        super().__init__()
        self.cbs0 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
        self.cbs1 = cbs(in_ // 2, in_ // 2, kernel_size=3, stride=1)
        self.cbs2 = cbs(in_ // 2, in_, kernel_size=1, stride=1)

    def forward(self, x):
        x0 = self.cbs0(x)
        x0 = self.cbs1(x0)
        x0 = self.cbs2(x0)
        return x + x0


class elan(torch.nn.Module):
    def __init__(self, in_, n):
        super().__init__()
        self.cbs0 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
        self.cbs1 = cbs(in_, in_ // 2, kernel_size=1, stride=1)
        self.sequential2 = torch.nn.Sequential(*(cbs(in_ // 2, in_ // 2, kernel_size=3, stride=1) for i in range(n)))
        self.sequential3 = torch.nn.Sequential(*(cbs(in_ // 2, in_ // 2, kernel_size=3, stride=1) for i in range(n)))
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


class linear_head(torch.nn.Module):
    def __init__(self, in_, out_):
        super().__init__()
        self.flatten0 = torch.nn.Flatten()
        self.Dropout1 = torch.nn.Dropout(0.2)
        self.linear2 = torch.nn.Linear(in_, in_ // 2)
        self.silu3 = torch.nn.SiLU()
        self.Dropout4 = torch.nn.Dropout(0.2)
        self.linear5 = torch.nn.Linear(in_ // 2, out_)

    def forward(self, x):
        x = self.flatten0(x)
        x = self.Dropout1(x)
        x = self.linear2(x)
        x = self.silu3(x)
        x = self.Dropout4(x)
        x = self.linear5(x)
        return x
