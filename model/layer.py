import math
import torch


class llg(torch.nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = torch.nn.Linear(feature_in, feature_out, bias=False)
        self.ln = torch.nn.LayerNorm(feature_out)
        self.gelu = torch.nn.GELU()

    def forward(self, x):  # (batch,dim,feature_in) -> (batch,dim,feature_out)
        x = self.linear(x)
        x = self.ln(x)
        x = self.gelu(x)
        return x


class lgl(torch.nn.Module):
    def __init__(self, feature, n=2):
        super().__init__()
        self.linear0 = torch.nn.Linear(feature, n * feature, bias=False)
        self.gelu1 = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(n * feature, feature, bias=False)

    def forward(self, x):  # (batch,dim,feature) -> (batch,dim,feature)
        x = self.linear0(x)
        x = self.gelu1(x)
        x = self.linear2(x)
        return x


class clg(torch.nn.Module):
    def __init__(self, dim_in, dim_out, feature, kernel_size, stride):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                                      padding=(kernel_size - 1) // 2, bias=False)
        self.ln = torch.nn.LayerNorm(feature)
        self.gelu = torch.nn.GELU()

    def forward(self, x):  # (batch,dim_in,feature) -> (batch,dim_out,feature)
        x = self.conv1d(x)
        x = self.ln(x)
        x = self.gelu(x)
        return x


class attention(torch.nn.Module):  # 基本等同于torch.nn.MultiheadAttention
    def __init__(self, head, feature, dropout=0.2):
        super(attention, self).__init__()
        self.head = head
        self.divisor = math.sqrt(feature // self.head)
        self.query = torch.nn.Linear(feature, feature)
        self.key = torch.nn.Linear(feature, feature)
        self.value = torch.nn.Linear(feature, feature)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(feature, feature)

    def forward(self, query, key, value):  # 3*(batch,dim,feature) -> (batch,dim,feature)。key和value的dim可以与query不同
        batch, dim, feature = query.shape
        _, dim2, _ = key.shape
        query = self.query(query).reshape(batch, dim, self.head, -1).permute(0, 2, 1, 3)  # (batch,head,dim,-1)
        key = self.key(key).reshape(batch, dim2, self.head, -1).permute(0, 2, 3, 1)  # (batch,head,-1,dim2)
        value = self.value(value).reshape(batch, dim2, self.head, -1).permute(0, 2, 1, 3)  # (batch,head,dim2,-1)
        x = torch.matmul(query, key)  # (batch,head,dim,dim2)
        x = x / self.divisor
        x = self.softmax(x)
        x = self.dropout(x)
        x = torch.matmul(x, value)  # (batch,head,dim,-1)
        x = x.permute(0, 2, 1, 3).reshape(batch, dim, feature)  # (batch,dim,feature)
        x = self.linear(x)  # (batch,dim,feature)
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
    def __init__(self, dim, feature):
        super().__init__()
        self.input_dim = dim
        for i in range(self.input_dim):
            exec(f'self.linear{i} = torch.nn.Linear(feature, feature, bias=False)')

    def forward(self, x):  # (batch,dim,feature) -> (batch,dim,feature)
        x_list = []
        for i in range(self.input_dim):
            x_list.append(eval(f'self.linear{i}')(x[:, i, :]))
        x = torch.stack(x_list, dim=1)
        return x


class series_encode(torch.nn.Module):  # 归一化
    def __init__(self, mean_input, std_input):
        super().__init__()
        self.mean_input = torch.tensor(mean_input)
        self.std_input = torch.tensor(std_input)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = (x - self.mean_input.type(x.dtype).to(x.device)) / self.std_input.type(x.dtype).to(x.device)
        x = x.permute(0, 2, 1)
        return x


class series_decode(torch.nn.Module):  # 反归一化
    def __init__(self, mean_output, std_output):
        super().__init__()
        self.mean_output = torch.tensor(mean_output)
        self.std_output = torch.tensor(std_output)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x * self.std_output.type(x.dtype).to(x.device) + self.mean_output.type(x.dtype).to(x.device)
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
