import math
import torch


class rms_normalization(torch.nn.Module):
    def __init__(self, feature):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(feature))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6).type_as(x)
        x = x * self.weight
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


class llg(torch.nn.Module):
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = torch.nn.Linear(feature_in, feature_out, bias=False)
        self.ln = rms_normalization(feature_out)
        self.gelu = torch.nn.GELU()

    def forward(self, x):  # (batch,dim,feature_in) -> (batch,dim,feature_out)
        x = self.linear(x)
        x = self.ln(x)
        x = self.gelu(x)
        return x


class clg(torch.nn.Module):
    def __init__(self, dim_in, dim_out, feature, kernel_size, stride):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                                      padding=(kernel_size - 1) // 2, bias=False)
        self.ln = rms_normalization(feature)
        self.gelu = torch.nn.GELU()

    def forward(self, x):  # (batch,dim_in,feature) -> (batch,dim_out,feature)
        x = self.conv1d(x)
        x = self.ln(x)
        x = self.gelu(x)
        return x


class position(torch.nn.Module):
    def __init__(self, dim, feature):
        super().__init__()
        assert feature % 2 == 0
        dim_index = torch.arange(0, dim).unsqueeze(1).repeat(1, feature // 2)
        feature_index = torch.arange(0, feature, 2)
        half = dim_index / torch.pow(10000, (feature_index / feature))
        self.position = torch.zeros(dim, feature, dtype=torch.float32)
        self.position[:, 0::2] = torch.sin(half)
        self.position[:, 1::2] = torch.cos(half)

    def forward(self, x):  # (batch,dim,feature) -> (batch,dim,feature)
        x = x + self.position.type(x.dtype).to(x.device)
        return x


class rotary_position(torch.nn.Module):
    def __init__(self, dim, feature):
        super().__init__()
        assert feature % 2 == 0
        dim_index = torch.arange(0, dim).unsqueeze(1).repeat(1, feature // 2)
        feature_index = torch.arange(0, feature, 2)
        half = dim_index / torch.pow(10000, (feature_index / feature))
        merge = torch.concat([half, half], dim=1)
        self.position_sin = torch.sin(merge)
        self.position_cos = torch.cos(merge)

    def forward(self, x):  # (batch,dim,feature) -> (batch,dim,feature)
        x_sin = torch.concat([-x[:, :, x.shape[2] // 2:], x[:, :, :x.shape[2] // 2]], dim=2)
        x = x_sin * self.position_sin.type(x.dtype).to(x.device) + x * self.position_cos.type(x.dtype).to(x.device)
        return x


class attention(torch.nn.Module):  # 基本等同于torch.nn.MultiheadAttention
    def __init__(self, feature, head=8, bias=False, dropout=0.2, position=None):
        super().__init__()
        assert feature % head == 0
        self.head = head
        self.divisor = math.sqrt(feature // head)
        self.query = torch.nn.Linear(feature, feature, bias=False)
        self.key = torch.nn.Linear(feature, feature, bias=False)
        self.value = torch.nn.Linear(feature, feature, bias=False)
        self.position = position
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(feature, feature, bias=bias)

    def forward(self, query, key, value):  # 3*(batch,dim,feature) -> (batch,dim,feature)。key和value的dim可以与query不同
        batch, dim, feature = query.shape
        _, dim2, _ = key.shape
        query = self.query(query).reshape(batch, dim, self.head, -1).permute(0, 2, 1, 3)  # (batch,head,dim,-1)
        key = self.key(key).reshape(batch, dim2, self.head, -1).permute(0, 2, 1, 3)  # (batch,head,dim2,-1)
        value = self.value(value).reshape(batch, dim2, self.head, -1).permute(0, 2, 1, 3)  # (batch,head,dim2,-1)
        if self.position is not None:
            query = self.position(query)
            key = self.position(key)
        x = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch,head,dim,dim2)
        x = x / self.divisor
        x = self.softmax(x)
        x = self.dropout(x)
        x = torch.matmul(x, value)  # (batch,head,dim,-1)
        x = x.permute(0, 2, 1, 3).reshape(batch, dim, feature)  # (batch,dim,feature)
        x = self.linear(x)  # (batch,dim,feature)
        return x


class group_query_attention(torch.nn.Module):
    def __init__(self, feature, head=8, group=4, bias=False, dropout=0.2, position=None):
        super().__init__()
        assert feature % head == 0
        assert head % group == 0
        self.head = head
        self.group = group
        self.divisor = math.sqrt(feature // head)
        self.query = torch.nn.Linear(feature, feature, bias=False)
        self.key = torch.nn.Linear(feature, feature // group, bias=False)
        self.value = torch.nn.Linear(feature, feature // group, bias=False)
        self.position = position
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(feature, feature, bias=bias)

    def forward(self, query, key, value):  # 3*(batch,dim,feature) -> (batch,dim,feature)。key和value的dim可以与query不同
        batch, dim, feature = query.shape
        _, dim2, _ = key.shape
        query = self.query(query).reshape(batch, dim, self.head, -1).permute(0, 2, 1, 3)  # (batch,head,dim,-1)
        head_group = self.head // self.group
        key = self.key(key).reshape(batch, dim2, head_group, -1).permute(0, 2, 1, 3)  # (batch,head_group,dim2,-1)
        value = self.value(value).reshape(batch, dim2, head_group, -1).permute(0, 2, 1, 3)  # (batch,head_group,dim2,-1)
        key = key.repeat(1, self.group, 1, 1)  # (batch,head,dim2,-1)
        value = value.repeat(1, self.group, 1, 1)  # (batch,head,dim2,-1)
        if self.position is not None:
            query = self.position(query)
            key = self.position(key)
        x = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch,head,dim,dim2)
        x = x / self.divisor
        x = self.softmax(x)
        x = self.dropout(x)
        x = torch.matmul(x, value)  # (batch,head,dim,-1)
        x = x.permute(0, 2, 1, 3).reshape(batch, dim, feature)  # (batch,dim,feature)
        x = self.linear(x)  # (batch,dim,feature)
        return x


class split_linear(torch.nn.Module):
    def __init__(self, dim, feature):
        super().__init__()
        self.dim = dim
        for i in range(self.dim):
            exec(f'self.linear{i} = torch.nn.Linear(feature, feature, bias=True)')

    def forward(self, x):  # (batch,dim,feature) -> (batch,dim,feature)
        x_list = []
        for i in range(self.dim):
            x_list.append(eval(f'self.linear{i}')(x[:, i, :]))
        x = torch.stack(x_list, dim=1)
        return x


class series_encode(torch.nn.Module):  # 归一化
    def __init__(self, mean_input, std_input, mean_special=None, std_special=None):
        super().__init__()
        self.mean_input = torch.tensor(mean_input)
        self.std_input = torch.tensor(std_input)
        if mean_special is not None:
            self.mean_special = torch.tensor(mean_special)
            self.std_special = torch.tensor(std_special)

    def forward(self, x, special=None):
        x = x.permute(0, 2, 1)
        x = (x - self.mean_input.type(x.dtype).to(x.device)) / self.std_input.type(x.dtype).to(x.device)
        x = x.permute(0, 2, 1)
        if special is None:
            return x
        else:
            special = ((special - self.mean_special.type(x.dtype).to(x.device)) / self.std_special.type(x.dtype)
                       .to(x.device))
            return x, special


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
    def __init__(self, model, mean_input, mean_output, std_input, std_output, mean_special=None, std_special=None):
        super().__init__()
        self.series_encode = series_encode(mean_input, std_input, mean_special, std_special)
        self.model = model
        self.series_decode = series_decode(mean_output, std_output)

    def forward(self, x, special=None):
        if special is None:
            x = self.series_encode(x)
            x = self.model(x)
        else:
            x, special = self.series_encode(x, special)
            x = self.model(x, special)
        x = self.series_decode(x)
        return x
