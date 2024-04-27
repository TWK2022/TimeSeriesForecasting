# 根据diffusion_ts改编:https://github.com/Y-debug-sys/Diffusion-TS
# 多变量异标签
import math
import torch
from model.layer import attention, lgl, split_conv1d, split_linear


class ada_norm(torch.nn.Module):
    def __init__(self, feature):
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(feature, 2 * feature)

    def forward(self, x):  # (batch,dim,feature) -> (batch,dim,feature)
        half_feature = x.shape[2] // 2
        noise = torch.exp(-4 * torch.arange(half_feature, device=x.device) / half_feature).unsqueeze(0)
        noise = noise * torch.randint(0, 500, (x.shape[0], 1), device=x.device)
        noise = torch.concat([noise.sin(), noise.cos()], dim=1)
        noise = self.linear(self.silu(noise)).unsqueeze(1)
        noise_a, noise_b = torch.chunk(noise, 2, dim=2)
        x = x * (1 + noise_a) + noise_b
        return x


class encode_block(torch.nn.Module):
    def __init__(self, head, feature):
        super().__init__()
        self.ada_norm = ada_norm(feature)
        self.attention = attention(head, feature, dropout=0.2)
        self.lgl = lgl(feature, 4)
        self.normalization1 = torch.nn.LayerNorm(feature)
        self.normalization2 = torch.nn.LayerNorm(feature)

    def forward(self, x):  # (batch,dim,feature) -> (batch,dim,feature)
        x1 = self.ada_norm(x)
        x1 = self.attention(x1, x1, x1)
        x = self.normalization1(x + x1)
        x = self.normalization2(x + self.lgl(x))
        return x


class trend_block(torch.nn.Module):
    def __init__(self, feature):
        super().__init__()
        space = torch.arange(1, feature + 1) / (feature + 1)
        self.space1 = space ** 1
        self.space2 = space ** 2
        self.space3 = space ** 3

    def forward(self, x):  # (batch,dim,feature) -> (batch,dim,feature)
        x1 = x * self.space1.to(x.device)
        x2 = x * self.space2.to(x.device)
        x3 = x * self.space3.to(x.device)
        x = x1 + x2 + x3
        return x


class season_block(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  # (batch,dim,feature) -> (batch,dim,feature)
        batch, dim, feature = x.shape
        x = torch.fft.rfft(x, dim=2)  # 傅里叶变换
        x = x[:, :, 1:-1]
        f = torch.fft.rfftfreq(feature)[1:-1].to(x.device)
        f = f.unsqueeze(0).unsqueeze(0).repeat(batch, dim, 1)
        x = torch.cat([x, x.conj()], dim=2).unsqueeze(3)
        f = torch.cat([f, -f], dim=2).unsqueeze(3)
        time = torch.arange(feature, dtype=torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(x.device)
        amp = x.abs()
        phase = x.angle()
        x = amp * torch.cos(2 * math.pi * f * time + phase)
        x = torch.sum(x, dim=2)
        return x


class decode_block(torch.nn.Module):
    def __init__(self, dim, head, feature):
        super().__init__()
        self.ada_norm1 = ada_norm(feature)
        self.attention1 = attention(head, feature, dropout=0.2)
        self.normalization1 = torch.nn.LayerNorm(feature)
        self.ada_norm2 = ada_norm(feature)
        self.attention2 = attention(head, feature, dropout=0.2)
        self.normalization2 = torch.nn.LayerNorm(feature)
        self.conv1d = torch.nn.Conv1d(dim, 2 * dim, kernel_size=1, stride=1)
        self.trend = trend_block(feature)
        self.season = season_block()
        self.lgl = lgl(feature, 2)
        self.normalization3 = torch.nn.LayerNorm(feature)

    def forward(self, x, memory):  # x(batch,dim,feature) -> [3*(batch,dim,feature)]
        x1 = self.ada_norm1(x)
        x1 = self.attention1(x1, x1, x1)
        x = self.normalization1(x + x1)
        x2 = self.ada_norm2(x)
        x2 = self.attention2(x2, memory, memory)
        x = self.normalization2(x + x2)
        x1, x2 = self.conv1d(x).chunk(2, dim=1)
        trend = self.trend(x1)  # (batch,dim,feature)
        season = self.season(x2)  # (batch,dim,feature)
        x = self.normalization3(x + self.lgl(x))  # (batch,dim,feature)
        return x, trend, season


class diffusion_ts(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = len(args.input_column)
        output_dim = len(args.output_column)
        input_size = args.input_size
        output_size = args.output_size
        n_dict = {'s': 32, 'm': 64, 'l': 96}
        dim = n_dict[args.model_type]
        head = 8
        assert output_size % 2 == 0
        # 网络结构
        self.embedding = torch.nn.Conv1d(input_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.encode_position = torch.nn.Parameter(torch.randn(1, dim, input_size))
        self.encode_block1 = encode_block(head, input_size)
        self.encode_block2 = encode_block(head, input_size)
        self.encode_block3 = encode_block(head, input_size)
        self.linear = torch.nn.Linear(input_size, output_size)
        self.decode_position = torch.nn.Parameter(torch.randn(1, output_dim, output_size))
        self.decode_block1 = decode_block(output_dim, head, output_size)
        self.decode_block2 = decode_block(output_dim, head, output_size)
        self.decode_block3 = decode_block(output_dim, head, output_size)
        self.normalization1 = torch.nn.LayerNorm(output_size)
        self.normalization2 = torch.nn.LayerNorm(output_size)
        self.split_conv1d = split_conv1d(3 * output_dim, output_dim)
        self.split_linear = split_linear(output_dim, output_size)

    def forward(self, x):  # (batch,input_dim,input_size) -> (batch,output_dim,output_size)
        x0 = self.embedding(x)  # (batch,dim,input_size)
        x0 = x0 + self.encode_position
        x0 = self.encode_block1(x0)
        x0 = self.encode_block2(x0)
        x0 = self.encode_block3(x0)
        x0 = self.linear(x0)  # (batch,dim,output_size)
        x = self.decode_position.repeat(x.shape[0], 1, 1)  # (batch,output_dim,output_size)
        x, trend1, season1 = self.decode_block1(x, x0)
        x, trend2, season2 = self.decode_block2(x, x0)
        x, trend3, season3 = self.decode_block3(x, x0)
        trend = self.normalization1(trend1 + trend2 + trend3)
        season = self.normalization2(season1 + season2 + season3)
        x = torch.concat([x, trend, season], dim=1)
        x = self.split_conv1d(x)  # (batch,output_dim,output_size)
        x = self.split_linear(x)
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
    model = diffusion_ts(args)
    tensor = torch.randn((4, len(args.input_column), args.input_size), dtype=torch.float32)
    print(model)
    print(model(tensor).shape)
