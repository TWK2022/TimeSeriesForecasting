# 根据pathformer改编:https://github.com/decisionintelligence/pathformer
# 多变量异标签
import math
import torch
import numpy as np
from model.layer import attention, split_linear


class AMS(torch.nn.Module):
    def __init__(self, input_size, input_dim, feature, layer_number=1):
        super().__init__()
        self.start_linear = torch.nn.Linear(input_dim, 1)
        self.seasonality_model = seasonality_model()
        self.trend_model = trend_model(kernel_size=[4, 8, 12])
        self.transformer_list = torch.nn.ModuleList()
        for patch in [16, 12, 8]:
            patch_nums = input_size // patch
            self.transformer_list.append(Transformer_Layer(feature=feature, input_dim=input_dim, patch_nums=patch_nums,
                                                           patch_size=patch, layer_number=layer_number))
        self.w_gate = torch.nn.Parameter(torch.zeros(input_size, 3))
        self.w_noise = torch.nn.Parameter(torch.zeros(input_size, 3))
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        x0 = self._seasonality_and_trend_decompose(x)  # 季节和趋势分析
        gates, load = self._noisy_top_k_gating(x0)  # multi-scale router
        dispatcher = SparseDispatcher(3, gates)
        expert_inputs = dispatcher.dispatch(x)
        x_list = []
        for index in range(3):
            if expert_inputs[index].shape[0] > 0:  # 可能为0
                x_list.append(self.transformer_list[index](expert_inputs[index]))
        x = dispatcher.combine(x_list)
        return x

    def _seasonality_and_trend_decompose(self, x):
        x = x[:, :, :, 0]
        trend = self.trend_model(x)
        seasonality = self.seasonality_model(x)
        return x + seasonality + trend

    def _noisy_top_k_gating(self, x):
        x = self.start_linear(x).squeeze(-1)
        clean_logits = x @ self.w_gate
        if self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + 0.01
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(3, dim=1)
        top_k_logits = top_logits[:, :2]
        top_k_indices = top_indices[:, :2]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.training:
            batch = clean_logits.size(0)
            m = top_logits.size(1)
            top_values_flat = top_logits.flatten()
            threshold_positions_if_in = torch.arange(batch, device=clean_logits.device) * m + 2
            threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
            is_in = torch.gt(noisy_logits, threshold_if_in)
            threshold_positions_if_out = threshold_positions_if_in - 1
            threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
            normal = torch.distributions.normal.Normal(0, 1)  # 正态分布函数，前者为均值，后者为标准差
            prob_if_in = normal.cdf((clean_logits - threshold_if_in) / noise_stddev)  # 分别计算正态分布中小于每个值的概率
            prob_if_out = normal.cdf((clean_logits - threshold_if_out) / noise_stddev)
            load = torch.where(is_in, prob_if_in, prob_if_out).sum(dim=0)
        else:
            load = (gates > 0).sum(dim=0)
        return gates, load


class SparseDispatcher:
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(dim=0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out):
        stitched = torch.cat(expert_out, 0).exp()
        stitched = torch.einsum("ijkh,ik -> ijkh", stitched, self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3),
                            requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()


class seasonality_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.low_freq = 1

    def forward(self, x):  # (b, t, d)
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]
        x_freq, index_tuple = self.topk_freq(x_freq)
        f = f.unsqueeze(0).unsqueeze(2).repeat(x_freq.shape[0], 1, x_freq.shape[2]).to(x_freq.device)
        f = f[index_tuple].unsqueeze(2)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = torch.arange(t, dtype=torch.float).unsqueeze(0).unsqueeze(1).unsqueeze(3).to(x_freq.device)
        amp = (x_freq.abs() / t).unsqueeze(2)
        phase = x_freq.angle().unsqueeze(2)
        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)
        return torch.sum(x_time, dim=1).squeeze(1)

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), 3, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.shape[0]), torch.arange(x_freq.shape[2]))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


class trend_model(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * torch.nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        return moving_mean


class moving_avg(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class Transformer_Layer(torch.nn.Module):
    def __init__(self, feature, input_dim, patch_nums, patch_size, layer_number):
        super().__init__()
        self.feature = feature
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.layer_number = layer_number
        self.intra_embeddings = torch.nn.Parameter(torch.rand(self.patch_nums, 1, 1, input_dim, 16))
        self.embeddings_generator = torch.nn.ModuleList([torch.nn.Sequential(*[
            torch.nn.Linear(16, self.feature)]) for _ in range(self.patch_nums)])
        self.intra_feature = self.feature
        self.intra_patch_attention = Intra_Patch_Attention(self.intra_feature)
        self.weights_generator_distinct = WeightGenerator(self.intra_feature, self.intra_feature, mem_dim=16,
                                                          input_dim=input_dim, factorized=True, number_of_weights=2)
        self.weights_generator_shared = WeightGenerator(self.intra_feature, self.intra_feature, mem_dim=None,
                                                        input_dim=input_dim, factorized=False, number_of_weights=2)
        self.intra_Linear = torch.nn.Linear(self.patch_nums, self.patch_nums * self.patch_size)
        self.stride = patch_size
        self.inter_feature = self.feature * self.patch_size
        self.emb_linear = torch.nn.Linear(self.inter_feature, self.inter_feature)
        W_pos = torch.empty(self.patch_nums, 1)
        torch.nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = torch.nn.Parameter(W_pos)
        self.attention = attention(self.feature, self.inter_feature, dropout=0.2)
        self.norm_attn = torch.nn.Sequential(Transpose(1, 2), torch.nn.BatchNorm1d(self.feature), Transpose(1, 2))
        self.norm_ffn = torch.nn.Sequential(Transpose(1, 2), torch.nn.BatchNorm1d(self.feature), Transpose(1, 2))
        self.dropout = torch.nn.Dropout(0.1)
        self.ff = torch.nn.Sequential(torch.nn.Linear(self.feature, 64, bias=True), torch.nn.GELU(),
                                      torch.nn.Dropout(0.2), torch.nn.Linear(64, self.feature, bias=True))

    def forward(self, x):
        x0 = x
        intra_out_concat = None
        weights_shared, biases_shared = self.weights_generator_shared()
        weights_distinct, biases_distinct = self.weights_generator_distinct()
        for i in range(self.patch_nums):
            t = x[:, i * self.patch_size:(i + 1) * self.patch_size, :, :]
            intra_emb = self.embeddings_generator[i](self.intra_embeddings[i]).expand(x.shape[0], -1, -1, -1)
            t = torch.concat([intra_emb, t], dim=1)
            out = self.intra_patch_attention(intra_emb, t, t, weights_distinct, biases_distinct,
                                             weights_shared, biases_shared)
            if intra_out_concat == None:
                intra_out_concat = out
            else:
                intra_out_concat = torch.cat([intra_out_concat, out], dim=1)
        intra_out_concat = intra_out_concat.permute(0, 3, 2, 1)
        intra_out_concat = self.intra_Linear(intra_out_concat)
        intra_out_concat = intra_out_concat.permute(0, 3, 2, 1)
        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride)  # [b x patch_num x nvar x dim x patch_len]
        x = x.permute(0, 2, 1, 3, 4)  # [b x nvar x patch_num x dim x patch_len ]
        b, nvar, patch_num, dim, patch_len = x.shape
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3] * x.shape[-1])
        x = self.emb_linear(x)
        x = self.dropout(x + self.W_pos)
        inter_out = self.attention(x, x, x)  # [b*nvar, patch_num, dim]
        inter_out = torch.reshape(inter_out, (b, nvar, inter_out.shape[-2], inter_out.shape[-1]))
        inter_out = torch.reshape(inter_out, (b, nvar, inter_out.shape[-2], self.patch_size, self.feature))
        inter_out = torch.reshape(inter_out, (b, self.patch_size * self.patch_nums, nvar, self.feature))
        out = x0 + intra_out_concat + inter_out
        out = self.dropout(out)
        out = self.ff(out) + out
        return out


class Intra_Patch_Attention(torch.nn.Module):
    def __init__(self, feature):
        super().__init__()
        self.head_size = int(feature // 2)
        self.custom_linear = CustomLinear()

    def forward(self, query, key, value, weights_distinct, biases_distinct, weights_shared, biases_shared):
        batch_size = query.shape[0]
        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)
        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))
        att = torch.matmul(query, key)
        att /= (self.head_size ** 0.5)
        att = torch.softmax(att, dim=-1)
        x = torch.matmul(att, value)
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)
        if x.shape[0] == 0:
            x = x.repeat(1, 1, 1, int(weights_shared[0].shape[-1] / x.shape[-1]))
        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.relu(x)
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])
        return x


class CustomLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weights, biases):
        return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases


class WeightGenerator(torch.nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, input_dim, factorized, number_of_weights=4):
        super().__init__()
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.input_dim = input_dim
        self.factorized = factorized
        self.out_dim = out_dim
        if self.factorized:
            self.memory = torch.nn.Parameter(torch.randn(input_dim, mem_dim), requires_grad=True)
            self.generator = torch.nn.Sequential(*[torch.nn.Linear(mem_dim, 64), torch.nn.Tanh(),
                                                   torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 100)])
            self.mem_dim = 10
            self.P = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(in_dim, self.mem_dim),
                                                                requires_grad=True) for _ in range(number_of_weights)])
            self.Q = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(self.mem_dim, out_dim),
                                                                requires_grad=True) for _ in range(number_of_weights)])
            self.B = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(self.mem_dim ** 2, out_dim),
                                                                requires_grad=True) for _ in range(number_of_weights)])
        else:
            self.P = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(in_dim, out_dim),
                                                                requires_grad=True) for _ in range(number_of_weights)])
            self.B = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(1, out_dim),
                                                                requires_grad=True) for _ in range(number_of_weights)])
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))
            bias = [torch.matmul(memory, self.B[i]).squeeze(1) for i in range(self.number_of_weights)]
            memory = memory.view(self.input_dim, self.mem_dim, self.mem_dim)
            weights = [torch.matmul(torch.matmul(self.P[i], memory), self.Q[i]) for i in range(self.number_of_weights)]
            return weights, bias
        else:
            return self.P, self.B


class Transpose(torch.nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class pathformer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = len(args.input_column)
        output_dim = len(args.output_column)
        input_size = args.input_size
        output_size = args.output_size
        n_dict = {'s': 2, 'm': 3, 'l': 4}
        feature = n_dict[args.model_type]
        self.linear = torch.nn.Linear(1, feature)
        self.AMS0 = AMS(input_size, input_dim=input_dim, feature=feature, layer_number=1)
        self.AMS1 = AMS(input_size, input_dim=input_dim, feature=feature, layer_number=2)
        self.AMS2 = AMS(input_size, input_dim=input_dim, feature=feature, layer_number=3)
        self.linear_out = torch.nn.Linear(feature * input_size, output_size)
        self.conv1d = torch.nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.split_linear = split_linear(output_dim, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear(x.unsqueeze(3))
        x = self.AMS0(x)
        x = self.AMS1(x)
        x = self.AMS2(x)
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
        x = self.linear_out(x)
        x = self.conv1d(x)  # (batch,output_dim,output_size)
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
    model = pathformer(args)
    tensor = torch.randn((4, len(args.input_column), args.input_size), dtype=torch.float32)
    print(model)
    print(model(tensor).shape)
