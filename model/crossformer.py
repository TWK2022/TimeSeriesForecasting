# 根据crossformer改编:https://github.com/Thinklab-SJTU/Crossformer
# 多变量异标签
import torch
from model.layer import lsl, multihead_attention, split_linear


class attention_block(torch.nn.Module):
    def __init__(self, number, middle_dim, feature, head):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(number, middle_dim, feature))
        self.time_attention = multihead_attention(feature, head, dropout=0.2)
        self.param_attention = multihead_attention(feature, head, dropout=0.2)
        self.input_attention = multihead_attention(feature, head, dropout=0.2)
        self.lsl1 = lsl(feature, 2)
        self.lsl2 = lsl(feature, 2)
        self.normalization1 = torch.nn.LayerNorm(feature)
        self.normalization2 = torch.nn.LayerNorm(feature)
        self.normalization3 = torch.nn.LayerNorm(feature)
        self.normalization4 = torch.nn.LayerNorm(feature)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.dropout3 = torch.nn.Dropout(0.2)
        self.dropout4 = torch.nn.Dropout(0.2)

    def forward(self, x):  # (batch,dim,number,feature) -> (batch,dim,number,feature)
        batch, dim, number, feature = x.shape
        x = x.reshape(batch * dim, number, feature)  # (batch*dim,number,feature)
        x = self.normalization1(x + self.dropout1(self.time_attention(x, x, x)))
        x = self.normalization2(x + self.dropout2(self.lsl1(x)))  # (batch*dim,number,feature)
        x = x.reshape(batch * number, dim, feature)  # (batch*n,dim,feature)
        param = self.param.repeat(batch, 1, 1)  # (batch*n,middle_dim,feature)
        param = self.param_attention(param, x, x)  # (batch*n,middle_dim,feature)
        param = self.input_attention(x, param, param)  # (batch*n,dim,feature)
        x = self.normalization3(x + self.dropout3(param))
        x = self.normalization4(x + self.dropout4(self.lsl2(x)))
        x = x.reshape(batch, dim, number, feature)
        return x


class merge_feature(torch.nn.Module):
    def __init__(self, feature):
        super().__init__()
        self.normalization = torch.nn.LayerNorm(2 * feature)
        self.linear = torch.nn.Linear(2 * feature, feature)

    def forward(self, x):  # (batch,dim,number,feature) -> (batch,dim,number//2,feature)
        x = torch.concat([x[:, :, 0::2, :], x[:, :, 1::2, :]], dim=3)
        x = self.normalization(x)
        x = self.linear(x)
        return x


class encode(torch.nn.Module):
    def __init__(self, number, head, feature, middle_dim=5):
        super().__init__()
        self.attention_block1 = attention_block(number, middle_dim, feature, head)
        self.merge1 = merge_feature(feature)
        self.attention_block2 = attention_block(number // 2, middle_dim, feature, head)
        self.merge2 = merge_feature(feature)
        self.attention_block3 = attention_block(number // 4, middle_dim, feature, head)

    def forward(self, x):  # (batch,dim,number,feature) -> [(batch,dim,number、number、number//2、number//4,feature)]
        x0 = self.attention_block1(x)
        x1 = self.merge1(x0)
        x1 = self.attention_block2(x1)
        x2 = self.merge2(x1)
        x2 = self.attention_block3(x2)
        x_list = [x, x0, x1, x2]
        return x_list


class decode_block(torch.nn.Module):
    def __init__(self, number, head, feature, middle_dim=5):
        super().__init__()
        self.self_attention = attention_block(number, middle_dim, feature, head)
        self.encode_decode_attention = multihead_attention(feature, head, dropout=0.2)
        self.lsl = lsl(feature, 2)
        self.normalization1 = torch.nn.LayerNorm(feature)
        self.normalization2 = torch.nn.LayerNorm(feature)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, cross):  # (batch,dim,number,feature) -> (batch,dim,number,feature)
        batch, dim, number, feature = x.shape
        x = self.self_attention(x)
        x = x.reshape(batch * dim, number, feature)  # (batch*dim,number,feature)
        cross = cross.reshape(batch * dim, -1, feature)  # (batch*dim,-1,feature)
        x = self.normalization1(x + self.dropout(self.encode_decode_attention(x, cross, cross)))
        x = self.normalization2(x + self.lsl(x))
        x = x.reshape(batch, dim, number, feature)  # (batch,dim,number,feature)
        return x


class decode(torch.nn.Module):
    def __init__(self, number, segment, head, feature, middle_dim=5):
        super().__init__()
        self.decode_layer1 = decode_block(number, head, feature, middle_dim)
        self.linear1 = torch.nn.Linear(feature, segment)
        self.decode_layer2 = decode_block(number, head, feature, middle_dim)
        self.linear2 = torch.nn.Linear(feature, segment)
        self.decode_layer3 = decode_block(number, head, feature, middle_dim)
        self.linear3 = torch.nn.Linear(feature, segment)
        self.decode_layer4 = decode_block(number, head, feature, middle_dim)
        self.linear4 = torch.nn.Linear(feature, segment)
        self.normalization = torch.nn.LayerNorm(segment)

    def forward(self, x, encode_list):  # (batch,dim,number,feature) -> (batch,dim,number,segment)
        x = self.decode_layer1(x, encode_list[0])
        x1 = self.linear1(x)
        x = self.decode_layer2(x, encode_list[1])
        x2 = self.linear2(x)
        x = self.decode_layer3(x, encode_list[2])
        x3 = self.linear3(x)
        x = self.decode_layer4(x, encode_list[3])
        x4 = self.linear4(x)
        x = x1 + x2 + x3 + x4
        x = self.normalization(x)
        return x


class crossformer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = len(args.input_column)
        output_dim = len(args.output_column)
        input_size = args.input_size
        output_size = args.output_size
        n_dict = {'s': 8, 'm': 16, 'l': 32}
        feature = n_dict[args.model_type]
        head = 8
        middle_dim = 5
        self.segment = output_size // 4
        assert input_size % (self.segment * 4) == 0 or output_size % (self.segment * 2) == 0
        input_number = input_size // self.segment
        output_number = output_size // self.segment
        # 网络结构
        self.embedding = torch.nn.Linear(self.segment, feature)
        self.position = torch.nn.Parameter(torch.randn(1, input_dim, input_number, feature))
        self.normalization = torch.nn.LayerNorm(feature)
        self.encode = encode(input_number, head, feature, middle_dim=middle_dim)
        self.decode_input = torch.nn.Parameter(torch.randn(1, input_dim, output_number, feature))
        self.decode = decode(output_number, self.segment, head, feature, middle_dim=middle_dim)
        self.conv1d = torch.nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.split_linear = split_linear(output_dim, output_size, output_size)

    def forward(self, x):  # (batch,input_dim,input_size) -> (batch,output_dim,output_size)
        batch, input_dim, input_size = x.shape
        x = x.reshape(batch, input_dim, -1, self.segment)  # (batch,input_dim,input_number,segment)
        x = self.embedding(x)  # (batch,input_dim,input_number,feature)
        x += self.position
        x = self.normalization(x)
        encode_list = self.encode(x)
        decode_input = self.decode_input.repeat(batch, 1, 1, 1)  # (batch,input_dim,output_number,feature)
        x = self.decode(decode_input, encode_list)  # (batch,input_dim,output_number,segment)
        batch, dim, n, segment = x.shape
        x = x.reshape(batch, dim, -1)  # (batch,input_dim,output_size)
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
    model = crossformer(args)
    tensor = torch.randn((4, len(args.input_column), args.input_size), dtype=torch.float32)
    print(model)
    print(model(tensor).shape)
