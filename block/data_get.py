import numpy as np
import pandas as pd


def data_get(args):
    data_dict = data_prepare(args).load()
    return data_dict


class data_prepare:
    def __init__(self, args):
        self.data_path = args.data_path
        self.input_column = args.input_column
        self.output_column = args.output_column
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.divide = args.divide
        self.divide_all = args.divide_all
        self.z_score_cycle = args.z_score_cycle

    def load(self):
        # 读取数据
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8', index_col=0)
        except:
            df = pd.read_csv(self.data_path, encoding='gbk', index_col=0)
        input_data = np.array(df[self.input_column]).astype(np.float32)
        output_data = np.array(df[self.output_column]).astype(np.float32)
        # 划分数据
        add = self.input_size + self.output_size - 1  # 输入数据后面的补足
        data_len = len(df) - add  # 输入数据的真实数量
        boundary = int(data_len * self.divide[0] / (self.divide[0] + self.divide[1]))  # 数据划分
        if self.divide_all:
            train_input = input_data  # 训练数据
            train_output = output_data  # 训练标签
        else:
            train_input = input_data[0:boundary + add]  # 训练数据
            train_output = output_data[0:boundary + add]  # 训练标签
        assert len(train_input) >= self.input_size + self.output_size  # 训练集不满足一个batch
        val_input = input_data[boundary:len(df)].copy()  # 验证数据
        val_output = output_data[boundary:len(df)].copy()  # 验证标签
        # 周期
        if self.z_score_cycle == -1:
            max_cycle = input_data.shape[0]
        else:
            assert self.z_score_cycle <= len(train_input), f'! 周期设置不能大于训练集长度 !'
            max_cycle = train_input.shape[0] // self.z_score_cycle * self.z_score_cycle
        # 归一化
        mean_input = np.mean(train_input[0:max_cycle], axis=0)
        mean_output = np.mean(train_output[0:max_cycle], axis=0)
        std_input = np.std(train_input[0:max_cycle], axis=0)
        std_output = np.std(train_output[0:max_cycle], axis=0)
        train_input = (train_input - mean_input) / std_input
        val_input = (val_input - mean_input) / std_input
        train_output = (train_output - mean_output) / std_output
        val_output = (val_output - mean_output) / std_output
        # 记录数据
        data_dict = {}
        data_dict['train_input'] = train_input
        data_dict['train_output'] = train_output
        data_dict['val_input'] = val_input
        data_dict['val_output'] = val_output
        data_dict['mean_input'] = mean_input
        data_dict['mean_output'] = mean_output
        data_dict['std_input'] = std_input
        data_dict['std_output'] = std_output
        return data_dict
