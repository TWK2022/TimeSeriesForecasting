import numpy as np
import pandas as pd


def data_get(args):
    data_dict = data_prepare(args).load()
    return data_dict


class data_prepare(object):
    def __init__(self, args):
        self.input_column = args.input_column
        self.output_column = args.output_column
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.divide = args.divide
        self.data_path = args.data_path
        self.z_score_cycle = args.z_score_cycle

    def load(self):
        # 读取数据
        try:
            df = pd.read_csv(self.data_path, encoding='utf-8')
        except:
            df = pd.read_csv(self.data_path, encoding='gbk')
        input_data = np.array(df[self.input_column].astype(np.float32)).transpose(1, 0)
        output_data = np.array(df[self.output_column].astype(np.float32)).transpose(1, 0)
        # 划分数据集
        add = self.input_size + self.output_size - 1  # 输入数据后面的补足
        data_len = len(df) - add  # 输入数据的真实数量
        boundary = int(data_len * self.divide[0] / (self.divide[0] + self.divide[1]))  # 数据划分
        train_input = input_data[:, 0:boundary + add]  # 训练数据
        train_output = output_data[:, 0:boundary + add]  # 训练标签
        val_input = input_data[:, boundary:len(df)].copy()  # 验证数据
        val_output = output_data[:, boundary:len(df)].copy()  # 验证标签
        # 周期
        if self.z_score_cycle == -1:
            self.max_cycle = train_input.shape[1]
        else:
            assert self.z_score_cycle <= train_input.shape[1], f'周期设置不能大于训练集长度'
            self.max_cycle = train_input.shape[1] // self.z_score_cycle * self.z_score_cycle
        # 数据处理
        train_input, val_input, mean_input, std_input = self._z_score(train_input, val_input, self.input_column)
        train_output, val_output, mean_output, std_output = self._z_score(train_output, val_output, self.output_column)
        # 将所有数据存放到一个大字典中
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

    def _z_score(self, train_data, val_data, column):  # 减均值、除以方差
        mean_all = np.zeros(len(column))
        std_all = np.zeros(len(column))
        for i in range(len(column)):
            mean = np.mean(train_data[i, 0:self.max_cycle])
            std = np.std(train_data[i, 0:self.max_cycle])
            std = max(std, 0.01)
            mean_all[i] = mean
            std_all[i] = std
            train_data[i, :] = (train_data[i, :] - mean) / std
            val_data[i, :] = (val_data[i, :] - mean) / std
        return train_data, val_data, mean_all, std_all
