import numpy as np
import pandas as pd


def data_get(args):
    data_dict = data_prepare(args).load()
    return data_dict


class data_prepare:
    def __init__(self, args):
        self.args = args

    def load(self):
        # 读取数据
        try:
            df = pd.read_csv(self.args.data_path, encoding='utf-8', index_col=0)
        except:
            df = pd.read_csv(self.args.data_path, encoding='gbk', index_col=0)
        input_data = np.array(df[self.args.input_column]).astype(np.float32)
        output_data = np.array(df[self.args.output_column]).astype(np.float32)
        # 划分数据
        add = self.args.input_size + self.args.output_size - 1  # 输入数据后面的补足
        data_len = len(df) - add  # 输入数据的真实数量
        boundary = int(data_len * self.args.divide[0] / (self.args.divide[0] + self.args.divide[1]))  # 数据划分
        if self.args.divide_train == 1:  # 使用所有数据训练
            train_input = input_data  # 训练数据
            train_output = output_data  # 训练标签
        elif self.args.divide_train == 2:  # 使用验证集训练
            train_input = input_data[boundary:len(df)]  # 训练数据
            train_output = output_data[boundary:len(df)]  # 训练标签
        else:  # 使用训练集训练
            train_input = input_data[0:boundary + add]  # 训练数据
            train_output = output_data[0:boundary + add]  # 训练标签
        assert len(train_input) >= self.args.input_size + self.args.output_size  # 训练集不满足一个batch
        val_input = input_data[boundary:len(df)].copy()  # 验证数据
        val_output = output_data[boundary:len(df)].copy()  # 验证标签
        # 周期
        if self.args.z_score == 1:
            mean_input = np.mean(input_data, axis=0)
            mean_output = np.mean(output_data, axis=0)
            std_input = np.std(input_data, axis=0)
            std_output = np.std(output_data, axis=0)
        elif self.args.z_score == 2:
            mean_input = np.mean(val_input, axis=0)
            mean_output = np.mean(val_output, axis=0)
            std_input = np.std(val_input, axis=0)
            std_output = np.std(val_output, axis=0)
        else:
            mean_input = np.mean(train_input, axis=0)
            mean_output = np.mean(train_output, axis=0)
            std_input = np.std(train_input, axis=0)
            std_output = np.std(train_output, axis=0)
        # 归一化
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
        # 特殊数据(需要根据情况更改)
        data_dict['train_special'] = train_input[:, [0]] if 'special' in self.args.model else None
        data_dict['val_special'] = val_input[:, [0]] if 'special' in self.args.model else None
        data_dict['mean_special'] = mean_input[0] if 'special' in self.args.model else None
        data_dict['std_special'] = std_input[0] if 'special' in self.args.model else None
        return data_dict
