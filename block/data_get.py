import numpy as np
import pandas as pd


def data_get(args):
    data_dict = data_prepare(args)._load()
    return data_dict


class data_prepare(object):
    def __init__(self, args):
        self.input_column = args.input_column
        self.output_column = args.output_column
        self.divide = args.divide
        self.data_path = args.data_path

    def _load(self):
        # 读取数据
        df = pd.read_csv(self.data_path)
        input_data = np.array(df[self.input_column].astype(np.float32))
        output_data = np.array(df[self.output_column].astype(np.float32))
        # 划分数据集
        boundary = int(len(df) * self.divide[0] / (self.divide[0] + self.divide[1]))
        train_input = input_data[0:boundary]  # 训练数据
        train_output = output_data[0:boundary]  # 训练标签
        val_input = input_data[boundary:len(df)]  # 验证数据
        val_output = output_data[boundary:len(df)]  # 验证标签
        # 记录输入数据的处理方式
        input_mean = np.zeros(len(self.input_column))
        input_std = np.zeros(len(self.input_column))
        for i in range(len(self.input_column)):
            mean = np.mean(input_data[:, i])
            std = np.std(input_data[:, i])
            input_mean[i] = mean
            input_std[i] = std
            train_input[:, i] = (train_input[:, i] - mean) / std
            val_input[:, i] = (val_input[:, i] - mean) / std
        # 记录输出数据的处理方式(数据处理在模型中完成)
        output_mean = np.zeros(len(self.output_column))
        output_std = np.zeros(len(self.output_column))
        for i in range(len(self.output_column)):
            mean = np.mean(output_data[:, i])
            std = np.std(output_data[:, i])
            output_mean[i] = mean
            output_std[i] = std
            train_output[:, i] = (train_output[:, i] - mean) / std
            val_output[:, i] = (val_output[:, i] - mean) / std
        # 将所有数据存放到一个大字典中
        data_dict = {}
        data_dict['train_input'] = train_input
        data_dict['train_output'] = train_output
        data_dict['val_input'] = val_input
        data_dict['val_output'] = val_output
        data_dict['input_mean'] = input_mean
        data_dict['input_std'] = input_std
        data_dict['output_mean'] = output_mean
        data_dict['output_std'] = output_std
        return data_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', default='../dataset/ETTh.csv', type=str)
    parser.add_argument('--input_column', default='1,2,3', type=str)
    parser.add_argument('--output_column', default='1,2', type=str)
    parser.add_argument('--divide', default='9,1', type=str)
    args = parser.parse_args()
    args.input_column = args.input_column.split(',')
    args.output_column = args.output_column.split(',')
    args.divide = list(map(int, args.divide.split(',')))
    data_dict = data_get(args)
