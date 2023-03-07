import numpy as np
import pandas as pd


def data_get(args):
    data_dict = data_prepare(args)._load()
    return data_dict


class data_prepare(object):
    def __init__(self, args):
        self.args = args

    def _load(self):
        input_column = self.args.input_column
        output_column = self.args.output_column
        divide = self.args.divide
        # 读取数据
        df = pd.read_csv(self.args.data_path)
        input_data = np.array(df[input_column].astype(np.float32))
        output_data = np.array(df[output_column].astype(np.float32))
        # 划分数据集
        boundary = int(len(df) * divide[0] / (divide[0] + divide[1]))
        train_input = input_data[0:boundary]  # 训练数据
        train_output = output_data[0:boundary]  # 训练标签
        val_input = input_data[boundary:len(df)]  # 验证数据
        val_output = output_data[boundary:len(df)]  # 验证标签
        # 记录输入数据的处理方式
        input_mean = {}
        input_std = {}
        for i in range(len(input_column)):
            mean = np.mean(train_input[:, i])
            std = np.std(train_input[:, i])
            train_input[:, i] = (train_input[:, i] - mean) / std
            val_input[:, i] = (val_input[:, i] - mean) / std
            input_mean[input_column[i]] = mean
            input_std[input_column[i]] = std
        # 记录输出数据的处理方式
        output_mean = {}
        output_std = {}
        for i in range(len(output_column)):
            mean = np.mean(train_output[:, i])
            std = np.std(train_output[:, i])
            train_output[:, i] = (train_output[:, i] - mean) / std
            val_output[:, i] = (val_output[:, i] - mean) / std
            output_mean[output_column[i]] = mean
            output_std[output_column[i]] = std
        # 将所有数据存放到一个大字典中
        data_dict = {}
        data_dict['train_input'] = train_input
        data_dict['train_output'] = train_output
        data_dict['val_input'] = val_input
        data_dict['val_output'] = val_output
        data_dict['input_column'] = input_column
        data_dict['output_column'] = output_column
        data_dict['input_mean'] = input_mean
        data_dict['input_std'] = input_std
        data_dict['output_mean'] = output_mean
        data_dict['output_std'] = output_std
        return data_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', default='../dataset/demo.csv', type=str)
    parser.add_argument('--input_column', default='1,3', type=str)
    parser.add_argument('--output_column', default='1,3', type=str)
    parser.add_argument('--device', default='9,1', type=str)
    args = parser.parse_args()
    args.divide = list(map(int, args.divide.split(',')))
    data_dict = data_get(args)
