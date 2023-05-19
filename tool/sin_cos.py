import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|制造sin_cos周期数据|')
parser.add_argument('--sin_cycle', default=100, type=int, help='|周期|')
parser.add_argument('--cos_cycle', default=100, type=int, help='|周期|')
parser.add_argument('--multiple', default=3, type=int, help='|倍数|')
parser.add_argument('--add', default=2, type=int, help='|y轴偏移|')
parser.add_argument('--len', default=2500, type=int, help='|数据长度|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    data_base = np.ones(args.len) * args.multiple
    noise = np.random.rand(args.len)
    index = np.arange(args.len)
    sin = (data_base * np.sin(2 * np.pi / args.sin_cycle * index) + args.add) + noise
    sinadd = (data_base * np.sin(2 * np.pi / args.sin_cycle * index) + args.add) + noise + 2
    cos = (data_base * np.cos(2 * np.pi / args.cos_cycle * index) + args.add) + noise
    add = sin + cos
    multiply = sin * cos
    data = np.stack([sin, sinadd, cos, add, multiply], axis=1)
    column = ['sin', 'sinadd', 'cos', 'add', 'multiply']
    df = pd.DataFrame(data, columns=column)
    save_path = '../dataset/sin_cos.csv'
    df.to_csv(save_path, index=True, header=True)
    print(f'| 结果保存至:{save_path} |')
