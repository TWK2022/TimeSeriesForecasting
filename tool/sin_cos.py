import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|制造sin_cos周期数据|')
parser.add_argument('--cycle', default=100, type=int, help='|周期|')
parser.add_argument('--multiple', default=10, type=int, help='|倍数|')
parser.add_argument('--add', default=2, type=int, help='|y轴偏移|')
parser.add_argument('--len', default=2000, type=int, help='|长度|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    data_base = np.ones(args.len) * args.multiple
    index = np.arange(args.len)
    sin = data_base * np.sin(2 * np.pi / args.cycle * index) + args.add
    cos = data_base * np.cos(2 * np.pi / args.cycle * index) + args.add
    sin_add_cos = sin - cos
    sin_multiply_cos = sin * cos
    data = np.stack([sin, cos, sin_add_cos, sin_multiply_cos], axis=1)
    column = ['sin', 'cos', 'sin_add_cos', 'sin_multiply_cos']
    df = pd.DataFrame(data, columns=column)
    df.to_csv('../dataset/sin_cos.csv', index=True, header=True)
