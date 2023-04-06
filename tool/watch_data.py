import argparse
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='查看数据')
parser.add_argument('--data_path', default=r'../dataset/ETTh1.csv', type=str, help='|数据路径|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    df = pd.read_csv(args.data_path)
    name = args.data_path.split('/')[-1].split('.')[0] + '.jpg'
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('y')
    for column in df.columns[1:]:
        y = df[column][-1000:]
        plt.plot(y, label=column)
    plt.savefig(args.data_path.split('/')[-1].split('.')[0] + '.jpg')
    plt.show()
