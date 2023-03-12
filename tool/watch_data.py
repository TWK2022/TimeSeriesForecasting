import argparse
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', default=r'../dataset/ETTm.csv', type=str, help='|数据路径|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    df = pd.read_csv(args.data_path)
    name = args.data_path.split('/')[-1].split('.')[0] + '.jpg'
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('y')
    for i in df.columns[1:]:
        x = df[i][-1000:]
        plt.plot(x)
    plt.savefig(args.data_path.split('/')[-1].split('.')[0] + '.jpg')
    plt.show()
