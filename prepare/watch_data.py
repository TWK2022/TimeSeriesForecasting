import argparse
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|查看数据|')
parser.add_argument('--data_path', default='dataset/华银电力.csv', type=str, help='|数据位置|')
parser.add_argument('--column', default='收盘价,涨跌幅,成交量', type=str, help='|变量选择|')
parser.add_argument('--show_len', default=100, type=int, help='|显示的数据长度|')
args = parser.parse_args()
args.column = args.column.split(',')


# -------------------------------------------------------------------------------------------------------------------- #
def watch_data():
    df = pd.read_csv(args.data_path, index_col=0)
    name = args.data_path.split('/')[-1].split('.')[0]
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 使用字体时让坐标轴正常显示负号
    plt.title(name)
    plt.xlabel('time')
    plt.ylabel('value')
    for column in args.column:
        y = df[column][-args.show_len:].values
        plt.plot(y, label=column)
        plt.grid()
        plt.legend()
        plt.show()


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    watch_data()
