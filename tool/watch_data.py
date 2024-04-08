import argparse
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|查看数据|')
parser.add_argument('--data_path', default=r'../dataset/ETTh1.csv', type=str, help='|数据路径|')
parser.add_argument('--column', default='HUFL,HULL,MUFL,MULL,LUFL,LULL,OT', type=str, help='|数据路径|')
parser.add_argument('--show_len', default=500, type=int, help='|显示的数据长度|')
args = parser.parse_args()
args.column = args.column.split(',')
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    df = pd.read_csv(args.data_path)
    name = args.data_path.split('/')[-1].split('.')[0] + '.jpg'
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 使用字体时让坐标轴正常显示负号
    plt.title(name)
    plt.xlabel('time')
    plt.ylabel('value')
    for column in args.column:
        y = df[column][-args.show_len:].values
        plt.plot(y, label=column)
    plt.legend()
    save_path = args.data_path.split('/')[-1].split('.csv')[0] + '.jpg'
    plt.savefig(save_path)
    plt.show()
    print(f'| 结果保存至:{save_path} |')
