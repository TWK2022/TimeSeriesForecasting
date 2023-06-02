import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|将月指标简单补充成日指标，月指标当成该月的最后一天|')
parser.add_argument('--data_path', default=r'../dataset/economy_month.csv', type=str, help='|数据路径|')
parser.add_argument('--save_path', default=r'../dataset/economy_day.csv', type=str, help='|保存路径|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def add_date(month_date):
    month_date = pd.date_range(start=month_date[0], periods=len(month_date), freq='M')  # 得到每个月的最后一天
    month_number = month_date.day  # 两个月之间间隔的天数
    day_date = pd.date_range(start=month_date[0], end=month_date[-1], freq='D')  # 完整的天数
    return month_number, day_date


if __name__ == '__main__':
    df = pd.read_csv(args.data_path)
    month_date = df[df.columns[0]].values
    month_data = df[df.columns[1:]].values
    month_number, day_date = add_date(month_date)
    day_data = np.zeros((len(day_date), month_data.shape[1]))
    record = 0  # 记录填充的数据下标
    for i in range(len(month_data) - 1):
        start = month_data[i, :][np.newaxis]
        end = month_data[i + 1, :][np.newaxis]
        number = month_number[i + 1]
        add = np.arange(number)[:, np.newaxis].repeat(month_data.shape[1], axis=1)
        add = start + add * (end - start) / number  # 线性拟合
        day_data[record:record + number, :] = add
        record += number
    day_data[-1, :] = month_data[-1]
    df1 = pd.DataFrame(day_date, columns=['day'])
    df2 = pd.DataFrame(day_data, columns=df.columns[1:])
    df = pd.concat([df1, df2], axis=1)
    df.to_csv(args.save_path, index=False)
    print(f'| 结果保存至:{args.save_path} |')
