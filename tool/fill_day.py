import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|将月指标简单补充成日指标，月指标当成该月的最后一天|')
parser.add_argument('--data_path', default=r'../dataset/沪深300.csv', type=str, help='|数据路径|')
parser.add_argument('--save_path', default=r'../dataset/沪深300_day.csv', type=str, help='|保存路径|')
args = parser.parse_args()

# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    df = pd.read_csv(args.data_path)
    old_date = df[df.columns[0]].values
    old_date = pd.DatetimeIndex(old_date)
    old_data = df[df.columns[1:]].values
    new_date = pd.date_range(start=old_date[0], end=old_date[-1], freq='D')  # 得到每个月的最后一天完整的天数
    new_data = np.zeros((len(new_date), old_data.shape[1]))
    record = [0, 0, 0]  # [old_date下标,填补数记录,new_data空缺起始下标]
    for i in range(len(new_date)):
        if new_date[i] == old_date[record[0]]:
            if record[1] != 0:  # 之前有空缺记录
                end = old_data[record[0], :]
                add = np.arange(1, record[1] + 1)[:, np.newaxis].repeat(old_data.shape[1], axis=1)
                add = start + add * (end - start) / (record[2] + 1)  # 线性拟合
                new_data[record[2]:record[2] + record[1], :] = add
                record[1] = 0  # 清空记录
            new_data[i, :] = old_data[record[0], :]
            record[0] += 1
        else:
            if record[1] == 0:  # 之前无空缺记录
                record[2] = i  # 记录空缺的起始下标
                start = old_data[record[0] - 1, :]
            record[1] += 1  # 记录空缺的长度
    df1 = pd.DataFrame(new_date, columns=['day'])
    df2 = pd.DataFrame(new_data, columns=df.columns[1:])
    df = pd.concat([df1, df2], axis=1)
    df.to_csv(args.save_path, index=False)
    print(f'| 结果保存至:{args.save_path} |')
