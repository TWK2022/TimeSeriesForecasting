import os
import yaml
import finta
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|计算5日、10日、20日均线|')
parser.add_argument('--data_dir', default='dataset', type=str, help='|数据目录|')
parser.add_argument('--number_path', default='tushare/number.yaml', type=str, help='|选择的股票|')
parser.add_argument('--column', default='收盘价,成交量', type=str, help='|选择计算的变量|')
args = parser.parse_args()
args.column = args.column.split(',')


# -------------------------------------------------------------------------------------------------------------------- #
def fix(data):  # 修复数据中间的单个nan值
    judge_list = np.isnan(data)
    if judge_list.any():  # 存在nan值
        index_list = np.arange(1, len(data) - 1)[judge_list[1:-1]]
        for index in index_list:
            data[index] = (data[index - 1] + data[index + 1]) / 2
    return data


def add_zero(data):  # 补充0值
    judge_list = np.isnan(data)
    if judge_list.any():  # 存在nan值
        index_list = np.arange(len(data))[judge_list]
        data[index_list] = 0
    return data


def reciprocal(data, numerator):  # 求导数
    judge_list = np.where(data == 0, False, True)
    index_list = np.arange(len(data))[judge_list]
    data[index_list] = numerator / data[index_list]
    data = np.round(data, 6)
    return data


def data_deal(args):
    with open(args.number_path, 'r', encoding='utf-8') as f:
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    shangzheng = pd.read_csv(f'{args.data_dir}/上证指数.csv')['上证指数'].values
    shangzheng_vol = pd.read_csv(f'{args.data_dir}/上证指数.csv')['上证成交量'].values
    for industry in yaml_dict:
        for name in yaml_dict[industry].keys():
            path = f'{args.data_dir}/{name}.csv'
            if not os.path.exists(path):
                print(f'| 文件不存在:{path} |')
                continue
            df = pd.read_csv(path, index_col=0)
            value = df[args.column].values
            # 数据太少舍弃
            if len(value) < 200:
                print(f'| 数据太少舍弃:{name} |')
                continue
            # 上证指数
            df['上证指数'] = shangzheng[-len(df):]
            df['上证成交量'] = shangzheng_vol[-len(df):]
            # 计算指标
            df_count = pd.DataFrame(df[['开盘价', '最高价', '最低价', '收盘价', '成交量', '上证指数']].values,
                                    columns=['open', 'high', 'low', 'close', 'volume', '上证指数'], index=df.index)
            # 均线
            df['收盘价_SMA_5'] = finta.TA.SMA(df_count, 5, column='close')
            df['收盘价_SMA_10'] = finta.TA.SMA(df_count, 10, column='close')
            df['收盘价_SMA_20'] = finta.TA.SMA(df_count, 20, column='close')
            df['收盘价_SMA_30'] = finta.TA.SMA(df_count, 30, column='close')
            df['收盘价_SMA_60'] = finta.TA.SMA(df_count, 60, column='close')
            df['成交量_SMA_5'] = finta.TA.SMA(df_count, 5, column='volume')
            df['成交量_SMA_10'] = finta.TA.SMA(df_count, 10, column='volume')
            df['成交量_SMA_20'] = finta.TA.SMA(df_count, 20, column='volume')
            df['成交量_SMA_30'] = finta.TA.SMA(df_count, 30, column='volume')
            df['成交量_SMA_60'] = finta.TA.SMA(df_count, 60, column='volume')
            df['上证指数_SMA_5'] = finta.TA.SMA(df_count, 5, column='上证指数')
            df['上证指数_SMA_10'] = finta.TA.SMA(df_count, 10, column='上证指数')
            df['上证指数_SMA_20'] = finta.TA.SMA(df_count, 20, column='上证指数')
            df['上证指数_SMA_30'] = finta.TA.SMA(df_count, 30, column='上证指数')
            df['上证指数_SMA_60'] = finta.TA.SMA(df_count, 60, column='上证指数')
            # 指数均线
            df['收盘价_EMA_5'] = finta.TA.EMA(df_count, 5, column='close')
            df['收盘价_EMA_10'] = finta.TA.EMA(df_count, 10, column='close')
            df['收盘价_EMA_20'] = finta.TA.EMA(df_count, 20, column='close')
            df['收盘价_EMA_30'] = finta.TA.EMA(df_count, 30, column='close')
            df['收盘价_EMA_60'] = finta.TA.EMA(df_count, 60, column='close')
            df['成交量_EMA_5'] = finta.TA.EMA(df_count, 5, column='volume')
            df['成交量_EMA_10'] = finta.TA.EMA(df_count, 10, column='volume')
            df['成交量_EMA_20'] = finta.TA.EMA(df_count, 20, column='volume')
            df['成交量_EMA_30'] = finta.TA.EMA(df_count, 30, column='volume')
            df['成交量_EMA_60'] = finta.TA.EMA(df_count, 60, column='volume')
            df['上证指数_EMA_5'] = finta.TA.EMA(df_count, 5, column='上证指数')
            df['上证指数_EMA_10'] = finta.TA.EMA(df_count, 10, column='上证指数')
            df['上证指数_EMA_20'] = finta.TA.EMA(df_count, 20, column='上证指数')
            df['上证指数_EMA_30'] = finta.TA.EMA(df_count, 30, column='上证指数')
            df['上证指数_EMA_60'] = finta.TA.EMA(df_count, 60, column='上证指数')
            # 双指数均线
            df['收盘价_DEMA_5'] = finta.TA.DEMA(df_count, 5, column='close')
            df['收盘价_DEMA_10'] = finta.TA.DEMA(df_count, 10, column='close')
            df['收盘价_DEMA_20'] = finta.TA.DEMA(df_count, 20, column='close')
            df['收盘价_DEMA_30'] = finta.TA.DEMA(df_count, 30, column='close')
            df['收盘价_DEMA_60'] = finta.TA.DEMA(df_count, 60, column='close')
            df['成交量_DEMA_5'] = finta.TA.DEMA(df_count, 5, column='volume')
            df['成交量_DEMA_10'] = finta.TA.DEMA(df_count, 10, column='volume')
            df['成交量_DEMA_20'] = finta.TA.DEMA(df_count, 20, column='volume')
            df['成交量_DEMA_30'] = finta.TA.DEMA(df_count, 30, column='volume')
            df['成交量_DEMA_60'] = finta.TA.DEMA(df_count, 60, column='volume')
            df['上证指数_DEMA_5'] = finta.TA.DEMA(df_count, 5, column='上证指数')
            df['上证指数_DEMA_10'] = finta.TA.DEMA(df_count, 10, column='上证指数')
            df['上证指数_DEMA_20'] = finta.TA.DEMA(df_count, 20, column='上证指数')
            df['上证指数_DEMA_30'] = finta.TA.DEMA(df_count, 30, column='上证指数')
            df['上证指数_DEMA_60'] = finta.TA.DEMA(df_count, 60, column='上证指数')
            # 其他指标
            df['开盘涨幅'] = df['开盘价'] / df['昨收价(前复权)']
            df['震荡幅度'] = (df_count['high'] - df_count['low']) / df_count['open']
            # 补充数据
            df['量比'] = fix(df['量比'].values)
            df['市盈率ttm'] = add_zero(df['市盈率ttm'].values)
            df['市净率'] = add_zero(df['市净率'].values)
            df['市销率ttm'] = add_zero(df['市销率ttm'].values)
            # 修改数据(由于亏损为0，原来值越小越好，现在改为值越大越好)
            df['市盈率ttm'] = reciprocal(df['市盈率ttm'].values, 100)
            df['市净率'] = reciprocal(df['市净率'].values, 1)
            df['市销率ttm'] = reciprocal(df['市销率ttm'].values, 1)
            df.rename(columns={'市盈率ttm': 'r市盈率ttm'}, inplace=True)
            df.rename(columns={'市净率': 'r市净率'}, inplace=True)
            df.rename(columns={'市销率ttm': 'r市销率ttm'}, inplace=True)
            # 去除不需要的部分
            df = df[30:]
            # 保存
            df.to_csv(f'{args.data_dir}/{name}_add.csv', header=True, index=True)
            print(f'| 结果保存至:{args.data_dir}/{name}_add.csv |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    data_deal(args)
