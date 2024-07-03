import os
import yaml
import finta
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|补充数据|')
parser.add_argument('--data_dir', default='dataset', type=str, help='|数据目录|')
parser.add_argument('--number_path', default='tushare/number.yaml', type=str, help='|选择的股票|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
def fix(data, decimal=2):  # 修复数据中的nan值
    judge_list = np.isnan(data)
    if judge_list.any():  # 存在nan值
        index_list = np.arange(1, len(data))[judge_list[1:]]
        index_max = len(judge_list) - 1
        for index in index_list:
            index_next = index + 1
            record = 2
            while index_next <= index_max and np.isnan(data[index_next]):  # 下一个值也为空
                index_next += 1
                record += 1
            data[index] = data[index - 1] + (data[index_next] - data[index - 1]) / record
            data[index] = round(data[index], decimal)
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
            # 数据太少舍弃
            if len(df) < 200:
                print(f'| 数据太少舍弃:{name} |')
                continue
            # 上证指数
            df['上证指数'] = shangzheng[-len(df):]
            df['上证成交量'] = shangzheng_vol[-len(df):]
            # 计算指标
            df_count = pd.DataFrame(
                df[['开盘价', '最高价', '最低价', '收盘价', '成交量', '上证指数', '上证成交量']].values,
                columns=['open', 'high', 'low', 'close', 'volume', '上证指数', '上证成交量'], index=df.index)
            # 均线
            df['最高价_SMA_5'] = finta.TA.SMA(df_count, 5, column='close')
            df['最高价_SMA_10'] = finta.TA.SMA(df_count, 10, column='close')
            df['最高价_SMA_20'] = finta.TA.SMA(df_count, 20, column='close')
            df['最低价_SMA_5'] = finta.TA.SMA(df_count, 5, column='close')
            df['最低价_SMA_10'] = finta.TA.SMA(df_count, 10, column='close')
            df['最低价_SMA_20'] = finta.TA.SMA(df_count, 20, column='close')
            df['成交量_SMA_5'] = finta.TA.SMA(df_count, 5, column='volume')
            df['成交量_SMA_10'] = finta.TA.SMA(df_count, 10, column='volume')
            df['成交量_SMA_20'] = finta.TA.SMA(df_count, 20, column='volume')
            df['上证指数_SMA_5'] = finta.TA.SMA(df_count, 5, column='上证指数')
            df['上证指数_SMA_10'] = finta.TA.SMA(df_count, 10, column='上证指数')
            df['上证指数_SMA_20'] = finta.TA.SMA(df_count, 20, column='上证指数')
            df['上证成交量_SMA_5'] = finta.TA.SMA(df_count, 5, column='上证成交量')
            df['上证成交量_SMA_10'] = finta.TA.SMA(df_count, 10, column='上证成交量')
            df['上证成交量_SMA_20'] = finta.TA.SMA(df_count, 20, column='上证成交量')
            # 指数均线
            df['最高价_EMA_5'] = finta.TA.EMA(df_count, 5, column='close')
            df['最高价_EMA_10'] = finta.TA.EMA(df_count, 10, column='close')
            df['最高价_EMA_20'] = finta.TA.EMA(df_count, 20, column='close')
            df['最低价_EMA_5'] = finta.TA.EMA(df_count, 5, column='close')
            df['最低价_EMA_10'] = finta.TA.EMA(df_count, 10, column='close')
            df['最低价_EMA_20'] = finta.TA.EMA(df_count, 20, column='close')
            df['成交量_EMA_5'] = finta.TA.EMA(df_count, 5, column='volume')
            df['成交量_EMA_10'] = finta.TA.EMA(df_count, 10, column='volume')
            df['成交量_EMA_20'] = finta.TA.EMA(df_count, 20, column='volume')
            df['上证指数_EMA_5'] = finta.TA.EMA(df_count, 5, column='上证指数')
            df['上证指数_EMA_10'] = finta.TA.EMA(df_count, 10, column='上证指数')
            df['上证指数_EMA_20'] = finta.TA.EMA(df_count, 20, column='上证指数')
            df['上证成交量_EMA_5'] = finta.TA.EMA(df_count, 5, column='上证成交量')
            df['上证成交量_EMA_10'] = finta.TA.EMA(df_count, 10, column='上证成交量')
            df['上证成交量_EMA_20'] = finta.TA.EMA(df_count, 20, column='上证成交量')
            # 双指数均线
            df['最高价_DEMA_5'] = finta.TA.DEMA(df_count, 5, column='close')
            df['最高价_DEMA_10'] = finta.TA.DEMA(df_count, 10, column='close')
            df['最高价_DEMA_20'] = finta.TA.DEMA(df_count, 20, column='close')
            df['最低价_DEMA_5'] = finta.TA.DEMA(df_count, 5, column='close')
            df['最低价_DEMA_10'] = finta.TA.DEMA(df_count, 10, column='close')
            df['最低价_DEMA_20'] = finta.TA.DEMA(df_count, 20, column='close')
            df['成交量_DEMA_5'] = finta.TA.DEMA(df_count, 5, column='volume')
            df['成交量_DEMA_10'] = finta.TA.DEMA(df_count, 10, column='volume')
            df['成交量_DEMA_20'] = finta.TA.DEMA(df_count, 20, column='volume')
            df['上证指数_DEMA_5'] = finta.TA.DEMA(df_count, 5, column='上证指数')
            df['上证指数_DEMA_10'] = finta.TA.DEMA(df_count, 10, column='上证指数')
            df['上证指数_DEMA_20'] = finta.TA.DEMA(df_count, 20, column='上证指数')
            df['上证成交量_DEMA_5'] = finta.TA.DEMA(df_count, 5, column='上证成交量')
            df['上证成交量_DEMA_10'] = finta.TA.DEMA(df_count, 10, column='上证成交量')
            df['上证成交量_DEMA_20'] = finta.TA.DEMA(df_count, 20, column='上证成交量')
            # 其他指标
            df['震荡幅度'] = (df_count['high'] - df_count['low']) / df_count['open']
            # 买单情况
            df['小单'] = df['小单买入量'] - df['小单卖出量']
            df['中单'] = df['中单买入量'] - df['中单卖出量']
            df['大单'] = df['大单买入量'] - df['大单卖出量']
            df['特大单'] = df['特大单买入量'] - df['特大单卖出量']
            # 修复数据
            df['量比'] = fix(df['量比'].values, 2)
            df['5分位成本'] = fix(df['5分位成本'].values, 1)
            df['15分位成本'] = fix(df['15分位成本'].values, 1)
            df['50分位成本'] = fix(df['50分位成本'].values, 1)
            df['85分位成本'] = fix(df['85分位成本'].values, 1)
            df['95分位成本'] = fix(df['95分位成本'].values, 1)
            df['加权平均成本'] = fix(df['加权平均成本'].values, 2)
            df['胜率'] = fix(df['胜率'].values, 2)
            # 补充数据
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
            df = df[20:]
            # 保存
            df.to_csv(f'{args.data_dir}/{name}_add.csv', header=True, index=True)
            print(f'| 结果保存至:{args.data_dir}/{name}_add.csv |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    data_deal(args)
