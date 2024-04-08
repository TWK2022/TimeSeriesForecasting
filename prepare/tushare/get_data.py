import os
import yaml
import tushare
import argparse
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 获取数据的时间格式为20240101，不能为2024-01-01
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|通过tushare获取最新股票数据以补全数据库|')
parser.add_argument('--token', default='', type=str, help='|密钥|')
parser.add_argument('--save_path', default='dataset', type=str, help='|数据保存的目录|')
parser.add_argument('--number', default='number.yaml', type=str, help='|选用的股票|')
parser.add_argument('--start_time', default='20150101', type=str, help='|开始时间|')
parser.add_argument('--end_time', default='20240420', type=str, help='|结束时间|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
def read_yaml(path):  # 读取yaml文件
    with open(path, 'r', encoding='utf-8')as f:
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    return yaml_dict


def tushare_to_df(pro, number_dict, key, start_time):
    start_time = start_time.replace('-', '')
    df = pro.daily(ts_code=number_dict[key], start_date=start_time, end_date=args.end_time)
    df = df.drop(columns='ts_code')
    df.columns = ['日期', '开盘价', '最高价', '最低价', '收盘价', '昨收价', '涨跌额', '涨跌幅', '成交量', '成交额']
    df.index = pd.DatetimeIndex(df['日期'])
    df = df.drop(columns='日期')
    df = df.sort_index()
    df.dropna(axis=0, how='any')
    return df


def get_data(number_dict):
    tushare.set_token(args.token)  # 设置密钥
    pro = tushare.pro_api()  # 初始化接口
    for key in number_dict:  # 补充旧数据
        path = f'{args.save_path}/{key}.csv'
        if os.path.exists(path):
            df_old = pd.read_csv(path, index_col=0)
            end_time_old = df_old.index[-1]  # 最后一天
            if pd.DatetimeIndex([end_time_old])[0] >= pd.DatetimeIndex([args.end_time])[0]:  # 已经是最新数据
                print(f'| 已是最新数据: {path} |')
                continue
            else:
                print(f'| 补充数据: {path} |')
                df = tushare_to_df(pro, number_dict, key, end_time_old)
                df = df.drop(index=end_time_old)
                df = pd.concat([df_old, df])
                df.index = pd.DatetimeIndex(df.index)
                df.to_csv(path, index=True, header=True)
        else:
            print(f'| 新增数据: {path} |')
            df = tushare_to_df(pro, number_dict, key, args.start_time)
            df.to_csv(path, index=True, header=True)


if __name__ == '__main__':
    number_dict = read_yaml(args.number)  # 读取选用的股票号
    get_data(number_dict)
