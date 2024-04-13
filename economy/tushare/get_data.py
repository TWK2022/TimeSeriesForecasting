import os
import yaml
import tushare
import argparse
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 获取数据的时间格式为20240101，不能为2024-01-01
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|通过tushare获取最新股票数据以补全数据库|')
parser.add_argument('--token', default='80d83d7d966f3c4461771a4d56c55948c0752aac3606540d3f35dbcf', type=str,
                    help='|密钥|')
parser.add_argument('--save_path', default='../dataset', type=str, help='|数据保存的目录|')
parser.add_argument('--number', default='number.yaml', type=str, help='|选用的股票|')
parser.add_argument('--start_time', default='20170101', type=str, help='|开始时间|')
parser.add_argument('--end_time', default='20240601', type=str, help='|结束时间|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# -------------------------------------------------------------------------------------------------------------------- #
class get_data_class:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8')as f:
            self.number_dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.daily_column = ['trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol']
        self.daily_name = ['日期', '开盘价', '最高价', '最低价', '收盘价', '昨收价(前复权)', '涨跌额', '涨跌幅', '成交量']
        self.daily_basic_column = 'trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe_ttm,pb,ps_ttm'
        self.daily_basic_name = ['日期', '换手率', '换手率(自由流通股)', '量比', '市盈率(ttm)', '市净率', '市销率(ttm)']

    def get_data(self):
        tushare.set_token(args.token)  # 设置密钥
        pro = tushare.pro_api()  # 初始化接口
        for key in self.number_dict:
            path = f'{args.save_path}/{key}.csv'
            if os.path.exists(path):
                df_old = pd.read_csv(path, index_col=0)
                end_time_old = df_old.index[-1]
                if pd.DatetimeIndex([end_time_old])[0] >= pd.DatetimeIndex([args.end_time])[0]:
                    print(f'| 已是最新数据: {path} |')
                    continue
                else:
                    print(f'| 补充数据: {path} |')
                    df = self._tushare_to_df(pro, self.number_dict, key, end_time_old)
                    df = df.drop(index=end_time_old)
                    df = pd.concat([df_old, df])
                    df.index = pd.DatetimeIndex(df.index)
                    df.to_csv(path, index=True, header=True)
            else:
                print(f'| 新增数据: {path} |')
                df = self._tushare_to_df(pro, self.number_dict, key, args.start_time)
                df.to_csv(path, index=True, header=True)

    def _tushare_to_df(self, pro, number_dict, key, start_time):
        start_time = start_time.replace('-', '')
        # 基础信息
        df = pro.daily(ts_code=number_dict[key], start_date=start_time, end_date=args.end_time)[self.daily_column]
        df.columns = self.daily_name
        df.index = pd.DatetimeIndex(df['日期'])
        df = df.drop(columns='日期')
        df = df.sort_index()
        # 指标
        df_ = pro.daily_basic(ts_code=number_dict[key], start_date=start_time, end_date=args.end_time,
                              fields=self.daily_basic_column)
        df_.columns = self.daily_basic_name
        df_.index = pd.DatetimeIndex(df_['日期'])
        df_ = df_.drop(columns='日期')
        df_ = df_.sort_index()
        # 合并
        df = pd.concat([df, df_], axis=1)
        df = df.dropna(axis=0, how='any')
        return df


if __name__ == '__main__':
    model = get_data_class(args.number)
    model.get_data()
