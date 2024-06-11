import os
import time
import yaml
import tushare
import argparse
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 获取数据的时间格式为20240101，不能为2024-01-01
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|通过tushare获取最新股票数据以补全数据库|')
parser.add_argument('--token', default='', type=str, help='|密钥|')
parser.add_argument('--save_path', default='../dataset', type=str, help='|数据保存的目录|')
parser.add_argument('--number', default='number.yaml', type=str, help='|选用的股票|')
parser.add_argument('--start_time', default='20170101', type=str, help='|开始时间|')
parser.add_argument('--end_time', default='20240701', type=str, help='|结束时间|')
parser.add_argument('--frequency', default=200, type=int, help='|API每分钟可以调取的频率|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# -------------------------------------------------------------------------------------------------------------------- #
class data_get_class:
    def __init__(self, args):
        self.args = args
        with open(args.number, 'r', encoding='utf-8') as f:
            self.number_dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.daily_column = ['trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'vol']
        self.daily_name = ['日期', '开盘价', '最高价', '最低价', '收盘价', '昨收价(前复权)', '涨跌幅', '成交量']
        self.daily_basic_column = ['trade_date', 'turnover_rate', 'volume_ratio', 'pe_ttm', 'pb', 'ps_ttm']
        self.daily_basic_name = ['日期', '换手率', '量比', '市盈率ttm', '市净率', '市销率ttm']

    def data_get(self):
        tushare.set_token(self.args.token)  # 设置密钥
        pro = tushare.pro_api()  # 初始化接口
        # 指数数据
        df = pro.index_daily(ts_code='000001.SH', start_date=self.args.start_time, end_date=self.args.end_time,
                             fields=['trade_date', 'close', 'vol'])
        df = pd.DataFrame(df.values[:, 1:3], columns=['上证指数', '上证成交量'],
                          index=df['trade_date'].values).sort_index()
        df.to_csv(f'{self.args.save_path}/上证指数.csv', index=True, header=True)
        print(f'| 补充数据: {self.args.save_path}/上证指数.csv |')
        # 股票数据
        record_time = 1
        time_start = time.time()
        for industry in self.number_dict:
            industry_dict = self.number_dict[industry]
            for key in industry_dict:
                path = f'{self.args.save_path}/{key}.csv'
                if os.path.exists(path):
                    df_old = pd.read_csv(path, index_col=0)
                    end_time_old = df_old.index[-1]
                    if pd.DatetimeIndex([end_time_old])[0] >= pd.DatetimeIndex([self.args.end_time])[0]:
                        print(f'| 已是最新数据: {path} |')
                        continue
                    else:
                        print(f'| 补充数据: {path} |')
                        record_time += 1
                        df = self._tushare_to_df(pro, industry_dict, key, end_time_old)
                        df = df.drop(index=end_time_old)
                        df = pd.concat([df_old, df])
                        df.index = pd.DatetimeIndex(df.index)
                        df.to_csv(path, index=True, header=True)
                else:
                    print(f'| 新增数据: {path} |')
                    record_time += 1
                    df = self._tushare_to_df(pro, industry_dict, key, self.args.start_time)
                    df.to_csv(path, index=True, header=True)
                if record_time % self.args.frequency == 0:
                    time_end = time.time()
                    if time_end - time_start < 60:
                        time.sleep(60 + time_start - time_end)

    def _tushare_to_df(self, pro, industry_dict, key, start_time):
        start_time = start_time.replace('-', '')
        # 基础信息
        df = pro.daily(ts_code=industry_dict[key], start_date=start_time,
                       end_date=self.args.end_time, fields=self.daily_column)
        df.columns = self.daily_name
        df.index = pd.DatetimeIndex(df['日期'].values)
        df = df.drop(columns='日期')
        df = df.sort_index()
        # 指标
        df_ = pro.daily_basic(ts_code=industry_dict[key], start_date=start_time, end_date=self.args.end_time,
                              fields=self.daily_basic_column)
        df_.columns = self.daily_basic_name
        df_.index = pd.DatetimeIndex(df_['日期'].values)
        df_ = df_.drop(columns='日期')
        df_ = df_.sort_index()
        # 合并
        df = pd.concat([df, df_], axis=1)
        return df


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = data_get_class(args)
    model.data_get()
