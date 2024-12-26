import os
import time
import yaml
import tushare
import argparse
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 获取股票数据以补全数据库
# 获取数据的时间格式为20240101，不能为2024-01-01
# 需要5000积分
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|tushare获取最新股票数据以补全数据库|')
parser.add_argument('--token', default='', type=str, help='|密钥|')
parser.add_argument('--save_path', default='../dataset', type=str, help='|数据保存的目录|')
parser.add_argument('--number', default='number.yaml', type=str, help='|选用的股票|')
parser.add_argument('--start_time', default='20180101', type=str, help='|开始时间|')
parser.add_argument('--end_time', default='20250101', type=str, help='|结束时间|')
parser.add_argument('--frequency', default=90, type=int, help='|API每分钟可以调取的频率|')
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
        self.daily_column = ['trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol', 'amount']
        self.daily_name = ['日期', '开盘价', '最高价', '最低价', '收盘价', '涨跌额', '涨跌幅', '成交量', '成交额']
        self.daily_basic_column = ['trade_date', 'turnover_rate', 'volume_ratio', 'pe_ttm', 'pb', 'ps_ttm', 'total_mv']
        self.daily_basic_name = ['日期', '换手率', '量比', '市盈率ttm', '市净率', '市销率ttm', '总市值']
        self.stk_factor_column = ['trade_date', 'kdj_k', 'kdj_d', 'kdj_j', 'rsi_6', 'rsi_12', 'rsi_24']
        self.stk_factor_name = ['日期', 'KDJ_K', 'KDJ_D', 'KDJ_J', 'RSI_6', 'RSI_12', 'RSI_24']
        self.moneyflow_column = ['trade_date', 'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol', 'buy_lg_vol',
                                 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol', 'net_mf_vol', 'trade_count']
        self.moneyflow_name = ['日期', '小单买入量', '小单卖出量', '中单买入量', '中单卖出量', '大单买入量', '大单卖出量',
                               '特大单买入量', '特大单卖出量', '净流入量', '交易笔数']
        self.distribution_column = ['trade_date', 'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct',
                                    'weight_avg', 'winner_rate']
        self.distribution_name = ['日期', '5分位成本', '15分位成本', '50分位成本', '85分位成本', '95分位成本', '加权平均成本',
                                  '胜率']

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
        self.args.end_time = df.index[-1]
        # 股票数据
        all_dict = {}
        for industry in self.number_dict:
            for key in self.number_dict[industry].keys():
                all_dict[key] = self.number_dict[industry][key]
        record_time = 1
        time_start = time.time()
        for name, ts_code in all_dict.items():
            path = f'{self.args.save_path}/{name}.csv'
            if os.path.exists(path):
                df_old = pd.read_csv(path, index_col=0)
                end_time_old = df_old.index[-1]
                if pd.DatetimeIndex([end_time_old])[0] >= pd.DatetimeIndex([self.args.end_time])[0]:
                    print(f'| 已是最新数据: {path} |')
                    continue
                else:
                    print(f'| 补充数据: {path} |')
                    df = self._tushare_to_df(pro, ts_code, end_time_old)
                    df = df.drop(index=end_time_old)
                    df = pd.concat([df_old, df])
                    df.index = pd.DatetimeIndex(df.index)
                    df.to_csv(path, index=True, header=True)
                    record_time += 1
            else:
                print(f'| 新增数据: {path} |')
                df = self._tushare_to_df(pro, ts_code, self.args.start_time)
                df.to_csv(path, index=True, header=True)
                record_time += 1
            if record_time % self.args.frequency == 0:
                time_end = time.time()
                if time_end - time_start < 60:
                    time.sleep(61 + time_start - time_end)
                time_start = time.time()

    def _tushare_to_df(self, pro, ts_code, start_time):
        start_time = start_time.replace('-', '')
        # 基础信息
        df = pro.daily(ts_code=ts_code, start_date=start_time, end_date=self.args.end_time, fields=self.daily_column)
        df.columns = self.daily_name
        df.index = pd.DatetimeIndex(df['日期'].values)
        df = df.drop(columns='日期')
        df = df.sort_index()
        # 指标
        df1 = pro.daily_basic(ts_code=ts_code, start_date=start_time, end_date=self.args.end_time,
                              fields=self.daily_basic_column)
        df1.columns = self.daily_basic_name
        df1.index = pd.DatetimeIndex(df1['日期'].values)
        df1 = df1.drop(columns='日期')
        df1 = df1.sort_index()
        # 技术指标
        df2 = pro.stk_factor(ts_code=ts_code, start_date=start_time, end_date=self.args.end_time,
                             fields=self.stk_factor_column)
        df2.columns = self.stk_factor_name
        df2.index = pd.DatetimeIndex(df2['日期'].values)
        df2 = df2.drop(columns='日期')
        df2 = df2.sort_index()
        # 资金流向
        df3 = pro.moneyflow(ts_code=ts_code, start_date=start_time, end_date=self.args.end_time,
                            fields=self.moneyflow_column)
        df3.columns = self.moneyflow_name
        df3.index = pd.DatetimeIndex(df3['日期'].values)
        df3 = df3.drop(columns='日期')
        df3 = df3.sort_index()
        # 筹码分布
        df4 = pro.cyq_perf(ts_code=ts_code, start_date=start_time, end_date=self.args.end_time,
                           fields=self.distribution_column)
        df4.columns = self.distribution_name
        df4.index = pd.DatetimeIndex(df4['日期'].values)
        df4 = df4.drop(columns='日期')
        df4 = df4.sort_index()
        # 合并
        df = pd.concat([df, df1, df2, df3, df4], axis=1)
        return df


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = data_get_class(args)
    model.data_get()
