import os
import time
import yaml
import tqdm
import tushare
import argparse
import datetime
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 需要tushare账户有5000积分
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|tushare|')
parser.add_argument('--token', default='', type=str, help='|密钥|')
parser.add_argument('--save_dir', default='dataset', type=str, help='|数据根目录|')
parser.add_argument('--industry_start', default=['885', '886'], type=list, help='|只记录x开头的行业|')
parser.add_argument('--code_start', default=['00', '60'], type=list, help='|只记录x开头的股票|')
parser.add_argument('--drop_st', default=True, type=bool, help='|去除ST股票|')
parser.add_argument('--frequency', default=200, type=int, help='|API每分钟可以调取的频率|')
parser.add_argument('--start_time', default='20180101', type=str, help='|开始时间|')
args_default, _ = parser.parse_known_args()
args_default.end_time = str(datetime.datetime.now().date()).replace('-', '')  # 当天日期
args_default.save_dir = os.path.dirname(os.path.dirname(__file__)) + '/' + args_default.save_dir
if not os.path.exists(args_default.save_dir):
    os.makedirs(args_default.save_dir)


# -------------------------------------------------------------------------------------------------------------------- #
class tushare_block_class:
    '''
        stock_classification: 获取同花顺中股票的分类
        stock_performance: 获取股票的业绩预报
        data_get: 获取股票的日线数据
    '''

    def __init__(self, args=args_default):
        tushare.set_token(args.token)  # 设置密钥
        self.pro = tushare.pro_api()  # 初始化接口
        self.drop_st = args.drop_st  # 是否去除ST股票
        self.save_dir = args.save_dir  # 保存路径的根目录
        self.frequency = args.frequency  # API每分钟可以调取的频率
        self.code_start = args.code_start  # 只记录code_start开头的股票
        self.start_time = args.start_time  # 获取股票数据的开始时间
        self.end_time = args.end_time  # 获取股票数据的结束时间
        self.industry_start = args.industry_start  # 只记录code_start开头的股票
        # data_get设置
        self.daily_column = ['trade_date', 'open', 'high', 'low', 'close', 'change', 'pct_chg', 'vol']
        self.daily_name = ['日期', '开盘价', '最高价', '最低价', '收盘价', '涨跌额', '涨跌幅', '成交量']
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

    def stock_classification(self, save_path='stock_classification.yaml'):
        save_path = self.save_dir + '/' + save_path
        # 获取分类的类别
        df = self.pro.ths_index(fields=['ts_code', 'name'])
        keep_dict = {}
        for index, ts_code in enumerate(df.values[:, 0]):
            if ts_code[0:3] in self.industry_start:
                keep_dict[df.values[index, 1]] = ts_code
        # 获取类别中的股票
        record_time = 0
        result_dict = {}
        time_start = time.time()
        for industry, ts_code in tqdm.tqdm(keep_dict.items()):
            df = self.pro.ths_member(ts_code=ts_code, fields=['con_code', 'con_name'])
            industry_dict = {}
            for code, name in df.values:
                if self.drop_st and 'ST' in name:
                    continue
                if code[0:2] not in self.code_start:
                    continue
                industry_dict[name] = code
            result_dict[industry] = industry_dict
            record_time += 1
            if record_time % self.frequency == 0:
                time_end = time.time()
                if time_end - time_start < 61:
                    time.sleep(61 + time_start - time_end)
                time_start = time.time()
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(result_dict, f, allow_unicode=True, sort_keys=False)

    def stock_performance(self, save_path='stock_performance.yaml', classification_path='stock_classification.yaml'):
        save_path = self.save_dir + '/' + save_path
        classification_path = self.save_dir + '/' + classification_path
        # 获取股票业绩
        df = self.pro.forecast_vip(period='20241231',
                                   fields=['ts_code', 'end_date', 'net_profit_min', 'net_profit_max'])
        df = df.dropna(axis=0, how='any')  # 删除所有含Nan的行
        df.columns = ['股票代码', '报告日期', '利润下限', '利润上限']
        # 所有股票
        with open(classification_path, 'r', encoding='utf-8') as f:
            stock_classification = yaml.load(f, Loader=yaml.SafeLoader)
        all_dict = {}  # code->名称
        for industry in stock_classification:
            for name in stock_classification[industry].keys():
                all_dict[stock_classification[industry][name]] = name
        # 记录
        if os.path.exists(save_path):  # 更新数据
            with open(save_path, 'r', encoding='utf-8') as f:
                stock_performance = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            stock_performance = {}
        for index in range(len(df)):
            value = df.iloc[index].values
            code = value[0]
            if all_dict.get(code) is None:
                continue
            stock_performance[all_dict[code]] = [value[1], int((value[2] + value[3]) / 2)]
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(stock_performance, f, allow_unicode=True, sort_keys=False)

    def data_get(self, screen_path='industry_choice.yaml', save_path='stock'):
        screen_path = self.save_dir + '/' + screen_path
        save_path = self.save_dir + '/' + save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 所有股票
        with open(screen_path, 'r', encoding='utf-8') as f:
            yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
        all_dict = {}
        for industry in yaml_dict:
            for name in yaml_dict[industry].keys():
                all_dict[name] = yaml_dict[industry][name]
        # 上证指数
        df = self.pro.index_daily(ts_code='000001.SH', start_date=self.start_time, end_date=self.end_time,
                                  fields=['trade_date', 'close', 'vol'])
        df = pd.DataFrame(df.values[:, 1:3], columns=['上证指数', '上证成交量'],
                          index=df['trade_date'].values).sort_index()
        df.to_csv(f'{self.save_dir}/上证指数.csv', index=True, header=True)
        print(f'| 补充数据:{self.save_dir}/上证指数.csv |')
        self.end_time = df.index[-1]
        # 股票数据
        record_time = 0
        time_start = time.time()
        for name, ts_code in tqdm.tqdm(all_dict.items()):
            path = f'{save_path}/{name}.csv'
            if os.path.exists(path):
                df_old = pd.read_csv(path, index_col=0)
                end_time_old = df_old.index[-1]
                if pd.DatetimeIndex([end_time_old])[0] >= pd.DatetimeIndex([self.end_time])[0]:
                    # print(f'| 已是最新数据: {path} |')
                    continue
                else:
                    # print(f'| 补充数据: {path} |')
                    df = self._data_get(ts_code, end_time_old)
                    df = df.drop(index=end_time_old)
                    df = pd.concat([df_old, df])
                    df.index = pd.DatetimeIndex(df.index)
                    df.to_csv(path, index=True, header=True)
                    record_time += 1
            else:
                # print(f'| 新增数据: {path} |')
                df = self._data_get(ts_code, self.start_time)
                df.to_csv(path, index=True, header=True)
                record_time += 1
            if record_time % self.frequency == 0:
                time_end = time.time()
                if time_end - time_start < 61:
                    time.sleep(61 + time_start - time_end)
                time_start = time.time()

    def _data_get(self, ts_code, start_time):
        start_time = start_time.replace('-', '')
        # 基础信息
        df = self.pro.daily(ts_code=ts_code, start_date=start_time, end_date=self.end_time, fields=self.daily_column)
        df.columns = self.daily_name
        df.index = pd.DatetimeIndex(df['日期'].values)
        df = df.drop(columns='日期')
        df = df.sort_index()
        # 指标
        df1 = self.pro.daily_basic(ts_code=ts_code, start_date=start_time, end_date=self.end_time,
                                   fields=self.daily_basic_column)
        df1.columns = self.daily_basic_name
        df1.index = pd.DatetimeIndex(df1['日期'].values)
        df1 = df1.drop(columns='日期')
        df1 = df1.sort_index()
        # 技术指标
        df2 = self.pro.stk_factor(ts_code=ts_code, start_date=start_time, end_date=self.end_time,
                                  fields=self.stk_factor_column)
        df2.columns = self.stk_factor_name
        df2.index = pd.DatetimeIndex(df2['日期'].values)
        df2 = df2.drop(columns='日期')
        df2 = df2.sort_index()
        # 资金流向
        df3 = self.pro.moneyflow(ts_code=ts_code, start_date=start_time, end_date=self.end_time,
                                 fields=self.moneyflow_column)
        df3.columns = self.moneyflow_name
        df3.index = pd.DatetimeIndex(df3['日期'].values)
        df3 = df3.drop(columns='日期')
        df3 = df3.sort_index()
        # 筹码分布
        df4 = self.pro.cyq_perf(ts_code=ts_code, start_date=start_time, end_date=self.end_time,
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
    tushare_block = tushare_block_class()
    tushare_block.data_get()
