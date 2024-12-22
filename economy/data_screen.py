import os
import yaml
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|筛选有上升潜力的股票|')
parser.add_argument('--number_path', default='tushare/number.yaml', type=str, help='|选择的股票|')
parser.add_argument('--save_path', default='data_screen.yaml', type=str, help='|筛选结果保存位置|')
parser.add_argument('--save_remove', default='remove.yaml', type=str, help='|记录收盘价、换手率、成交量不满足要求的股票|')
parser.add_argument('--close_min', default=3, type=float, help='|筛选价格>close_min|')
parser.add_argument('--close_max', default=30, type=float, help='|筛选价格<close_max|')
parser.add_argument('--change', default=3, type=float, help='|筛选近期最大换手率>change|')
parser.add_argument('--volume', default=200000, type=float, help='|筛选近期最大成交量>volume|')
parser.add_argument('--fluctuate', default=1.05, type=float, help='|筛选近期最高价/最低价>fluctuate|')
parser.add_argument('--reserve', default=False, type=bool, help='|自选股票是否需要筛选|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
def data_screen(args):
    with open(args.number_path, 'r', encoding='utf-8') as f:
        number_dict = yaml.load(f, Loader=yaml.SafeLoader)
    result_dict = {}
    remove_dict = {}
    record_all = 0
    record_screen = 0
    for industry in number_dict:
        industry_dict = number_dict[industry]
        record_all += len(industry_dict)
        result_dict[industry] = {}
        remove_dict[industry] = {}
        for name in industry_dict:
            if not os.path.exists(f'dataset/{name}_add.csv'):
                print(f'! 文件不存在:dataset/{name}_add.csv !')
                continue
            df = pd.read_csv(f'dataset/{name}_add.csv', index_col=0)
            close_data = df['收盘价'].values
            change_data = df['换手率'].values
            volume_data = df['成交量'].values
            # 检查是否存在nan值
            if np.isnan(df.values).any():
                print(f'! 存在nan值:dataset/{name}_add.csv !')
                continue
            # 上市日期删选
            if len(df) < 200:
                print(f'! 文件数据太少:dataset/{name}_add.csv !')
                continue
            # 自选股票
            if not args.reserve and industry == '自选':
                result_dict[industry][name] = industry_dict[name]
                record_screen += 1
                continue
            # 收盘价筛选
            if close_data[-1] < args.close_min or close_data[-1] > args.close_max:
                remove_dict[industry][name] = industry_dict[name]
                continue
            # 换手率筛选
            if np.max(change_data[-5:]) < args.change:
                remove_dict[industry][name] = industry_dict[name]
                continue
            # 成交量筛选
            if np.max(volume_data[-5:]) < args.volume:
                remove_dict[industry][name] = industry_dict[name]
                continue
            # 波动量筛选
            high = df['最高价'].values[-5:]
            low = df['最低价'].values[-5:]
            if np.max(high / low) < args.fluctuate:
                continue
            # 盈利情况筛选
            pe_ttm = df['r市盈率ttm'].values[-1]
            pb = df['r市净率'].values[-1]
            ps_ttm = df['r市销率ttm'].values[-1]
            if pe_ttm < 1 or pb == 0 or ps_ttm == 0:
                continue
            # KDJ指标筛选
            kdj_k = df['KDJ_K'].values[-2:]
            kdj_j = df['KDJ_J'].values[-2:]
            if kdj_k[-1] < kdj_j[-1] < kdj_j[-2]:
                continue
            if kdj_j[-1] < kdj_k[-1] and kdj_j[-2] > kdj_k[-2]:
                continue
            # 记录
            result_dict[industry][name] = industry_dict[name]
            record_screen += 1
    # 保存
    with open(args.save_path, 'w', encoding='utf-8') as f:
        yaml.dump(result_dict, f, allow_unicode=True, sort_keys=False)
    with open(args.save_remove, 'w', encoding='utf-8') as f:
        yaml.dump(remove_dict, f, allow_unicode=True, sort_keys=False)
    # 显示
    print(f'| 总数:{record_all} | 筛选数:{record_screen} |')
    print(f'| 结果保存至:{args.save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    data_screen(args)
