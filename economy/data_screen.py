import os
import yaml
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|筛选有上升潜力的股票|')
parser.add_argument('--yaml_path', default='tushare/number.yaml', type=str, help='|选择的股票|')
parser.add_argument('--save_path', default='data_screen.yaml', type=str, help='|筛选结果保存位置|')
parser.add_argument('--close', default=5, type=float, help='|筛选价格>close|')
parser.add_argument('--change', default=2, type=float, help='|筛选近期换手率>change|')
parser.add_argument('--volume', default=80000, type=float, help='|筛选近期成交量>volume|')
parser.add_argument('--fluctuate', default=1.03, type=float, help='|筛选近期最高价/最低价>fluctuate|')
parser.add_argument('--other', default=False, type=bool, help='|自选股票是否需要筛选|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
def data_screen(args):
    with open(args.yaml_path, 'r', encoding='utf-8') as f:
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    result_dict = {}
    record_all = 0
    record_screen = 0
    for industry in yaml_dict:
        industry_dict = yaml_dict[industry]
        record_all += len(industry_dict)
        result_dict[industry] = {}
        for name in industry_dict:
            if not os.path.exists(f'dataset/{name}_add.csv'):
                print(f'| 文件不存在:dataset/{name}_add.csv |')
                continue
            df = pd.read_csv(f'dataset/{name}_add.csv', index_col=0)
            close_data = df['收盘价'].values
            close_SMA_5 = df['收盘价_SMA_5'].values
            change_data = df['换手率'].values
            volume_data = df['成交量'].values
            # 检查是否存在nan值
            if np.isnan(df.values).any():
                print(f'| 存在nan值:dataset/{name}_add.csv |')
                continue
            # 上市日期删选
            if len(df) < 200:
                continue
            # 自选股票
            if not args.other and industry == '自选':
                result_dict[industry][name] = float(round(close_data[-1] / close_SMA_5[-1], 2))
                record_screen += 1
                continue
            # 加权均值
            ratio = np.linspace(0.1, 1.9, 10)
            # 收盘价筛选
            if close_data[-1] < args.close:
                continue
            # 换手率筛选
            change_mean = np.mean(change_data[-10:] * ratio)
            if change_mean < args.change:
                continue
            # 成交量筛选
            volume_mean = np.mean(volume_data[-10:] * ratio)
            if volume_mean < args.volume:
                continue
            # 波动量筛选
            high = df['最高价'].values[-10:]
            low = df['最低价'].values[-10:]
            if np.mean(high / low) < args.fluctuate:
                continue
            # 盈利情况筛选
            pe_ttm = df['r市盈率ttm'].values[-1]
            pb = df['r市净率'].values[-1]
            ps_ttm = df['r市销率ttm'].values[-1]
            if pe_ttm == 0 or pb == 0 or ps_ttm == 0:
                continue
            # 记录
            result_dict[industry][name] = float(round(close_data[-1] / close_SMA_5[-1], 2))
            record_screen += 1
        result_dict[industry] = dict(sorted(result_dict[industry].items(), key=lambda x: x[1]))
    # 保存
    with open(args.save_path, 'w', encoding='utf-8') as f:
        yaml.dump(result_dict, f, allow_unicode=True, sort_keys=False)
    # 显示
    print(f'| 总数:{record_all} | 筛选数:{record_screen} |')
    print(f'| 结果保存至:{args.save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    data_screen(args)
