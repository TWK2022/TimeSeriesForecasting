import yaml
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|筛选有上升潜力的股票|')
parser.add_argument('--yaml_path', default='tushare/number.yaml', type=str, help='|选择的股票|')
parser.add_argument('--save_path', default='data_screen.yaml', type=str, help='|筛选结果保存位置|')
parser.add_argument('--history', default=100, type=int, help='|计算指标时采用最近history日内的数据|')
parser.add_argument('--close', default=1, type=float, help='|筛选价格<close*历史加权均值|')
parser.add_argument('--change', default=3, type=float, help='|筛选平均换手率，换手率和成交量要满足其一|')
parser.add_argument('--volume', default=100000, type=float, help='|筛选平均成交量，换手率和成交量要满足其一|')
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
            df = pd.read_csv(f'dataset/{name}_add.csv', index_col=0)
            close_data = df['收盘价'].values
            change_data = df['换手率'].values
            volume_data = df['成交量'].values
            # 检查是否存在nan值
            if np.isnan(df.values).any():
                print(f'| 存在nan值:dataset/{name}.csv |')
                continue
            # 上市日期删选
            if len(df) < 2 * args.history:
                continue
            # 加权均值
            ratio = 0.1 + 1.9 * np.arange(args.history) / (args.history - 1)
            # 收盘价筛选
            mean = np.mean(close_data[-args.history:] * ratio)
            if close_data[-1] / mean > args.close:
                continue
            # 换手率和成交量筛选
            change_mean = np.mean(change_data[-args.history:] * ratio)
            volume_mean = np.mean(volume_data[-args.history:] * ratio)
            if change_mean < args.change and volume_mean < args.volume:
                continue
            # 连续3天上涨
            if close_data[-1] > close_data[-2] > close_data[-3]:
                continue
            # 5日均线和10日均线筛选
            close_5 = df['收盘价_5'].values
            close_10 = df['收盘价_10'].values
            for index in range(len(close_5) - 1, 0, -1):
                if close_5[index] >= close_10[index] and close_5[index - 1] < close_10[index - 1]:  # 上穿
                    day = len(close_5) - index + 1
                    break
                if close_5[index] <= close_10[index] and close_5[index - 1] > close_10[index - 1]:  # 下穿
                    day = -(len(close_5) - index + 1)
                    break
            if day == -1:  # 刚刚下穿
                continue
            # 记录
            result_dict[industry][name] = float(round(close_data[-1] / mean, 2))
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
