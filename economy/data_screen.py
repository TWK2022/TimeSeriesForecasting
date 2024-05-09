import yaml
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|筛选有上升潜力的股票|')
parser.add_argument('--yaml_path', default='tushare/number.yaml', type=str, help='|选择的股票|')
parser.add_argument('--save_path', default='data_screen.yaml', type=str, help='|筛选结果保存位置|')
parser.add_argument('--threshold', default=0.9, type=float, help='|筛选指标小于threshold的股票，1表示<=历史加权均值|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
def data_screen(yaml_path, save_path, threshold):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    result_dict = {}
    record_all = 0
    record_screen = 0
    for industry in yaml_dict:
        industry_dict = yaml_dict[industry]
        record_all += len(industry_dict)
        result_dict[industry] = {}
        for key in industry_dict:
            # 筛选
            df = pd.read_csv(f'dataset/{key}.csv', index_col=0)
            # 收盘价筛选
            close_data = df['收盘价'].values
            ratio = 0.1 + 1.9 * np.arange(len(close_data)) / (len(close_data) - 1)
            mean = np.mean(close_data * ratio)
            close_metric = np.mean(close_data[-3:]) / mean
            if close_metric > threshold:
                continue
            # 记录
            result_dict[industry][key] = float(round(close_metric, 2))
            record_screen += 1
        result_dict[industry] = dict(sorted(result_dict[industry].items(), key=lambda x: x[1]))
    # 保存
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(result_dict, f, allow_unicode=True, sort_keys=False)
    # 显示
    print(f'| 总数:{record_all} | 筛选数:{record_screen} |')
    print(f'| 结果保存至:{save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    data_screen(args.yaml_path, args.save_path, args.threshold)
