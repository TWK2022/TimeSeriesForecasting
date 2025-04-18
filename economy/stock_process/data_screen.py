import os
import yaml
import tqdm
import argparse
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 筛选有上升潜力的股票
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|筛选有上升潜力的股票|')
parser.add_argument('--industry_choice', default='dataset/industry_choice.yaml', type=str, help='|待筛选股票|')
parser.add_argument('--data_dir', default='dataset/stock_add', type=str, help='|股票数据|')
parser.add_argument('--save_path', default='dataset/industry_screen.yaml', type=str, help='|保存位置|')
parser.add_argument('--close_min', default=3, type=float, help='|价格>close_min|')
parser.add_argument('--close_max', default=100, type=float, help='|价格<close_max|')
parser.add_argument('--market_value_min', default=50, type=float, help='|总市值(亿)>market_value_min|')
parser.add_argument('--market_value_max', default=1000, type=float, help='|总市值(亿)<market_value_max|')
parser.add_argument('--change', default=3, type=float, help='|近期最大换手率>change|')
parser.add_argument('--date_line', default=True, type=bool, help='|日线筛选|')
args_default = parser.parse_args()
project_dir = os.path.dirname(os.path.dirname(__file__))
args_default.industry_choice = project_dir + '/' + args_default.industry_choice
args_default.data_dir = project_dir + '/' + args_default.data_dir
args_default.save_path = project_dir + '/' + args_default.save_path


# -------------------------------------------------------------------------------------------------------------------- #
class data_screen_class:
    def __init__(self, args=args_default):
        self.data_dir = args.data_dir
        self.save_path = args.save_path
        self.close_min = args.close_min
        self.close_max = args.close_max
        self.market_value_min = args.market_value_min
        self.market_value_max = args.market_value_max
        self.change = args.change
        self.date_line = args.date_line
        with open(args.industry_choice, 'r', encoding='utf-8') as f:
            self.industry_choice = yaml.load(f, Loader=yaml.SafeLoader)

    def data_screen(self):
        result_dict = {}
        for industry in tqdm.tqdm(self.industry_choice.keys()):
            result_dict[industry] = {}
            for name in self.industry_choice[industry].keys():
                path = self.data_dir + f'/{name}_add.csv'
                if not os.path.exists(path):
                    print(f'! 文件不存在:{path} !')
                    continue
                df = pd.read_csv(path, index_col=0)
                close = df['收盘价'].values
                market_value = df['总市值'].values
                change = df['换手率'].values
                close_5 = df['收盘价_SMA_5'].values
                close_10 = df['收盘价_SMA_10'].values
                mean_value = df['均价']
                # 检查是否存在nan值
                if np.isnan(df.values).any():
                    print(f'! 存在nan值:{path} !')
                    continue
                # 上市日期删选
                if len(df) < 200:
                    print(f'! 文件数据太少:{path} !')
                    continue
                # 股价筛选
                if close[-1] < self.close_min or close[-1] > self.close_max:
                    continue
                # 市值筛选
                if market_value[-1] / 10000 < self.market_value_min or market_value[-1] / 10000 > self.market_value_max:
                    continue
                # 换手率筛选
                if np.max(change[-5:]) < self.change:
                    continue
                # 日线筛选
                if self.date_line and (mean_value[-1] > close_5[-1] and mean_value[-1] > close_10[-1]):
                    continue
                # 记录
                result_dict[industry][name] = self.industry_choice[industry][name]
        # 保存
        with open(self.save_path, 'w', encoding='utf-8') as f:
            yaml.dump(result_dict, f, allow_unicode=True, sort_keys=False)
        print(f'| 保存结果:{self.save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = data_screen_class()
    model.data_screen()
