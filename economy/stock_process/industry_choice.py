import os
import yaml
import argparse
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------- #
# 筛选出某个行业的股票
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|选择股票|')
parser.add_argument('--industry', default='军工,医药,国产操作系统,存储芯片,机器人概念,旅游概念,电力,社区团购,粮食概念,跨境支付,跨境电商',
                    type=str, help='|选择行业，如"A,B,C"|')
parser.add_argument('--stock_path', default='dataset/stock_all.yaml', type=str, help='|股票列表|')
parser.add_argument('--data_dir', default='dataset/industry', type=str, help='|行业csv文件|')
parser.add_argument('--save_path', default='dataset/industry_choice.yaml', type=str, help='|保存位置|')
args_default = parser.parse_args()
args_default.industry = args_default.industry.split(',')
project_dir = os.path.dirname(os.path.dirname(__file__))
args_default.stock_path = project_dir + '/' + args_default.stock_path
args_default.data_dir = project_dir + '/' + args_default.data_dir
args_default.save_path = project_dir + '/' + args_default.save_path


# -------------------------------------------------------------------------------------------------------------------- #
class industry_choice_class:
    def __init__(self, args=args_default):
        self.industry = args.industry
        self.data_dir = args.data_dir
        self.save_path = args.save_path
        with open(args_default.stock_path, 'r', encoding='utf-8') as f:
            self.stock_dict = yaml.load(f, Loader=yaml.SafeLoader)

    def industry_choice(self):
        result_dict = {}
        record = 0
        for industry in self.industry:
            result_dict[industry] = {}
            path = f'{self.data_dir}/{industry}.csv'
            name_all = pd.read_csv(path)['股票'].values
            for name in name_all:
                result_dict[industry][name] = self.stock_dict[name]
                record += 1
        # 保存
        with open(self.save_path, 'w', encoding='utf-8') as f:
            yaml.dump(result_dict, f, allow_unicode=True, sort_keys=False)
        print(f'| 总数:{record} | 保存结果:{self.save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = industry_choice_class()
    model.industry_choice()
