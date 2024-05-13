import yaml
import argparse

# -------------------------------------------------------------------------------------------------------------------- #
# 筛选出某个行业的股票
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|选择股票|')
parser.add_argument('--yaml_path', default='number_all.yaml', type=str, help='|所有股票信息|')
parser.add_argument('--save_path', default='number.yaml', type=str, help='|保存位置|')
parser.add_argument('--industry', default='汽车整车', type=str, help='|行业，必选，如"A,B,C"|')
parser.add_argument('--area', default='', type=str, help='|地区，空则不筛选，如"A、B、C"|')
parser.add_argument('--time', default='', type=str, help='|上市时间，筛选time之前的，空则不筛选|')
parser.add_argument('--type', default='其他', type=str, help='|企业类型，有其他、中央国企、地方国企，空则不筛选|')
args = parser.parse_args()
args.industry = args.industry.split(',')
args.time = int(args.time) if args.time else 0


# -------------------------------------------------------------------------------------------------------------------- #
def industry_choice(args):
    with open(args.yaml_path, 'r', encoding='utf-8') as f:
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    result_dict = {}
    for industry in args.industry:
        result_dict[industry] = {}
        dict_ = yaml_dict[industry]
        for name in dict_.keys():
            number = dict_[name][0]
            area = dict_[name][2]
            type_ = dict_[name][3]
            time = dict_[name][4]
            if 'ST' in name:
                continue
            if args.area and area not in args.area:
                continue
            if args.type and type_ != args.type:
                continue
            if args.time and time < args.time:
                continue
            result_dict[industry][name] = number
    with open(args.save_path, 'w', encoding='utf-8') as f:
        yaml.dump(result_dict, f, allow_unicode=True)
    print(f'| 保存结果至:{args.save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    industry_choice(args)
