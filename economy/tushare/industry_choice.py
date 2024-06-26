import os
import yaml
import argparse

# -------------------------------------------------------------------------------------------------------------------- #
# 筛选出某个行业的股票
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|选择股票|')
parser.add_argument('--yaml_path', default='number_all.yaml', type=str, help='|所有股票信息|')
parser.add_argument('--reserve_path', default='reserve.yaml', type=str, help='|自选股票信息(可选)|')
parser.add_argument('--remove_path', default='remove.yaml', type=str, help='|要去除的股票信息(可选)|')
parser.add_argument('--save_path', default='number.yaml', type=str, help='|保存位置|')
parser.add_argument('--industry', default='互联网,电气设备,通信设备,半导体', type=str, help='|行业，必选，如"A,B,C"|')
parser.add_argument('--area', default='', type=str, help='|地区，空则不筛选，如"A、B、C"|')
parser.add_argument('--time', default='20240101', type=str, help='|上市时间，筛选time之前的，空则不筛选|')
parser.add_argument('--type', default='', type=str, help='|企业类型，有其他、中央国企、地方国企，空则不筛选|')
args = parser.parse_args()
args.industry = args.industry.split(',')
args.time = int(args.time) if args.time else 0


# -------------------------------------------------------------------------------------------------------------------- #
def industry_choice(args):
    with open(args.yaml_path, 'r', encoding='utf-8') as f:
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    result_dict = {}
    record = 0
    reserve_list = []
    remove_list = []
    # 自选股票
    if os.path.exists(args.reserve_path):
        with open(args.reserve_path, 'r', encoding='utf-8') as f:
            reserve_dict = yaml.load(f, Loader=yaml.SafeLoader)
        if reserve_dict is not None or result_dict['自选'] is not None:
            result_dict['自选'] = reserve_dict['自选']
            record += len(reserve_dict['自选'])
            reserve_list = reserve_dict['自选'].keys()
    if os.path.exists(args.remove_path):
        with open(args.remove_path, 'r', encoding='utf-8') as f:
            remove_dict = yaml.load(f, Loader=yaml.SafeLoader)
        if remove_dict is not None or result_dict['去除'] is not None:
            remove_list = remove_dict['去除'].keys()
    # 行业选择
    for industry in args.industry:
        result_dict[industry] = {}
        dict_ = yaml_dict[industry]
        for name in dict_.keys():
            number = dict_[name][0]
            type_ = dict_[name][2]
            time = dict_[name][3]
            if name in reserve_list:
                continue
            if name in remove_list:
                continue
            if 'ST' in name:
                continue
            if args.type and type_ != args.type:
                continue
            if args.time and time > args.time:
                continue
            result_dict[industry][name] = number
            record += 1
    # 保存
    with open(args.save_path, 'w', encoding='utf-8') as f:
        yaml.dump(result_dict, f, allow_unicode=True, sort_keys=False)
    print(f'| 总数:{record} | 保存结果至:{args.save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    industry_choice(args)
