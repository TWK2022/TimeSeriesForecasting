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
parser.add_argument('--industry', default='数据安全,鸿蒙概念,跨境支付(CIPS)', type=str, help='|行业或概念，如"A,B,C"|')
parser.add_argument('--drop_st', default=True, type=bool, help='|是否去除ST股票|')
args = parser.parse_args()
args.industry = args.industry.split(',')


# -------------------------------------------------------------------------------------------------------------------- #
def industry_choice(args):
    with open(args.yaml_path, 'r', encoding='utf-8') as f:
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    result_dict = {}
    record = 0
    remove_dict = {}
    # 自选股票
    if os.path.exists(args.reserve_path):
        with open(args.reserve_path, 'r', encoding='utf-8') as f:
            reserve_dict = yaml.load(f, Loader=yaml.SafeLoader)
        if reserve_dict is not None:
            result_dict['自选'] = reserve_dict['自选']
            record += len(reserve_dict['自选'])
    if os.path.exists(args.remove_path):
        with open(args.remove_path, 'r', encoding='utf-8') as f:
            remove_dict = yaml.load(f, Loader=yaml.SafeLoader)
    # 行业选择
    for industry in args.industry:
        result_dict[industry] = {}
        dict_ = yaml_dict[industry]
        for name in dict_.keys():
            number = dict_[name]
            if remove_dict is not None and remove_dict.get(industry):
                if name in remove_dict[industry].keys():
                    continue
            if args.drop_st and 'ST' in name:
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
