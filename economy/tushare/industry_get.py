import os
import time
import yaml
import tqdm
import tushare
import argparse

# -------------------------------------------------------------------------------------------------------------------- #
# 获取同花顺的行业和概念分类，得到number_all.yaml
# 需要5000积分
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|tushare获取同花顺的行业和概念分类|')
parser.add_argument('--token', default='', type=str, help='|密钥|')
parser.add_argument('--save_path', default='number_all.yaml', type=str, help='|保存位置|')
parser.add_argument('--start_code', default=['00', '60'], type=list, help='|只记录start_code开头的股票|')
parser.add_argument('--drop_st', default=True, type=bool, help='|是否去除ST股票|')
parser.add_argument('--frequency', default=200, type=int, help='|API每分钟可以调取的频率|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# -------------------------------------------------------------------------------------------------------------------- #
def industry_get(args):
    tushare.set_token(args.token)  # 设置密钥
    pro = tushare.pro_api()  # 初始化接口
    # 获取行业分类数据
    df = pro.ths_index(fields='ts_code,name')
    value = df.values
    result_dict = {}
    for index, ts_code in enumerate(value[:, 0]):
        if ts_code[0:3] == '881' or ts_code[0:3] == '885' or ts_code[0:3] == '886':
            result_dict[value[index, 1]] = value[index, 0]
    # 获取行业股票数据
    record_time = 1
    time_start = time.time()
    for industry, ts_code in tqdm.tqdm(result_dict.items()):
        df = pro.ths_member(ts_code=ts_code, fields='code,name')
        industry_dict = {}
        for code, name in df.values:
            if args.drop_st and 'ST' in name:
                continue
            if code[0:2] not in args.start_code:
                continue
            industry_dict[name] = code
        result_dict[industry] = industry_dict
        if record_time % args.frequency == 0:
            time_end = time.time()
            if time_end - time_start < 60:
                time.sleep(60 + time_start - time_end)
            time_start = time.time()
    with open(args.save_path, 'w', encoding='utf-8') as f:
        yaml.dump(result_dict, f, allow_unicode=True, sort_keys=False)


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    industry_get(args)
