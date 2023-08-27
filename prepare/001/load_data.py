import os
import yaml
import WindPy
import argparse
import numpy as np
import pandas as pd

# 股票数据使用前复权，只获取交易日数据
# wind试用版调用一次函数大概只能获取2500个数据，多了会报错
# 除此之外每日和每周都有总额限定，超了之后会报错，要等一段时间再试
# 数据精度为float64
# -------------------------------------------------------------------------------------------------------------------- #
# 设置已是最新数据
parser = argparse.ArgumentParser(description='|从WindPy获取最新股票数据以补全数据库|')
parser.add_argument('--save_path', default='dataset', type=str, help='|数据保存的位置|')
parser.add_argument('--number', default='number.yaml', type=str, help='|选用的股票|')
parser.add_argument('--variable', default='variable.yaml', type=str, help='|选用的变量|')
parser.add_argument('--start_time', default='2010-01-01', type=str, help='|开始时间|')
parser.add_argument('--end_time', default='2023-08-26', type=str, help='|结束时间，不超过当前日期的前一天|')
parser.add_argument('--max_data', default=2500, type=int, help='|wind函数每次的获取上限，用于将数据划分为多次获取|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.save_path), f'| os.path.exists(args.save_path) = {os.path.exists(args.save_path)} |'
print(f'| {args.start_time} --> {args.end_time} |')
# -------------------------------------------------------------------------------------------------------------------- #
# 测试
# WindPy.w.start()  # 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒
# assert WindPy.w.isconnected(), f'| WindPy.w.isconnected()={WindPy.w.isconnected()} |'
# wind_data = WindPy.w.wsd('600000.SH', 'open', '2010-01-01', '2010-02-01', 'Currency=CNY;PriceAdj=F')


# -------------------------------------------------------------------------------------------------------------------- #
def read_yaml(path):  # 读取yaml文件
    with open(path, 'r', encoding='utf-8')as f:
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    return yaml_dict


def deal(number_dict, variable_dict):
    name_list = list(number_dict.keys())
    number_list = list(number_dict.values())
    variable_tuple = variable_dict.items()
    column_list = [_[0] for _ in variable_tuple]  # 中文变量名
    variable_list = [_[1]['参数'] for _ in variable_tuple]  # WindPy中变量参数
    option_list = [_[1]['设置'] for _ in variable_tuple]  # WindPy中变量设置
    variable = ';'.join(variable_list)
    option = ';'.join(option_list).replace('None;', '')
    return name_list, number_list, column_list, variable, option


def wind_to_df(number, column, variable, option, start_time, end_time, divide):
    date = pd.date_range(start_time, end_time, freq='D')
    df_list = []
    for i in range(len(date) // divide):
        start = date[i * divide]
        end = date[(i + 1) * divide - 1]
        wind_data = WindPy.w.wsd(number, variable, start, end, f'{option};Currency=CNY;PriceAdj=F')
        data = np.array(wind_data.Data, dtype=np.float64).T
        index = wind_data.Times
        df_list.append(pd.DataFrame(data, columns=column, index=index, dtype=np.float64))
    start = date[len(date) - len(date) % divide]
    end = end_time
    wind_data = WindPy.w.wsd(number, variable, start, end, f'{option};Currency=CNY;PriceAdj=F')
    data = np.array(wind_data.Data, dtype=np.float64).T
    index = wind_data.Times
    df_list.append(pd.DataFrame(data, columns=column, index=index, dtype=np.float64))
    df = pd.concat(df_list, axis=0)
    return df


def get_data(number_dict, variable_dict, start_time, end_time, save_path, divide):  # 从WindPy获取数据
    name_list, number_list, column_list, variable, option = deal(number_dict, variable_dict)  # 参数处理
    WindPy.w.start()  # 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒
    assert WindPy.w.isconnected(), f'| WindPy.w.isconnected()={WindPy.w.isconnected()} |'
    for name, number in zip(name_list, number_list):
        file_name = f'{save_path}/{name}.csv'
        if os.path.exists(file_name):  # 补充旧数据
            df_old = pd.read_csv(file_name, index_col=0)
            column_old = list(df_old.columns)
            assert column_old == column_list, f'(column_old == column_list) = {column_old == column_list}'
            start_time_choose = df_old.index[-1]  # 最后一天
            if pd.DatetimeIndex([start_time_choose])[0] >= pd.DatetimeIndex([end_time])[0]:  # 已经是最新数据
                print(f'| 已是最新数据: {file_name} |')
                continue
            else:
                print(f'| 补充数据: {file_name} |')
            df_add = wind_to_df(number, column_list, variable, option, start_time_choose, end_time, divide)[1:]
            df = pd.concat([df_old, df_add], axis=0)
            df.to_csv(file_name, index=True, header=True)
        else:  # 增加新a数据
            print(f'| 新增数据: {file_name} |')
            df = wind_to_df(number, column_list, variable, option, start_time, end_time, divide)
            df.to_csv(file_name, index=True, header=True)


# 程序args
if __name__ == '__main__':
    number_dict = read_yaml(args.number)  # 读取选用的股票号
    variable_dict = read_yaml(args.variable)  # 读取选用的变量
    divide = int(args.max_data // (len(variable_dict) * 0.7))  # 计算函数每次能获取的最大日期数，0.7是因为约0.7的比例为交易日
    get_data(number_dict, variable_dict, args.start_time, args.end_time, args.save_path, divide)
