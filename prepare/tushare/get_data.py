import tushare
import argparse

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='||')
parser.add_argument('--token', default='', type=str, help='|密钥|')
args = parser.parse_args()

# -------------------------------------------------------------------------------------------------------------------- #


if __name__ == '__main__':
    tushare.set_token(args.token)  # 设置密钥
    pro = tushare.pro_api()  # 初始化接口
    df = pro.daily(ts_code='000001.SZ', start_date='20140101', end_date='20240201')
