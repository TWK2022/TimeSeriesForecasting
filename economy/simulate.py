import os
import torch
import argparse
import numpy as np
import pandas as pd
from model.layer import deploy
from block.util import read_column

# -------------------------------------------------------------------------------------------------------------------- #
# 交易策略:
# a开头是人为制定的策略，可以加入人为经验
# b开头是根据模型预测结果制定的策略，考验模型的预测能力
# c开头是人为制定的保守策略，通常会对c进行修改
# 由于当天得到数据时股票已经收盘，因此以今明两天收盘价的平均值作为实际交易股价，这里存在误差
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|测试|')
parser.add_argument('--model_path', default='../best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--data_path', default=r'dataset/华银电力_add.csv', type=str, help='|数据位置|')
parser.add_argument('--input_column', default='../input_column.txt', type=str, help='|选择输入的变量，可传入.txt|')
parser.add_argument('--input_size', default=96, type=int, help='|输入长度|')
parser.add_argument('--output_size', default=24, type=int, help='|输出长度|')
parser.add_argument('--divide', default='19,1', type=str, help='|训练集和验证集划分比例，取验证集测试|')
parser.add_argument('--device', default='cpu', type=str, help='|推理设备|')
parser.add_argument('--rise', default=1.2, type=float, help='|上涨预期，大于预期才会买入，数值越大越保险，基准为1.2|')
parser.add_argument('--a_rise_still', default=1.05, type=float, help='|第2天发现还在上涨，先不卖出，数值太大无效|')
parser.add_argument('--a_decline_still', default=0.97, type=float, help='|第2天发现还在下降，先不买入，数值太小无效|')
parser.add_argument('--safe', default=True, type=bool, help='|True时启用保守策略，后面的参数都为保守策略|')
parser.add_argument('--safe_mean', default=1.0, type=float, help='|股价小于[safe_mean*历史均值]才会买入，数值太大无效|')
args = parser.parse_args()
args.divide = list(map(int, args.divide.split(',')))
args.input_column = read_column(args.input_column)  # column处理
# -------------------------------------------------------------------------------------------------------------------- #
assert os.path.exists(args.model_path), f'! model_path不存在:{args.model_path} !'
assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'


# -------------------------------------------------------------------------------------------------------------------- #
class project_class:
    def __init__(self, args):
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.device = args.device
        self.rise = args.rise
        self.a_rise_still = args.a_rise_still
        self.a_decline_still = args.a_decline_still
        self.safe = args.safe
        self.safe_mean = args.safe_mean
        divide = args.divide
        model_path = args.model_path
        data_path = args.data_path
        input_column = args.input_column
        # 模型
        model_dict = torch.load(model_path, map_location='cpu')
        model = model_dict['model']
        model = deploy(model, model_dict['mean_input'], model_dict['mean_output'], model_dict['std_input'],
                       model_dict['std_output']).eval().to(self.device)
        self.model = model.half() if self.device == 'cuda' else model.float()
        epoch = model_dict['epoch_finished']
        mse = round(model_dict['val_mse'], 4)
        print(f'| 模型加载成功:{model_path} | epoch:{epoch} | mse:{mse}|')
        # 数据
        try:
            df = pd.read_csv(data_path, encoding='utf-8', index_col=0)
        except:
            df = pd.read_csv(data_path, encoding='gbk', index_col=0)
        add = self.input_size + self.output_size - 1  # 输入数据后面的补足
        data_len = len(df) - add  # 输入数据的真实数量
        boundary = int(data_len * divide[0] / (divide[0] + divide[1]))  # 数据划分
        self.input_data = np.array(df[input_column]).astype(np.float32).T[:, boundary:len(df)]  # 模型输入
        self.close_data = np.array(df['收盘价']).astype(np.float32).T[boundary:len(df)]  # 收盘价
        self.open_data = np.array(df['开盘价']).astype(np.float32).T[boundary:len(df)]  # 开盘价
        self.high_data = np.array(df['最高价']).astype(np.float32).T[boundary:len(df)]  # 最高价
        self.low_data = np.array(df['最低价']).astype(np.float32).T[boundary:len(df)]  # 最低价
        # 历史加权均值
        close_data = df['收盘价']
        ratio = 0.5 + np.arange(len(close_data)) / (len(close_data) - 1)
        self.mean = np.mean(close_data * ratio)
        # 记录
        self.state = None  # 买卖状态
        self.buy_list = None  # 买入价格
        self.sell_list = None  # 卖出价格

    def predict(self):  # 在不预知未来情况下的模型收益
        with torch.no_grad():
            self.state = 0
            self.buy_list = []
            self.sell_list = []
            for index in range(self.input_size, self.input_data.shape[1] - 1):  # index是预测的第一步
                tensor = torch.tensor(self.input_data[:, index - self.input_size:index]).unsqueeze(0).to(self.device)
                pred = self.model(tensor)[0][0].cpu().numpy()
                now = self.close_data[index - 1]
                next_ = self.close_data[index]
                max_ = np.max(pred)
                min_ = np.min(pred)
                if max_ > self.rise * now:  # 预测上涨
                    self._buy(now, next_, pred)
                elif min_ < now:  # 预测下降
                    self._sell(now, next_, pred)
                else:  # 预测小幅波动
                    pass
            self._metric('模型', now)

    def predict_true(self):  # 在预知未来情况下的理想收益
        self.state = 0
        self.buy_list = []
        self.sell_list = []
        for index in range(self.input_size, self.input_data.shape[1] - 1):  # index是预测的第一步
            pred = self.close_data[index:min(index + self.output_size, self.input_data.shape[1])]
            now = self.close_data[index - 1]
            next_ = self.close_data[index]
            max_ = np.max(pred)
            min_ = np.min(pred)
            if max_ > self.rise * now:  # 预测上涨
                self._buy(now, next_, pred)
            elif min_ < now:  # 预测下降
                self._sell(now, next_, pred)
            else:  # 预测小幅波动
                pass
        self._metric('理想', now)

    def _buy(self, now, next_, pred):
        if self.state == 1:  # 已经买入
            return
        # a人为策略
        if now * self.a_decline_still > next_:  # 第2天发现依然有下降趋势，先不买入
            return
        # b模型策略
        if now > np.mean(pred[0:3]):  # 预测还有下降空间，先不买入
            return
        # c保守策略
        if self.safe and now > self.safe_mean * self.mean:  # 股价处于历史高位，先不买入
            return
        # 买入
        self.state = 1
        self.buy_list.append((now + next_) / 2)

    def _sell(self, now, next_, pred):
        if self.state == 0:  # 没有买入
            return
        # a人为策略
        if now * self.a_rise_still < next_:  # 第2天发现依然有上涨趋势，先不卖出
            return
        # b模型策略
        if now < np.mean(pred[0:3]):  # 预测还有上升空间，先不卖出
            return
        # 卖出
        self.state = 0
        self.sell_list.append((now + next_) / 2)

    def _metric(self, name, now):  # 计算指标
        if len(self.buy_list) != len(self.sell_list):  # 最后一次还未卖出
            self.sell_list.append(now)
        buy = np.array(self.buy_list)
        sell = np.array(self.sell_list)
        value = sell - buy
        ratio_sum = np.sum(value / buy) if len(buy) else 0
        ratio_mean = np.mean(value / buy) if len(buy) else 0
        fault = value[np.where(value < 0, True, False)]
        print(f'| {name} | rise:{self.rise} | 总收益率:{ratio_sum:.2f} | 单次操作收益率:{ratio_mean:.2f} |'
              f' 操作次数:{len(value)} | 亏损次数:{len(fault)} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = project_class(args)
    model.predict()
    model.predict_true()
