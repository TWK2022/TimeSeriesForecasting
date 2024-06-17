import os
import torch
import argparse
import numpy as np
import pandas as pd
from model.layer import deploy
from block.util import read_column

# -------------------------------------------------------------------------------------------------------------------- #
# 交易策略:
# a开头是人为制定的策略，可以加入人为经验，这里存在主观性，不一定能达到
# b开头是根据模型预测结果制定的策略，考验模型的预测能力
# 实际买入和卖出时的股价存在误差
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|测试|')
parser.add_argument('--special', default=True, type=bool, help='|特殊模型|')
parser.add_argument('--model_path', default='../last.pt', type=str, help='|pt模型位置|')
parser.add_argument('--data_path', default=r'dataset/XX_add.csv', type=str, help='|数据位置|')
parser.add_argument('--input_column', default='../input_column.txt', type=str, help='|选择输入的变量，可传入.txt|')
parser.add_argument('--input_size', default=96, type=int, help='|输入长度|')
parser.add_argument('--output_size', default=12, type=int, help='|输出长度|')
parser.add_argument('--divide', default='19,1', type=str, help='|训练集和验证集划分比例，取验证集测试|')
parser.add_argument('--device', default='cpu', type=str, help='|推理设备|')
parser.add_argument('--rise', default=1.1, type=float, help='|上涨预期，大于预期才会买入，数值越大越保险，基准为1.1|')
parser.add_argument('--a', default=True, type=bool, help='|使用人为策略|')
args = parser.parse_args()
args.divide = list(map(int, args.divide.split(',')))
args.input_column = read_column(args.input_column)  # column处理
# -------------------------------------------------------------------------------------------------------------------- #
assert os.path.exists(args.model_path), f'! model_path不存在:{args.model_path} !'
assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'


# -------------------------------------------------------------------------------------------------------------------- #
class project_class:
    def __init__(self, args):
        self.args = args
        divide = args.divide
        model_path = args.model_path
        data_path = args.data_path
        input_column = args.input_column
        # 模型
        model_dict = torch.load(model_path, map_location='cpu')
        model = model_dict['model']
        model = deploy(model, model_dict['mean_input'], model_dict['mean_output'], model_dict['std_input'],
                       model_dict['std_output'], model_dict['mean_special'],
                       model_dict['std_special']).eval().to(self.args.device)
        self.model = model.half() if self.args.device == 'cuda' else model.float()
        print(f'| 模型:{model_path} | train_loss:{model_dict["train_loss"]:.4f} |'
              f'val_loss:{model_dict["val_loss"]:.4f} |')
        # 数据
        try:
            df = pd.read_csv(data_path, encoding='utf-8', index_col=0)
        except:
            df = pd.read_csv(data_path, encoding='gbk', index_col=0)
        add = self.args.input_size + self.args.output_size - 1  # 输入数据后面的补足
        data_len = len(df) - add  # 输入数据的真实数量
        boundary = int(data_len * divide[0] / (divide[0] + divide[1]))  # 数据划分
        self.input_data = np.array(df[input_column]).astype(np.float32).T[:, boundary:len(df)]  # 模型输入
        self.close_data = np.array(df['收盘价']).astype(np.float32).T[boundary:len(df)]  # 收盘价
        self.open_data = np.array(df['开盘价']).astype(np.float32).T[boundary:len(df)]  # 开盘价
        self.high_data = np.array(df['最高价']).astype(np.float32).T[boundary:len(df)]  # 最高价
        self.low_data = np.array(df['最低价']).astype(np.float32).T[boundary:len(df)]  # 最低价
        self.close_10_data = np.array(df['收盘价_SMA_10']).astype(np.float32).T[boundary:len(df)]  # 10日均线
        # 记录
        self.state = None  # 买卖状态
        self.buy_list = None  # 买入价格
        self.sell_list = None  # 卖出价格

    def predict(self):  # 在不预知未来情况下的模型收益
        with (torch.no_grad()):
            self.state = 0
            self.buy_list = []
            self.sell_list = []
            for index in range(self.args.input_size, self.input_data.shape[1] - 1):  # index是预测的第1步
                tensor = torch.tensor(self.input_data[:, index - self.args.input_size:index]
                                      ).unsqueeze(0).to(self.args.device)
                special = torch.tensor(self.open_data[index:index + 1]).to(self.args.device)
                if self.args.special:
                    pred = self.model(tensor, special)[0][0].cpu().numpy()
                else:
                    pred = self.model(tensor)[0][0].cpu().numpy()
                now = self.close_data[index - 1]
                if self.state == 0:  # 准备买入
                    self._buy(index, pred)
                elif self.state == 1:  # 准备卖出
                    self._sell(index, pred)
            self._metric('模型', now, True)

    def predict_true(self):  # 在预知未来情况下的理想收益
        self.state = 0
        self.buy_list = []
        self.sell_list = []
        for index in range(self.args.input_size, self.input_data.shape[1] - 1):  # index是预测的第1步
            pred = self.close_data[index:min(index + self.args.output_size, self.input_data.shape[1])]
            now = self.close_data[index - 1]
            if self.state == 0:  # 准备买入
                self._buy(index, pred)
            elif self.state == 1:  # 准备卖出
                self._sell(index, pred)
        self._metric('理想', now, False)

    def _buy(self, index, pred):
        now_10 = self.close_10_data[index - 1]
        next_low = self.low_data[index]
        buy_value = next_low * 1.02  # 理想买入价格
        # a人为策略
        if self.args.a:
            if buy_value > 1.1 * now_10:  # 股价处于历史高位，先不买入
                return
        # b模型策略
        if buy_value * self.args.rise > np.max(pred[0:3]):  # 预测上涨幅度不大，先不买入
            return
        if buy_value > np.mean(pred[0:3]):  # 预测还有下降空间，先不买入
            return
        # 买入
        self.state = 1
        self.buy_list.append(buy_value)

    def _sell(self, index, pred):
        now = self.close_data[index - 1]
        next_open = self.open_data[index]
        next_high = self.high_data[index]
        sell_value = next_high * 0.98  # 理想卖出价格
        # a人为策略
        if self.args.a:
            if next_open < now * 0.99 and next_high < now:  # 第2天发现有明显下降趋势，直接卖出
                self.state = 0
                self.sell_list.append(sell_value)
                return
        # b模型策略
        if sell_value < np.mean(pred[0:3]):  # 预测还有上升空间，先不卖出
            return
        # 卖出
        self.state = 0
        self.sell_list.append(sell_value)

    def _metric(self, name, now, record=False):  # 计算指标
        sell_last = 0
        if len(self.buy_list) != len(self.sell_list):  # 最后一次还未卖出
            self.sell_list.append(now)
            sell_last = 1
        buy = np.array(self.buy_list)
        sell = np.array(self.sell_list)
        value = sell - buy
        income_sum = np.sum(value / buy) if len(buy) else 0
        income_mean = np.mean(value / buy) if len(buy) else 0
        fault = value[np.where(value < 0, True, False)]
        print(f'| {name} | rise:{self.args.rise} | 总收益率:{income_sum:.2f} | 单次操作收益率:{income_mean:.2f} |'
              f' 操作次数:{len(value)} | 亏损次数:{len(fault)} | 最后是否持有:{sell_last} |')
        # 记录日志
        if record:
            with open('log.txt', 'w', encoding='utf-8') as f:
                f.write(f'总收益率:{income_sum:.2f}\n单次操作收益率:{income_mean:.2f}\n'
                        f'操作次数:{len(value)}\n亏损次数:{len(fault)}\n最后是否持有:{sell_last}')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = project_class(args)
    model.predict()
    model.predict_true()
