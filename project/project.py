import os
import torch
import argparse
import numpy as np
import pandas as pd
from model.layer import deploy
from block.util import read_column

# -------------------------------------------------------------------------------------------------------------------- #
# 交易策略:
# a开头是人为制定的策略，可以加入人为的经验
# b开头是根据模型预测结果制定的策略，考验模型的预测能力
# 由于当天得到数据时股票已经收盘，因此以今明两天收盘价的平均值作为实际交易股价，这里存在误差
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|测试|')
parser.add_argument('--model_path', default='../best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--data_path', default=r'../prepare/dataset/华银电力_add.csv', type=str, help='|数据位置|')
parser.add_argument('--input_column', default='../input_column.txt', type=str, help='|选择输入的变量，可传入.txt|')
parser.add_argument('--output_column', default='../output_column.txt', type=str, help='|选择预测的变量，可传入.txt|')
parser.add_argument('--input_size', default=96, type=int, help='|输入长度|')
parser.add_argument('--output_size', default=24, type=int, help='|输出长度|')
parser.add_argument('--divide', default='19,1', type=str, help='|训练集和验证集划分比例，取验证集测试|')
parser.add_argument('--rise', default=1.3, type=float, help='|上涨预期，大于预期才会买入，数值越大越保险，基准为1.3|')
parser.add_argument('--decline', default=0.95, type=float, help='|下跌预期，小于预期才会卖出，基准为0.95|')
parser.add_argument('--a_rise_max', default=1.3, type=float, help='|达到预期收益，直接卖出，数值太大时无效|')
parser.add_argument('--a_rise_still', default=1.05, type=float, help='|第2天发现还在上涨，先不卖出，数值太大时无效|')
parser.add_argument('--device', default='cpu', type=str, help='|推理设备|')
args = parser.parse_args()
args.divide = list(map(int, args.divide.split(',')))
args.input_column = read_column(args.input_column)  # column处理
args.output_column = read_column(args.output_column)  # column处理
# -------------------------------------------------------------------------------------------------------------------- #
assert os.path.exists(args.model_path), f'! model_path不存在:{args.model_path} !'
assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'


# -------------------------------------------------------------------------------------------------------------------- #
class project_class:
    def __init__(self, args):
        self.rise = args.rise
        self.decline = args.decline
        self.a_rise_max = args.a_rise_max
        self.a_rise_still = args.a_rise_still
        self.input_size = args.input_size
        self.output_size = args.output_size
        divide = args.divide
        model_path = args.model_path
        device = args.device
        data_path = args.data_path
        input_column = args.input_column
        output_column = args.output_column
        # 模型
        model_dict = torch.load(model_path, map_location='cpu')
        model = model_dict['model']
        model = deploy(model, model_dict['mean_input'], model_dict['mean_output'], model_dict['std_input'],
                       model_dict['std_output']).eval().to(device)
        self.model = model.half() if device == 'cuda' else model.float()
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
        self.input_data = np.array(df[input_column]).astype(np.float32).T[:, boundary:len(df)]
        self.output_data = np.array(df[output_column]).astype(np.float32).T[:, boundary:len(df)]
        # 记录
        self.state = None  # 买卖状态
        self.buy_list = None  # 买入价格
        self.sell_list = None  # 卖出价格

    def predict(self):  # 在不预知未来情况下的模型收益
        with torch.no_grad():
            self.state = 0
            self.buy_list = []
            self.sell_list = []
            for index in range(self.input_size, self.input_data.shape[1] - self.output_size):  # index是预测的第一步
                tensor = torch.tensor(self.input_data[:, index - self.input_size:index]).unsqueeze(0)
                pred = self.model(tensor)[0][0].cpu().numpy()
                now = self.output_data[0, index - 1]
                next_ = self.output_data[0, index]
                max_ = np.max(pred[3:-3])
                min_ = np.min(pred[3:-3])
                if max_ > self.rise * now:  # 预测上涨
                    self._rise(now, next_, pred)
                elif min_ < self.decline * now:  # 预测下降
                    self._decline(now, next_, pred)
                else:  # 预测小幅波动
                    pass
            self._metric('模型', now)

    def predict_true(self):  # 在预知未来情况下的完美收益
        self.state = 0
        self.buy_list = []
        self.sell_list = []
        for index in range(self.input_size, self.input_data.shape[1] - self.output_size):  # index是预测的第一步
            pred = self.output_data[0, index:index + self.output_size]
            now = self.output_data[0, index - 1]
            next_ = self.output_data[0, index]
            max_ = np.max(pred[3:-3])
            min_ = np.min(pred[3:-3])
            if max_ > self.rise * now:  # 预测上涨
                self._rise(now, next_, pred)
            elif min_ < self.decline * now:  # 预测下降
                self._decline(now, next_, pred)
            else:  # 预测小幅波动
                pass
        self._metric('完美', now)

    def _rise(self, now, next_, pred):
        if self.state == 1:  # 已经买入
            return
        # a人为策略
        pass
        # b模型策略
        if now > np.mean(pred[0:3]):  # 预测还有下降空间，先不买入
            return
        # 买入
        self.state = 1
        self.buy_list.append((now + next_) / 2)

    def _decline(self, now, next_, pred):
        if self.state == 0:  # 没有买入
            return
        # a人为策略
        if now > self.a_rise_max * self.buy_list[-1]:  # 达到预期收益，直接卖出
            self.state = 0
            self.sell_list.append(now + next_ / 2)
            return
        if now < self.a_rise_still * next_:  # 第2天发现依然有上涨趋势，先不卖出
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
        print(f'| {name} | 总收益率:{ratio_sum:.2f} | 单次操作收益率:{ratio_mean:.2f} |'
              f' 操作次数:{len(value)} | 亏损次数:{len(fault)} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = project_class(args)
    model.predict()
    model.predict_true()
