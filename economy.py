import os
import yaml
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.layer import deploy
from block.util import read_column

# -------------------------------------------------------------------------------------------------------------------- #
# 集成
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|集成|')
parser.add_argument('--input_column', default='input_column.txt', type=str)
parser.add_argument('--input_size', default=96, type=int)
parser.add_argument('--output_size', default=4, type=int)
parser.add_argument('--model', default='special_add', type=str)
parser.add_argument('--model_type', default='l', type=str)
# economy/tushare/industry_choice.py
parser.add_argument('--industry_choice', default=False, type=bool)
parser.add_argument('--industry', default='电气设备,运输设备,通信设备,工程机械,医疗保健,小金属,黄金,铝,铜,铅锌', type=str)
# economy/tushare/data_get.py
parser.add_argument('--data_get', default=False, type=bool)
parser.add_argument('--token', default='', type=str)
parser.add_argument('--end_time', default='20240601', type=str)
# economy/data_deal.py
parser.add_argument('--data_deal', default=False, type=bool)
# economy/data_screen.py
parser.add_argument('--data_screen', default=False, type=bool)
# run.py | 训练测试基础模型
parser.add_argument('--run_base_test', default=False, type=bool)
# run.py | 训练测试模型
parser.add_argument('--run_test', default=False, type=bool)
parser.add_argument('--run_test_again', default=False, type=bool)
# simulate.py
parser.add_argument('--simulate', default=False, type=bool)
parser.add_argument('--rise', default=1.1, type=float)
# run.py | 训练正式基础模型
parser.add_argument('--run_base', default=False, type=bool)
# run.py | 训练正式模型
parser.add_argument('--run', default=False, type=bool)
parser.add_argument('--run_again', default=True, type=bool)
# def feature
parser.add_argument('--feature', default=False, type=bool)
parser.add_argument('--draw_threshold', default=1.1, type=float)
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
class economy_class:
    def __init__(self, args):
        self.args = args
        self.path = os.getcwd()
        self.path_economy = f'{self.path}/economy'
        self.path_tushare = f'{self.path}/economy/tushare'

    def predict(self):
        # economy/tushare目录
        os.chdir(self.path_tushare)
        if self.args.industry_choice:
            self._industry_choice()
        if self.args.data_get:
            self._data_get()
        # economy目录
        os.chdir(self.path_economy)
        if self.args.data_deal:
            self._data_deal()
        if self.args.data_screen:
            self._data_screen()
        # 原目录
        os.chdir(self.path)
        if self.args.run_base_test:
            self._run_base_test()
        if self.args.run_test:
            self._run_test()
        # economy目录
        os.chdir(self.path_economy)
        if self.args.simulate:
            self._simulate()
        # 原目录
        os.chdir(self.path)
        if self.args.run_base:
            self._run_base()
        if self.args.run:
            self._run()
        if self.args.feature:
            self._feature()

    def _industry_choice(self):
        print('economy/tushare/industry_choice.py')
        os.system(f'python industry_choice.py --industry {self.args.industry}')

    def _data_get(self):
        print('economy/tushare/data_get.py')
        os.system(f'python data_get.py --token {self.args.token} --end_time {self.args.end_time}')

    def _data_deal(self):
        print('economy/data_deal.py')
        os.system(f'python data_deal.py')

    def _data_screen(self):
        print('economy/data_screen.py')
        os.system(f'python data_screen.py')

    def _run_base_test(self, data_dir='economy/dataset', model_dir='economy/model_test'):
        print('run.py | 训练测试基础模型')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open('economy/data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        for industry in screen_dict:
            name_list = screen_dict[industry].keys()
            for name in name_list:
                data_path = f'{data_dir}/{name}_add.csv'
                weight = f'{model_dir}/base_test.pt'
                epoch = 20
                os.system(f'python run.py --data_path {data_path} --input_column {self.args.input_column}'
                          f' --output_column 收盘价 --input_size {self.args.input_size}'
                          f' --output_size {self.args.output_size} --divide 19,1 --weight {weight}'
                          f' --weight_again True --model {self.args.model} --model_type {self.args.model_type}'
                          f' --epoch {epoch} --lr_end_epoch {epoch}')
                shutil.move('last.pt', weight)

    def _run_test(self, data_dir='economy/dataset', model_dir='economy/model_test'):
        print('run.py | 训练测试模型')
        assert os.path.exists(f'{model_dir}/base_test.pt')
        with open('economy/data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        if os.path.exists('economy/model.yaml'):
            with open('economy/model.yaml', 'r', encoding='utf-8') as f:  # 模型信息
                model_dict = yaml.load(f, Loader=yaml.SafeLoader)
            model_dict = model_dict if model_dict else {}  # 初始化
        else:
            model_dict = {}
        for industry in screen_dict:
            name_list = screen_dict[industry].keys()
            for name in name_list:
                data_path = f'{data_dir}/{name}_add.csv'
                model_path = f'{model_dir}/{name}.pt'
                weight = f'{model_dir}/base_test.pt'
                epoch = 30
                if os.path.exists(model_path):
                    if self.args.run_test_again or not model_dict.get(name):
                        weight = model_path
                        epoch = 10
                    else:
                        continue
                os.system(f'python run.py --data_path {data_path} --input_column {self.args.input_column}'
                          f' --output_column 收盘价 --input_size {self.args.input_size}'
                          f' --output_size {self.args.output_size} --divide 19,1 --weight {weight}'
                          f' --weight_again True --model {self.args.model} --model_type {self.args.model_type}'
                          f' --epoch {epoch} --lr_end_epoch {epoch}')
                shutil.move('last.pt', model_path)
                # 记录模型信息
                dict_ = torch.load(model_path, map_location='cpu')
                mae_true = round(float(dict_['val_mae'] * dict_['std_output']), 4)
                df = pd.read_csv(data_path, index_col=0)
                time = str(df.index[-1])
                if model_dict.get(name):
                    model_dict[name][0], model_dict[name][1] = time, mae_true
                else:
                    model_dict[name] = [time, mae_true, None, None]
                with open('economy/model.yaml', 'w', encoding='utf-8') as f:
                    yaml.dump(model_dict, f, allow_unicode=True)
                del dict_, df

    def _simulate(self):
        print('simulate.py')
        with open('data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open('model.yaml', 'r', encoding='utf-8') as f:  # 模型信息
            model_dict = yaml.load(f, Loader=yaml.SafeLoader)
        for industry in screen_dict:
            name_list = screen_dict[industry].keys()
            for name in name_list:
                model_path = f'model_test/{name}.pt'
                data_path = f'dataset/{name}_add.csv'
                os.system(f'python simulate.py --model_path {model_path} --data_path {data_path}'
                          f' --input_size {self.args.input_size} --output_size {self.args.output_size}'
                          f' --rise {self.args.rise}')
                # 打开日志
                with open('log.txt', 'r', encoding='utf-8') as f:
                    log = f.readlines()
                income_mean = round(float(log[1].strip()[8:]), 2)
                # 记录模型信息
                model_dict[name][2] = income_mean
                with open('model.yaml', 'w', encoding='utf-8') as f:
                    yaml.dump(model_dict, f, allow_unicode=True)

    def _run_base(self, data_dir='economy/dataset', model_dir='economy/model'):
        print('run.py | 训练正式基础模型')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open('economy/data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        for industry in screen_dict:
            name_list = screen_dict[industry].keys()
            for name in name_list:
                data_path = f'{data_dir}/{name}_add.csv'
                weight = f'{model_dir}/base.pt'
                epoch = 20
                os.system(f'python run.py --data_path {data_path} --input_column {self.args.input_column}'
                          f' --output_column 收盘价 --input_size {self.args.input_size}'
                          f' --output_size {self.args.output_size} --divide 19,1 --divide_train 1'
                          f' --weight {weight} --weight_again True --model {self.args.model}'
                          f' --model_type {self.args.model_type} --epoch {epoch} --lr_end_epoch {epoch}')
                shutil.move('last.pt', weight)

    def _run(self, data_dir='economy/dataset', model_dir='economy/model'):
        print('run.py | 训练正式模型')
        assert os.path.exists(f'{model_dir}/base.pt')
        with open('economy/data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open('economy/model.yaml', 'r', encoding='utf-8') as f:  # 模型信息
            model_dict = yaml.load(f, Loader=yaml.SafeLoader)
        for industry in screen_dict:
            name_list = screen_dict[industry].keys()
            for name in name_list:
                data_path = f'{data_dir}/{name}_add.csv'
                model_path = f'{model_dir}/{name}.pt'
                weight = f'{model_dir}/base.pt'
                epoch = 30
                if os.path.exists(model_path):
                    if self.args.run_again:
                        weight = model_path
                        epoch = 10
                    else:
                        continue
                os.system(f'python run.py --data_path {data_path} --input_column {self.args.input_column}'
                          f' --output_column 收盘价 --input_size {self.args.input_size}'
                          f' --output_size {self.args.output_size} --divide 19,1 --divide_train 2 --weight {weight}'
                          f' --weight_again True --model {self.args.model} --model_type {self.args.model_type}'
                          f' --epoch 50 --lr_end_epoch 50')  # 末尾数据加强训练
                os.system(f'python run.py --data_path {data_path} --input_column {self.args.input_column}'
                          f' --output_column 收盘价 --input_size {self.args.input_size}'
                          f' --output_size {self.args.output_size} --divide 19,1 --divide_train 1 --weight best.pt'
                          f' --weight_again True --model {self.args.model} --model_type {self.args.model_type}'
                          f' --epoch {epoch} --lr_end_epoch {epoch}')  # 所有数据训练
                shutil.move('best.pt', model_path)
                # 记录模型信息
                df = pd.read_csv(data_path, index_col=0)
                time = str(df.index[-1])
                model_dict[name][3] = time
                with open('economy/model.yaml', 'w', encoding='utf-8') as f:
                    yaml.dump(model_dict, f, allow_unicode=True)

    def _feature(self, data_dir='economy/dataset', model_dir='economy/model'):
        if not os.path.exists('save_image'):
            os.makedirs('save_image')
        with open('economy/data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open('economy/model.yaml', 'r', encoding='utf-8') as f:  # 模型信息
            model_dict = yaml.load(f, Loader=yaml.SafeLoader)
        input_column = read_column(self.args.input_column)  # column处理
        for industry in screen_dict:
            name_list = screen_dict[industry].keys()
            for name in name_list:
                data_path = f'{data_dir}/{name}_add.csv'
                model_path = f'{model_dir}/{name}.pt'
                dict_ = torch.load(model_path, map_location='cpu')
                model = dict_['model']
                model = deploy(model, dict_['mean_input'], dict_['mean_output'], dict_['std_input'],
                               dict_['std_output'], dict_['mean_special'], dict_['std_special']).eval()
                df = pd.read_csv(data_path, encoding='utf-8', index_col=0)
                input_data = np.array(df[input_column]).astype(np.float32).T
                input_data = input_data[:, -self.args.input_size:]
                close_data = np.array(df['收盘价']).astype(np.float32)[-100:]
                tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                special = torch.tensor(1.01 * close_data[-2:-1])  # 假设第2天开盘小涨
                # 推理
                with torch.no_grad():
                    if 'special' in self.args.model:
                        pred = model(tensor, special)[0][0].cpu().numpy()
                    else:
                        pred = model(tensor)[0][0].cpu().numpy()
                # 画图
                ratio = np.max(pred[0:5]) / close_data[-1]
                if ratio > self.args.draw_threshold or industry == '自选':  # 有上涨空间或自选股票
                    last_day = str(df.index[-1])
                    mean_judge = self._count(df['收盘价_SMA_5'].values, df['收盘价_SMA_10'].values)
                    save_path = f'save_image/{last_day}__{industry}__{name}__{mean_judge}__{ratio:.2f}' \
                                f'__{model_dict[name][2]}.jpg'
                    self._draw(pred, close_data, f'{last_day}_{name}', save_path)

    def _count(self, close_5, close_10):  # 判断金叉+和死叉-，+1表示今天金叉，-2表示昨天死叉
        for index in range(len(close_5) - 1, 0, -1):
            if close_5[index] >= close_10[index] and close_5[index - 1] < close_10[index - 1]:
                return f'+{len(close_5) - index}'
            if close_5[index] <= close_10[index] and close_5[index - 1] > close_10[index - 1]:
                return f'-{len(close_5) - index}'
        return 'None'

    def _draw(self, pred, close_data, name, save_path):
        zero = torch.zeros(len(close_data))
        pred = np.concatenate([zero, pred], axis=0)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 使用字体时让坐标轴正常显示负号
        plt.title(name)
        plt.grid()
        plt.plot(close_data, color='green', label='true')
        plt.plot(pred, color='cyan', label='pred')
        plt.savefig(save_path)
        plt.close()
        print(f'| 画图保存位置:{save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = economy_class(args)
    model.predict()
