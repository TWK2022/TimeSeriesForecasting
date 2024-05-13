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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# -------------------------------------------------------------------------------------------------------------------- #
# 集成
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|集成|')
parser.add_argument('--input_column', default='input_column.txt', type=str)
parser.add_argument('--input_size', default=96, type=int)
parser.add_argument('--output_size', default=24, type=int)
# economy/tushare/industry_choice.py
parser.add_argument('--industry_choice', default=False, type=bool)
parser.add_argument('--industry', default='船舶,航空,水运,电气设备', type=str)
# economy/tushare/data_get.py
parser.add_argument('--data_get', default=False, type=bool)
parser.add_argument('--token', default='', type=str)
parser.add_argument('--end_time', default='20240601', type=str)
# economy/data_screen.py
parser.add_argument('--data_screen', default=False, type=bool)
parser.add_argument('--close', default=1.2, type=float)
parser.add_argument('--change', default=1, type=float)
parser.add_argument('--volume', default=50000, type=float)
parser.add_argument('--volume_ratio', default=0.8, type=float)
# economy/data_deal.py
parser.add_argument('--data_deal', default=False, type=bool)
parser.add_argument('--delete_column', default='市盈率(ttm),市净率,市销率(ttm)', type=str)
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
parser.add_argument('--run_again', default=False, type=bool)
# def feature
parser.add_argument('--feature', default=False, type=bool)
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
        if self.args.data_screen:
            self._data_screen()
        if self.args.data_deal:
            self._data_deal()
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

    def _data_screen(self):
        print('economy/data_screen.py')
        os.system(f'python data_screen.py --close {self.args.close} --change {self.args.change}'
                  f' --volume {self.args.volume} --volume_ratio {self.args.volume_ratio}')

    def _data_deal(self):
        print('economy/data_deal.py')
        os.system(f'python data_deal.py --delete_column {self.args.delete_column}')

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
                os.system(f'python run.py --data_path {data_path} --input_column input_column.txt'
                          f' --output_column 收盘价_5 --input_size 96 --output_size {self.args.output_size}'
                          f' --divide 19,1 --weight {model_dir}/base_test.pt --weight_again True'
                          f' --model itransformer --model_type l --epoch 10 --lr_end_epoch 10')
                shutil.move('last.pt', f'{model_dir}/base_test.pt')

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
            model_dict[industry] = model_dict[industry] if model_dict.get(industry) else {}  # 初始化
            for name in name_list:
                data_path = f'{data_dir}/{name}_add.csv'
                model_path = f'{model_dir}/{name}.pt'
                if os.path.exists(model_path) and not self.args.run_test_again:
                    continue
                os.system(f'python run.py --data_path {data_path} --input_column input_column.txt'
                          f' --output_column output_column.txt --input_size 96 --output_size {self.args.output_size}'
                          f' --divide 19,1 --weight {model_dir}/base_test.pt --weight_again True'
                          f' --model itransformer --model_type l --epoch 30 --lr_end_epoch 30')
                shutil.move('last.pt', model_path)
                # 打开日志
                with open('log.txt', 'r', encoding='utf-8') as f:
                    log = f.readlines()
                val_loss = float(log[3].strip()[9:])
                # 记录模型信息
                df = pd.read_csv(data_path, index_col=0)
                time = str(df.index[-1])
                model_dict[industry][name] = [time, val_loss, None, None]
                with open('economy/model.yaml', 'w', encoding='utf-8') as f:
                    yaml.dump(model_dict, f, allow_unicode=True)

    def _simulate(self):
        print('simulate.py')
        with open('data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open('model.yaml', 'r', encoding='utf-8') as f:  # 模型信息
            model_dict = yaml.load(f, Loader=yaml.SafeLoader)
        for industry in screen_dict:
            name_list = screen_dict[industry].keys()
            for name in name_list:
                if model_dict[industry][name][1] > 0.3:  # 测试模型效果不好
                    continue
                os.system(f'python simulate.py --model_path model_test/{name}.pt --data_path dataset/{name}_add.csv'
                          f' --rise {self.args.rise}')
                # 打开日志
                with open('log.txt', 'r', encoding='utf-8') as f:
                    log = f.readlines()
                income_mean = round(float(log[1].strip()[8:]), 2)
                # 记录模型信息
                model_dict[industry][name][2] = income_mean
                with open('model.yaml', 'w', encoding='utf-8') as f:
                    yaml.dump(model_dict, f, allow_unicode=True)

    def _run_base(self, data_dir='economy/dataset', model_dir='economy/model'):
        print('run.py | 训练基础正式模型')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open('economy/data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        for industry in screen_dict:
            name_list = screen_dict[industry].keys()
            for name in name_list:
                data_path = f'{data_dir}/{name}_add.csv'
                os.system(f'python run.py --data_path {data_path} --input_column input_column.txt'
                          f' --output_column 收盘价_5 --input_size 96 --output_size {self.args.output_size}'
                          f' --divide 19,1 --divide_all True --weight {model_dir}/base.pt --weight_again True'
                          f' --model itransformer --model_type l --epoch 50 --lr_end_epoch 30')
                shutil.move('last.pt', f'{model_dir}/base.pt')

    def _run(self, data_dir='economy/dataset', model_dir='economy/model'):
        print('run.py | 训练正式模型')
        with open('economy/data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open('economy/model.yaml', 'r', encoding='utf-8') as f:  # 模型信息
            model_dict = yaml.load(f, Loader=yaml.SafeLoader)
        for industry in screen_dict:
            name_list = screen_dict[industry].keys()
            for name in name_list:
                data_path = f'{data_dir}/{name}_add.csv'
                model_path = f'{model_dir}/{name}.pt'
                if os.path.exists(model_path) and not self.args.run_again:
                    continue
                if model_dict[industry][name][1] > 0.3:  # 测试模型效果不好
                    continue
                os.system(f'python run.py --data_path {data_path} --input_column input_column.txt'
                          f' --output_column output_column.txt --input_size 96 --output_size {self.args.output_size}'
                          f' --divide 19,1 --divide_all True --weight {model_dir}/base.pt --weight_again True'
                          f' --model itransformer --model_type l --epoch 50 --lr_end_epoch 50')
                shutil.move('best.pt', model_path)
                # 记录模型信息
                df = pd.read_csv(data_path, index_col=0)
                time = str(df.index[-1])
                model_dict[industry][name][3] = time
                with open('model.yaml', 'w', encoding='utf-8') as f:
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
                if model_dict[industry][name][1] > 0.3:  # 测试模型效果不好
                    continue
                data_path = f'{data_dir}/{name}_add.csv'
                model_path = f'{model_dir}/{name}.pt'
                dict_ = torch.load(model_path, map_location='cpu')
                model = dict_['model']
                model = deploy(model, dict_['mean_input'], dict_['mean_output'], dict_['std_input'],
                               dict_['std_output']).eval()
                df = pd.read_csv(data_path, encoding='utf-8', index_col=0)
                input_data = np.array(df[input_column]).astype(np.float32).T
                input_data = input_data[:, -self.args.input_size:]
                tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                close_data = np.array(df['收盘价']).astype(np.float32)
                close_data = close_data[-200:]
                # 推理
                with torch.no_grad():
                    pred = model(tensor)[0][0].cpu().numpy()
                # 画图
                ratio = np.max(pred) / close_data[-1]
                if ratio > 1.1:  # 有上涨空间
                    last_day = str(df.index[-1])
                    judge = self._count(df['收盘价_5'].values, df['收盘价_10'].values)
                    save_path = f'save_image/{last_day}_{name}_{judge}_{ratio:.2f}_{model_dict[industry][name][2]}.jpg'
                    self._draw(pred, close_data, f'{last_day}_{name}', save_path)

    def _count(self, close_5, close_10):  # 判断金叉+和死叉1，+0表示今天金叉，+1表示昨天金叉
        for index in range(len(close_5) - 1, 0, -1):
            if close_5[index] >= close_10[index] and close_5[index - 1] < close_10[index - 1]:
                return f'+{len(close_5) - index + 1}'
            if close_5[index] <= close_10[index] and close_5[index - 1] > close_10[index - 1]:
                return f'-{len(close_5) - index + 1}'
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
