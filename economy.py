import os
import PIL
import yaml
import torch
import shutil
import prophet
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
parser.add_argument('--output_column', default='均价', type=str)
parser.add_argument('--input_size', default=96, type=int)
parser.add_argument('--output_size', default=24, type=int)
parser.add_argument('--model', default='tsf', type=str)
parser.add_argument('--model_type', default='l', type=str)
parser.add_argument('--device', default='cuda', type=str)
# economy/tushare/industry_choice.py
parser.add_argument('--industry_choice', default=False, type=bool)
parser.add_argument('--industry', default='数据安全,鸿蒙概念,跨境支付(CIPS)', type=str)
# economy/tushare/data_get.py
parser.add_argument('--data_get', default=False, type=bool)
parser.add_argument('--token', default='', type=str)
parser.add_argument('--end_time', default='20240801', type=str)
# economy/data_deal.py
parser.add_argument('--data_deal', default=False, type=bool)
# economy/data_screen.py
parser.add_argument('--data_screen', default=False, type=bool)
# run.py | 训练测试模型
parser.add_argument('--run_test', default=False, type=bool)
parser.add_argument('--run_test_again', default=False, type=bool)
# simulate.py
parser.add_argument('--simulate', default=False, type=bool)
parser.add_argument('--rise', default=1.02, type=float)
parser.add_argument('--rise_max', default=1.1, type=float)
# run.py | 训练正式模型
parser.add_argument('--run', default=False, type=bool)
parser.add_argument('--run_again', default=True, type=bool)
# def feature
parser.add_argument('--feature', default=False, type=bool)
parser.add_argument('--threshold', default=1.02, type=float)
parser.add_argument('--threshold_max', default=1.1, type=float)
parser.add_argument('--simulate_score', default=0, type=float)
parser.add_argument('--prophet', default=False, type=bool)
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
        if self.args.run_test:
            self._run_test()
        # economy目录
        os.chdir(self.path_economy)
        if self.args.simulate:
            self._simulate()
        # 原目录
        os.chdir(self.path)
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

    def _run_test(self, data_dir='economy/dataset', model_dir='economy/model_test'):
        print('run.py | 训练测试模型')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open('economy/data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        if os.path.exists('economy/model.yaml'):
            with open('economy/model.yaml', 'r', encoding='utf-8') as f:  # 模型信息
                model_dict = yaml.load(f, Loader=yaml.SafeLoader)
            model_dict = model_dict if model_dict else {}  # 初始化
        else:
            model_dict = {}
        all_dict = {}
        for industry in screen_dict:
            for key in screen_dict[industry].keys():
                all_dict[key] = screen_dict[industry][key]
        for name in all_dict.keys():
            data_path = f'{data_dir}/{name}_add.csv'
            model_path = f'{model_dir}/{name}.pt'
            weight = model_path
            epoch = 80
            lr_start = 0.001
            lr_end_ratio = 0.001
            if os.path.exists(model_path):
                if self.args.run_test_again or not model_dict.get(name):
                    epoch = 20
                    lr_start = 0.0001
                    lr_end_ratio = 0.01
                else:
                    continue
            os.system(f'python run.py --data_path {data_path} --input_column {self.args.input_column}'
                      f' --output_column {self.args.output_column} --input_size {self.args.input_size}'
                      f' --output_size {self.args.output_size} --divide 19,1 --z_score 1 --weight {weight}'
                      f' --weight_again True --model {self.args.model} --model_type {self.args.model_type}'
                      f' --batch 64 --epoch {epoch} --lr_start {lr_start} --lr_end_ratio {lr_end_ratio}'
                      f' --lr_end_epoch {epoch} --device {self.args.device}')
            shutil.move('last.pt', model_path)
            # 记录模型信息
            dict_ = torch.load(model_path, map_location='cpu')
            mae_true = round(float(dict_['val_mae'] * dict_['std_output'][0]), 4)
            df = pd.read_csv(data_path, index_col=0)
            time = str(df.index[-1])
            if model_dict.get(name):
                model_dict[name][0], model_dict[name][1] = time, mae_true
            else:
                model_dict[name] = [time, mae_true, None, None]
            del dict_, df
        with open('economy/model.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(model_dict, f, allow_unicode=True)

    def _simulate(self):
        print('simulate.py')
        with open('data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open('model.yaml', 'r', encoding='utf-8') as f:  # 模型信息
            model_dict = yaml.load(f, Loader=yaml.SafeLoader)
        all_dict = {}
        for industry in screen_dict:
            for key in screen_dict[industry].keys():
                all_dict[key] = screen_dict[industry][key]
        income_all = 0
        for name in all_dict.keys():
            model_path = f'model_test/{name}.pt'
            data_path = f'dataset/{name}_add.csv'
            if not os.path.exists(model_path):
                print(f'| 模型不存在:{model_path} |')
                continue
            os.system(f'python simulate.py --model_path {model_path} --data_path {data_path}'
                      f' --input_size {self.args.input_size} --output_size {self.args.output_size}'
                      f' --rise {self.args.rise} --rise_max {self.args.rise_max} --device {self.args.device}')
            # 打开日志
            with open('log.txt', 'r', encoding='utf-8') as f:
                log = f.readlines()
            income_mean = round(float(log[1].strip()[8:]), 3)
            # 记录模型信息
            if model_dict.get(name):
                model_dict[name][2] = income_mean
            else:
                model_dict[name] = [None, None, income_mean, None]
            # 统计
            income_all += income_mean
        with open('model.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(model_dict, f, allow_unicode=True)
        print(f'| 总收益:{income_all:.3f} |')

    def _run(self, data_dir='economy/dataset', model_dir='economy/model'):
        print('run.py | 训练正式模型')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open('economy/data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open('economy/model.yaml', 'r', encoding='utf-8') as f:  # 模型信息
            model_dict = yaml.load(f, Loader=yaml.SafeLoader)
        all_dict = {}
        for industry in screen_dict:
            for key in screen_dict[industry].keys():
                all_dict[key] = screen_dict[industry][key]
        for name in all_dict.keys():
            data_path = f'{data_dir}/{name}_add.csv'
            model_path = f'{model_dir}/{name}.pt'
            weight = model_path
            epoch = 80
            lr_start = 0.001
            lr_end_ratio = 0.001
            if os.path.exists(model_path):
                if self.args.run_again:
                    epoch = 20
                    lr_start = 0.0001
                    lr_end_ratio = 0.01
                else:
                    continue
            os.system(f'python run.py --data_path {data_path} --input_column {self.args.input_column}'
                      f' --output_column {self.args.output_column} --input_size {self.args.input_size}'
                      f' --output_size {self.args.output_size} --divide 19,1 --divide_train 1 --z_score 1'
                      f' --weight {weight} --weight_again True --model {self.args.model}'
                      f' --model_type {self.args.model_type} --batch 64 --epoch {epoch} --lr_start {lr_start}'
                      f' --lr_end_ratio {lr_end_ratio} --lr_end_epoch {epoch}'
                      f' --device {self.args.device}')  # 所有数据训练
            os.system(f'python run.py --data_path {data_path} --input_column {self.args.input_column}'
                      f' --output_column {self.args.output_column} --input_size {self.args.input_size}'
                      f' --output_size {self.args.output_size} --divide 4,1 --divide_train 2 --z_score 1'
                      f' --weight last.pt --weight_again True --model {self.args.model}'
                      f' --model_type {self.args.model_type} --batch 64 --epoch 20 --lr_start 0.0001'
                      f' --lr_end_epoch 20 --lr_end_ratio 0.01 --device {self.args.device}')  # 末尾数据加强训练
            shutil.move('last.pt', model_path)
            # 记录模型信息
            df = pd.read_csv(data_path, index_col=0)
            time = str(df.index[-1])
            if model_dict.get(name):
                model_dict[name][3] = time
            else:
                model_dict[name] = [None, None, 0, time]
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
                if not os.path.exists(model_path):
                    print(f'| 模型不存在:{model_path} |')
                    continue
                dict_ = torch.load(model_path, map_location='cpu')
                model = dict_['model']
                model = deploy(model, dict_['mean_input'], dict_['mean_output'], dict_['std_input'],
                               dict_['std_output']).eval()
                df = pd.read_csv(data_path, encoding='utf-8', index_col=0)
                input_data = df[input_column].values.astype(np.float32).T
                input_data = input_data[:, -self.args.input_size:]
                mean_data = df['均价'].values[-100:]
                increase = np.abs(df['涨跌幅'].values)[-1]
                tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
                # 推理
                with torch.no_grad():
                    pred = model(tensor)[0].cpu().numpy()
                pred_mean = pred[0]
                # 画图
                ratio = np.mean(pred_mean[0:3]) / mean_data[-1]  # 上涨幅度
                if model_dict.get(name):
                    simulate_score = model_dict[name][2]
                else:
                    simulate_score = 0
                if industry == '自选' or increase > 6 or (self.args.threshold < ratio < self.args.threshold_max
                                                        and simulate_score > self.args.simulate_score):  # 有上涨空间
                    last_day = str(df.index[-1])
                    save_path = f'save_image/{last_day}__{industry}__{name}__{ratio:.2f}__{simulate_score}.jpg'
                    self._draw(pred_mean, mean_data, f'{last_day}_{name}', save_path)
                    if self.args.prophet:
                        self._draw_prophet(df)
                        self._image_merge(save_path, 'save_image/_.jpg')

    def _draw(self, pred_mean, mean_data, name, save_path):
        zero = torch.zeros(len(mean_data))
        pred_mean = np.concatenate([zero, pred_mean], axis=0)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 使用字体时让坐标轴正常显示负号
        plt.title(name)
        plt.grid()
        plt.plot(mean_data, color='red', label='mean_data')
        plt.plot(pred_mean, color='green', label='pred_mean')
        plt.savefig(save_path)
        plt.close()
        print(f'| 画图保存位置:{save_path} |')

    def _draw_prophet(self, df):
        ds = df.index.values
        df = pd.DataFrame(df['收盘价'].values, columns=['y'])
        df['ds'] = ds
        model = prophet.Prophet()
        model.changepoint_prior_scale = 0.5
        model.fit(df)
        future = model.make_future_dataframe(periods=100)
        df_pred = model.predict(future)
        figure = model.plot_components(df_pred)
        figure.savefig('save_image/_.jpg')
        plt.close()

    def _image_merge(self, path1, path2):
        image1 = PIL.Image.open(path1).resize((1000, 500))
        image2 = PIL.Image.open(path2).resize((1000, 500))
        image = PIL.Image.new('RGB', (1000, 1000), (0, 0, 0))
        image.paste(image1, (0, 0))
        image.paste(image2, (0, 500))
        image.save(path1)


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = economy_class(args)
    model.predict()
