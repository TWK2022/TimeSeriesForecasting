import os
import yaml
import shutil
import argparse

# -------------------------------------------------------------------------------------------------------------------- #
# 集成
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|集成|')
# economy/tushare/industry_choice.py
parser.add_argument('--industry_choice', default=False, type=bool)
parser.add_argument('--industry', default='船舶,航空', type=str)
# economy/tushare/data_get.py
parser.add_argument('--data_get', default=False, type=bool)
parser.add_argument('--token', default='', type=str)
parser.add_argument('--end_time', default='20240601', type=str)
# economy/data_screen.py
parser.add_argument('--data_screen', default=False, type=bool)
parser.add_argument('--threshold', default=0.9, type=float)
# economy/data_add.py
parser.add_argument('--data_add', default=False, type=bool)
# run.py | 训练基础测试模型
parser.add_argument('--run_base_test', default=False, type=bool)
# run.py | 训练测试模型
parser.add_argument('--run_test', default=False, type=bool)
# simulate.py
parser.add_argument('--simulate', default=True, type=bool)
parser.add_argument('--rise', default=1.2, type=float)
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
class economy_class:
    def __init__(self, args):
        self.args = args
        self.path = os.getcwd()
        self.path_economy = f'{self.path}/economy'
        self.path_tushare = f'{self.path}/economy/tushare'

    def run(self):
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
        if self.args.data_add:
            self._data_add()
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

    def _industry_choice(self):
        print('economy/tushare/industry_choice.py')
        os.system(f'python industry_choice.py --industry {self.args.industry}')

    def _data_get(self):
        print('economy/tushare/data_get.py')
        os.system(f'python data_get.py --token {self.args.token} --end_time {self.args.end_time}')

    def _data_screen(self):
        print('economy/data_screen.py')
        os.system(f'python data_screen.py --threshold {self.args.threshold}')

    def _data_add(self):
        print('economy/data_add.py')
        os.system(f'python data_add.py')

    def _run_base_test(self, data_dir='economy/dataset', model_dir='economy/model_test'):
        print('run.py | 训练基础测试模型')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        with open('economy/data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        for industry in screen_dict:
            name_list = screen_dict[industry].keys()
            for name in name_list:
                data_path = f'{data_dir}/{name}_add.csv'
                os.system(f'python run.py --data_path {data_path} --input_column input_column.txt'
                          f' --output_column output_column.txt --input_size 96 --output_size 24 --divide 19,1'
                          f' --weight {model_dir}/base_test.pt --weight_again True'
                          f' --model itransformer --model_type l --epoch 10 --lr_end_epoch 10')
                shutil.move('last.pt', f'{model_dir}/base_test.pt')

    def _run_test(self, data_dir='economy/dataset', model_dir='economy/model_test'):
        print('run.py | 训练测试模型')
        assert os.path.exists('economy/model_test/base_test.pt')
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
                if os.path.exists(model_path):  # 已有模型则不再训练
                    continue
                os.system(f'python run.py --data_path {data_path} --input_column input_column.txt'
                          f' --output_column output_column.txt --input_size 96 --output_size 24 --divide 19,1'
                          f' --weight economy/model_test/base_test.pt --weight_again True'
                          f' --model itransformer --model_type l --epoch 30 --lr_end_epoch 30')
                shutil.move('last.pt', model_path)
                # 打开日志
                with open('log.txt', 'r', encoding='utf-8') as f:
                    log = f.readlines()
                train_loss = float(log[2].strip()[11:])
                val_loss = float(log[3].strip()[9:])
                # 记录模型信息
                model_dict[industry][name] = [train_loss, val_loss, None, None]
                with open('economy/model.yaml', 'w', encoding='utf-8') as f:
                    yaml.dump(model_dict, f, allow_unicode=True, sort_keys=False)

    def _simulate(self):
        print('simulate.py')
        with open('data_screen.yaml', 'r', encoding='utf-8') as f:  # 股票选择
            screen_dict = yaml.load(f, Loader=yaml.SafeLoader)
        with open('model.yaml', 'r', encoding='utf-8') as f:  # 模型信息
            model_dict = yaml.load(f, Loader=yaml.SafeLoader)
        for industry in screen_dict:
            name_list = screen_dict[industry].keys()
            for name in name_list:
                os.system(f'python simulate.py --model_path model_test/{name}.pt --data_path dataset/{name}_add.csv'
                          f' --rise {self.args.rise}')
                # 打开日志
                with open('log.txt', 'r', encoding='utf-8') as f:
                    log = f.readlines()
                income_sum = float(log[0].strip()[5:])
                income_mean = float(log[1].strip()[8:])
                # 记录模型信息
                model_dict[industry][name][2:4] = [income_sum, income_mean]
                with open('model.yaml', 'w', encoding='utf-8') as f:
                    yaml.dump(model_dict, f, allow_unicode=True, sort_keys=False)


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = economy_class(args)
    model.run()
