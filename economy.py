import os
import yaml
import argparse

# -------------------------------------------------------------------------------------------------------------------- #
# 需要先训练一个预训练模型:economy/model/base.pt
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|集成|')
# economy/tushare/data_choice.py
parser.add_argument('--data_choice', default=False, type=bool)
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
# run.py | 训练测试模型
parser.add_argument('--run', default=False, type=bool)
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
def economy(args):
    path = os.getcwd()
    path_economy = f'{path}/economy'
    path_tushare = f'{path}/economy/tushare'
    # economy/tushare目录
    os.chdir(path_tushare)
    if args.data_choice:
        print('economy/tushare/data_choice.py')
        os.system(f'python data_choice.py --industry {args.industry}')
    if args.data_get:
        print('economy/tushare/data_get.py')
        os.system(f'python data_get.py --token {args.token} --end_time {args.end_time}')
    # economy目录
    os.chdir(path_economy)
    if args.data_screen:
        print('economy/data_screen.py')
        os.system(f'python data_screen.py --threshold {args.threshold}')
    if args.data_add:
        print('economy/data_add.py')
        os.system(f'python data_add.py')
    # 原目录
    os.chdir(path)
    if args.run:
        print('run.py | 训练测试模型')
        screen_path = 'economy/data_screen.txt'
        with open(screen_path, 'r', encoding='utf-8') as f:  # 模型位置
            model_list = [f'economy/model/{_.split()[0]}_test.pt' for _ in f.readlines()]
        with open(screen_path, 'r', encoding='utf-8') as f:  # 数据位置
            data_list = [f'economy/dataset/{_.split()[0]}_add.csv' for _ in f.readlines()]
        with open('economy/model.yaml', 'r', encoding='utf-8') as f:  # 模型信息
            yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
        for model_path, data_path in zip(model_list, data_list):
            model_name = model_path.split('/')[-1]
            if os.path.exists(model_path):  # 已有模型则不再训练
                continue
            os.system(f'python run.py --data_path {data_path} --input_column input_column.txt '
                      f'--output_column output_column.txt --divide 19,1 --weight economy/model/base.pt'
                      f' --weight_again True --save_path economy/model/best.pt --model itransformer --model_type l '
                      f'--epoch 30 --lr_end_epoch 30')
            os.rename('economy/model/base.pt', f'economy/model/{model_name}')
            # 打开日志
            with open('log.txt', 'r', encoding='utf-8') as f:
                log = f.readlines()
            train_loss = float(log[2].strip()[11:])
            val_loss = float(log[3].strip()[9:])
            # 记录模型信息
            yaml_dict[model_name] = [train_loss, val_loss]
            with open('economy/model.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(yaml_dict, f, allow_unicode=True)


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    economy(args)
