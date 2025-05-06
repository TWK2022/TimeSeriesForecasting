import os
import yaml
import tqdm
import wandb
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
from model.layer import deploy
from train_class import train_class

# -------------------------------------------------------------------------------------------------------------------- #
# 股票预测
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|股票预测|')
parser.add_argument('--train_again', default=True, type=bool, help='|继续训练新数据|')
parser.add_argument('--data_screen', default='dataset/data_screen.yaml', type=str, help='|筛选数据|')
parser.add_argument('--data_dir', default='dataset/stock_add', type=str, help='|股票数据|')
parser.add_argument('--save_dir', default='predict/weight', type=str, help='|保存模型|')
parser.add_argument('--simulate_save_path', default='dataset/simulate.yaml', type=str, help='|保存结果|')
parser.add_argument('--predict_save_path', default='dataset/predict.yaml', type=str, help='|保存结果|')
parser.add_argument('--log', default=False, type=bool, help='|日志|')
parser.add_argument('--print_info', default=False, type=bool, help='|打印信息|')
parser.add_argument('--wandb', default=False, type=bool, help='|wandb可视化|')
parser.add_argument('--data_path', default='', type=str, help='|数据位置|')
parser.add_argument('--input_len', default=8, type=int, help='|输入时间长度|')
parser.add_argument('--output_len', default=2, type=int, help='|输出时间长度|')
parser.add_argument('--input_column', default='predict/input_column.txt', type=str, help='|输入变量，可传入txt|')
parser.add_argument('--output_column', default='predict/output_column.txt', type=str, help='|预测变量，可传入txt|')
parser.add_argument('--weight_path', default='', type=str, help='|加载模型，没有模型会创建新模型|')
parser.add_argument('--weight_again', default=True, type=bool, help='|重置学习率等状态，在weight_path上重新训练|')
parser.add_argument('--save_epoch', default=10, type=int, help='|每x轮和最后一轮保存模型|')
parser.add_argument('--save_path', default='predict/last.pt', type=str, help='|保存模型|')
parser.add_argument('--save_best', default='predict/best.pt', type=str, help='|保存最佳模型|')
parser.add_argument('--epoch', default=100, type=int, help='|总轮数(包含加载模型已训练轮数)|')
parser.add_argument('--batch', default=64, type=int, help='|批量大小，分布式时为总批量|')
parser.add_argument('--divide', default=[19, 1], type=list, help='|训练集和验证集划分比例|')
parser.add_argument('--divide_train', default=1, type=int, help='|训练集数据:0训练集，1所有数据，2验证集|')
parser.add_argument('--z_score', default=1, type=int, help='|归一化时:0训练集，1所有数据，2验证集|')
parser.add_argument('--model', default='linear', type=str, help='|模型选择|')
parser.add_argument('--model_type', default='l', type=str, help='|模型型号|')
parser.add_argument('--loss', default='mse_decay', type=str, help='|损失函数|')
parser.add_argument('--warmup_ratio', default=0.01, type=float, help='|预热训练步数占总步数比例，最少5步，基准为0.01|')
parser.add_argument('--lr_start', default=0.0001, type=float, help='|初始学习率，adam算法，批量大时要增大，基准为0.0001|')
parser.add_argument('--lr_end_ratio', default=0.01, type=float, help='|最终学习率=lr_end_ratio*lr_start，基准为0.01|')
parser.add_argument('--lr_end_epoch', default=100, type=int, help='|最终学习率的轮数，每1步都调整，余玄下降法|')
parser.add_argument('--regularization', default='L2', type=str, help='|正则化，有L2、None|')
parser.add_argument('--r_value', default=0.0005, type=float, help='|正则化权重，可从0.0001开始增加直到最佳值|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--latch', default=True, type=bool, help='|模型和数据是否为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU处理数据进程数，0为一个主进程，一般为0、2、4、8|')
parser.add_argument('--ema', default=True, type=bool, help='|平均指数移动(EMA)调整参数|')
parser.add_argument('--amp', default=True, type=bool, help='|混合float16精度训练，CPU时不可用|')
parser.add_argument('--distributed', default=False, type=bool, help='|单机多卡分布式训练，batch为总batch|')
parser.add_argument('--local_rank', default=0, type=int, help='|分布式训练使用命令后会自动传入的参数|')
args = parser.parse_args()
project_dir = os.path.dirname(os.path.dirname(__file__))
args.data_screen = project_dir + '/' + args.data_screen
args.simulate_save_path = project_dir + '/' + args.simulate_save_path
args.predict_save_path = project_dir + '/' + args.predict_save_path
args.data_dir = project_dir + '/' + args.data_dir
args.save_dir = project_dir + '/' + args.save_dir
args.weight_path = project_dir + '/' + args.weight_path
args.save_path = project_dir + '/' + args.save_path
args.save_best = project_dir + '/' + args.save_best
args.input_column = project_dir + '/' + args.input_column
args.output_column = project_dir + '/' + args.output_column
args.input_column = train_class.read_column(args.input_column)  # column处理
args.output_column = train_class.read_column(args.output_column)  # column处理
args.device = args.device if torch.cuda.is_available() else 'cpu'  # 没有GPU时使用CPU
args.device_number = max(torch.cuda.device_count(), 1)  # 使用的GPU数，可能为CPU
# 混合float16精度训练
args.amp = torch.cuda.amp.GradScaler() if args.amp and args.device != 'cpu' else False
# 分布式训练
if args.distributed:
    torch.distributed.init_process_group(backend='nccl')
    args.device = torch.device('cuda', args.local_rank)
# wandb可视化: https://wandb.ai
if args.wandb and args.local_rank == 0:  # 分布式时只记录一次wandb
    args.wandb_run = wandb.init(project='TimeSeriesForecasting', name='train', config=args)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
# 设置
torch.manual_seed(999)  # 为CPU设置随机种子
torch.cuda.manual_seed_all(999)  # 为所有GPU设置随机种子
torch.backends.cudnn.deterministic = True  # 固定每次返回的卷积算法
torch.backends.cudnn.enabled = True  # cuDNN使用非确定性算法
torch.backends.cudnn.benchmark = False  # 训练前cuDNN会先搜寻每个卷积层最适合实现它的卷积算法，加速运行；但对于复杂变化的输入数据，可能会有过长的搜寻时间，对于训练比较快的网络建议设为False


# -------------------------------------------------------------------------------------------------------------------- #
class predict_class:
    def __init__(self, args=args):
        self.args = args
        self.epoch_default = args.epoch
        self.weight_path_default = args.weight_path
        self.lr_start_default = args.lr_start
        self.lr_end_epoch_default = args.lr_end_epoch
        self.simulate_dict = {}  # 记录验证结果
        with open(args.data_screen, 'r', encoding='utf-8') as f:
            self.data_screen = yaml.load(f, Loader=yaml.SafeLoader)
        self.stock_all = {}
        for industry in self.data_screen:
            for name in self.data_screen[industry].keys():
                self.stock_all[name] = self.data_screen[industry][name]

    def predict(self):
        self.train()
        self.simulate()
        self.feature()

    def train(self):
        args = self.args
        for name in tqdm.tqdm(self.stock_all.keys()):
            weight_path = f'{args.save_dir}/{name}.pt'
            args.data_path = f'{args.data_dir}/{name}_add.csv'
            if os.path.exists(weight_path):  # 已存在
                if not args.train_again:
                    continue
                # 继续训练
                args.weight_path = weight_path
                args.epoch = 30
                args.lr_start = self.lr_start_default * 0.2
                args.lr_end_epoch = 30
            else:  # 正常训练
                args.weight_path = weight_path
                args.epoch = self.epoch_default
                args.lr_start = self.lr_start_default
                args.lr_end_epoch = self.lr_end_epoch_default
            train = train_class(args)
            train.train()
            shutil.move(args.save_path, weight_path)  # 移动文件

    def simulate(self):
        args = self.args
        result_dict = {}
        total_all = 0  # 总次数
        correct_all = 0  # 正确次数
        for industry in tqdm.tqdm(self.data_screen.keys()):
            result_dict[industry] = {}
            self.simulate_dict[industry] = {}
            for name in self.data_screen[industry].keys():
                weight_path = f'{args.save_dir}/{name}.pt'
                data_path = f'{args.data_dir}/{name}_add.csv'
                # 模型
                model_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
                model_dict['model'] = deploy(model_dict['model'], model_dict['mean_input'], model_dict['mean_output'],
                                             model_dict['std_input'], model_dict['std_output']).eval().to(args.device)
                # 数据
                df = pd.read_csv(data_path, encoding='utf-8', index_col=0)
                input_data = df[args.input_column].values.astype(np.float32)
                close = df['收盘价'].values.astype(np.float32)
                value = df['均价'].values.astype(np.float32)
                # 预测
                total = 0  # 总次数
                correct = 0  # 正确次数
                with torch.no_grad():
                    for index in range(0, len(input_data) - args.input_len - args.output_len + 1):
                        input_ = input_data[index:index + args.input_len]
                        output_ = value[index + args.input_len:index + args.input_len + args.output_len]
                        close_ = close[index + args.input_len - 1]  # 前1天收盘价
                        tensor = torch.tensor(input_, dtype=torch.float32).unsqueeze(0).to(self.args.device)
                        pred_value = model_dict['model'](tensor)[0][0].cpu().numpy()
                        # 回测策略: 预测上涨时，计算第1-2天任意1天上涨的概率
                        increase = np.mean(pred_value[0:2] / close_)  # 涨幅
                        if increase > 1:
                            total += 1
                            total_all += 1
                            if output_[0] > close_ or output_[1] > close_:
                                correct += 1
                                correct_all += 1
                # 记录
                result_dict[industry][name] = f'{correct / (total + 1e-6):.2f} | 样本{total}'
                self.simulate_dict[industry][name] = round(correct / (total + 1e-6), 2)
        result_dict['总计'] = f'{correct_all / (total_all + 1e-6):.2f} | 样本{total_all}'
        # 保存
        with open(args.simulate_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(result_dict, f, allow_unicode=True, sort_keys=False)
        print(f'| 保存结果: {args.simulate_save_path} |')

    def feature(self):
        args = self.args
        result_dict = {}
        for industry in tqdm.tqdm(self.data_screen.keys()):
            result_dict[industry] = {}
            for name in self.data_screen[industry].keys():
                if self.simulate_dict[industry][name] < 0.60:  # 正确率太低舍弃
                    continue
                weight_path = f'{args.save_dir}/{name}.pt'
                data_path = f'{args.data_dir}/{name}_add.csv'
                # 模型
                model_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
                model_dict['model'] = deploy(model_dict['model'], model_dict['mean_input'], model_dict['mean_output'],
                                             model_dict['std_input'], model_dict['std_output']).eval().to(args.device)
                # 数据
                df = pd.read_csv(data_path, encoding='utf-8', index_col=0)
                input_data = df[args.input_column].values.astype(np.float32)
                input_data = input_data[-args.input_len:]
                tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(args.device)
                # 预测
                with torch.no_grad():
                    pred_value = model_dict['model'](tensor)[0][0].cpu().numpy()
                pred_value = pred_value / df['均价'].values[-1]
                increase = round(float(np.mean(pred_value[0:2])), 2)
                if increase > 1:
                    result_dict[industry][name] = increase
        # 保存
        with open(args.predict_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(result_dict, f, allow_unicode=True, sort_keys=False)
        print(f'| 保存结果: {args.predict_save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = predict_class()
    model.predict()
