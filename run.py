import wandb
import torch
import argparse
from train_class import train_class

# -------------------------------------------------------------------------------------------------------------------- #
# 分布式数据并行训练:
# python -m torch.distributed.launch --master_port 9999 --nproc_per_node n run.py --distributed True
# master_port为GPU之间的通讯端口，空闲的即可。n为GPU数量
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|时间序列预测|')
parser.add_argument('--log', default=True, type=bool, help='|日志|')
parser.add_argument('--print_info', default=True, type=bool, help='|打印信息|')
parser.add_argument('--wandb', default=False, type=bool, help='|wandb可视化|')
parser.add_argument('--data_path', default='dataset/ETTh1.csv', type=str, help='|数据位置|')
parser.add_argument('--input_len', default=96, type=int, help='|输入时间长度|')
parser.add_argument('--output_len', default=24, type=int, help='|输出时间长度|')
parser.add_argument('--input_column', default='HUFL,HULL,MUFL,MULL,LUFL,LULL,OT', type=str, help='|输入变量，可为.txt|')
parser.add_argument('--output_column', default='HUFL,HULL,MUFL,MULL,LUFL,LULL,OT', type=str, help='|输出变量，可为.txt|')
parser.add_argument('--weight_path', default='last.pt', type=str, help='|加载模型，没有模型会创建新模型|')
parser.add_argument('--weight_again', default=True, type=bool, help='|重置学习率等状态，在weight_path上重新训练|')
parser.add_argument('--save_epoch', default=10, type=int, help='|每x轮和最后一轮保存模型|')
parser.add_argument('--save_path', default='last.pt', type=str, help='|保存模型|')
parser.add_argument('--save_best', default='best.pt', type=str, help='|保存最佳模型|')
parser.add_argument('--epoch', default=100, type=int, help='|总轮数(包含加载模型已训练轮数)|')
parser.add_argument('--batch', default=64, type=int, help='|批量大小，分布式时为总批量|')
parser.add_argument('--divide', default=[19, 1], type=list, help='|训练集和验证集划分比例|')
parser.add_argument('--divide_train', default=0, type=int, help='|训练集数据:0训练集，1所有数据，2验证集|')
parser.add_argument('--z_score', default=1, type=int, help='|归一化时:0训练集，1所有数据，2验证集|')
parser.add_argument('--model', default='linear', type=str, help='|模型选择|')
parser.add_argument('--model_type', default='m', type=str, help='|模型型号|')
parser.add_argument('--loss', default='mse_decay', type=str, help='|损失函数|')
parser.add_argument('--warmup_ratio', default=0.01, type=float, help='|预热训练步数占总步数比例，最少5步，基准为0.01|')
parser.add_argument('--lr_start', default=1e-4, type=float, help='|初始学习率，adam算法，批量大时要增大，基准为1e-4|')
parser.add_argument('--lr_end_ratio', default=0.01, type=float, help='|最终学习率=lr_end_ratio*lr_start，基准为0.01|')
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
# 设置
torch.manual_seed(999)  # 为CPU设置随机种子
torch.cuda.manual_seed_all(999)  # 为所有GPU设置随机种子
torch.backends.cudnn.deterministic = True  # 固定每次返回的卷积算法
torch.backends.cudnn.enabled = True  # cuDNN使用非确定性算法
torch.backends.cudnn.benchmark = False  # 训练前cuDNN会先搜寻每个卷积层最适合实现它的卷积算法，加速运行；但对于复杂变化的输入数据，可能会有过长的搜寻时间，对于训练比较快的网络建议设为False
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    train = train_class(args)
    train.train()
