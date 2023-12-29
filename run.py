import os
import wandb
import torch
import argparse
from block.util import read_column
from block.data_get import data_get
from block.loss_get import loss_get
from block.model_get import model_get
from block.train_get import train_get

# -------------------------------------------------------------------------------------------------------------------- #
# 分布式训练：
# python -m torch.distributed.launch --master_port 9999 --nproc_per_node n run.py --distributed True
# master_port为GPU之间的通讯端口，空闲的即可
# n为GPU数量
# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|时间序列预测|')
parser.add_argument('--data_path', default=r'./dataset/sin_cos.csv', type=str, help='|数据路径|')
parser.add_argument('--divide', default='9,1', type=str, help='|训练集和验证集划分比例|')
parser.add_argument('--weight', default='last.pt', type=str, help='|已有模型的位置，如果没找到模型则会创建新模型|')
parser.add_argument('--save_path', default='best.pt', type=str, help='|保存模型的位置|')
parser.add_argument('--wandb', default=False, type=bool, help='|是否使用wandb可视化|')
parser.add_argument('--wandb_project', default='TimeSeriesForecasting', type=str, help='|wandb项目名称|')
parser.add_argument('--wandb_name', default='train', type=str, help='|wandb项目中的训练名称|')
parser.add_argument('--model', default='linear_conv', type=str, help='|自定义模型选择|')
parser.add_argument('--model_type', default='m', type=str, help='|模型型号，部分模型有|')
parser.add_argument('--input_column', default='sin,cos', type=str, help='|选择输入的变量，可传入.txt|')
parser.add_argument('--output_column', default='mix', type=str, help='|选择预测的变量，可传入.txt|')
parser.add_argument('--input_size', default=512, type=int, help='|输入的长度|')
parser.add_argument('--output_size', default=256, type=int, help='|输出的长度|')
parser.add_argument('--z_score_cycle', default=-1, type=int, help='|以训练集的周期长度计算mean和std，-1时用所有训练集|')
parser.add_argument('--epoch', default=200, type=int, help='|训练轮数|')
parser.add_argument('--batch', default=64, type=int, help='|训练批量大小|')
parser.add_argument('--loss', default='mse_decay', type=str, help='|损失函数|')
parser.add_argument('--lr_start', default=0.001, type=float, help='|初始学习率，adam算法，3轮预热训练，基准为0.001|')
parser.add_argument('--lr_end_ratio', default=0.2, type=float, help='|最终学习率=lr_end_ratio*lr_start，基准为0.2|')
parser.add_argument('--lr_adjust_num', default=100, type=int, help='|学习率下降调整次数，余玄下降法，要小于总轮次|')
parser.add_argument('--lr_adjust_threshold', default=0.97, type=float, help='|本轮损失下降一定比例时才调整，基准为0.97|')
parser.add_argument('--regularization', default='L2', type=str, help='|正则化，有L2、None|')
parser.add_argument('--r_value', default=0.0005, type=float, help='|正则化的权重系数|')
parser.add_argument('--device', default='cuda', type=str, help='|训练设备|')
parser.add_argument('--latch', default=True, type=bool, help='|模型和数据是否为锁存，True为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU处理数据的进程数，0表示只有一个主进程，一般为0、2、4、8|')
parser.add_argument('--ema', default=True, type=bool, help='|使用平均指数移动(EMA)调整参数|')
parser.add_argument('--amp', default=True, type=bool, help='|混合float16精度训练，CPU时不可用|')
parser.add_argument('--distributed', default=False, type=bool, help='|单机多卡分布式训练，分布式训练时batch为总batch|')
parser.add_argument('--local_rank', default=0, type=int, help='|分布式训练使用命令后会自动传入的参数|')
args = parser.parse_args()
args.divide = list(map(int, args.divide.split(',')))
args.input_column = read_column(args.input_column)  # column处理
args.output_column = read_column(args.output_column)  # column处理
args.device_number = max(torch.cuda.device_count(), 1)  # 使用的GPU数，可能为CPU
print(f'| args:{args} |')
# 为CPU设置随机种子
torch.manual_seed(999)
# 为所有GPU设置随机种子
torch.cuda.manual_seed_all(999)
# 固定每次返回的卷积算法
torch.backends.cudnn.deterministic = True
# cuDNN使用非确定性算法
torch.backends.cudnn.enabled = True
# 训练前cuDNN会先搜寻每个卷积层最适合实现它的卷积算法，加速运行；但对于复杂变化的输入数据，可能会有过长的搜寻时间，对于训练比较快的网络建议设为False
torch.backends.cudnn.benchmark = False
# wandb可视化:https://wandb.ai
if args.wandb and args.local_rank == 0:  # 分布式时只记录一次wandb
    args.wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
# 混合float16精度训练
if args.amp:
    args.amp = torch.cuda.amp.GradScaler()
# 分布式训练
if args.distributed:
    torch.distributed.init_process_group(backend="nccl")
    args.device = torch.device("cuda", args.local_rank)
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
if args.local_rank == 0:
    assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'
    if os.path.exists(args.weight):
        print(f'| 加载已有模型:{args.weight} |')
    else:
        assert os.path.exists(f'model/{args.model}.py'), f'! 没有自定义模型:{args.model} !'
        print(f'| 创建自定义模型:{args.model} | 型号:{args.model_type} |')
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    # 数据
    data_dict = data_get(args)
    # 模型
    model_dict = model_get(args)
    # 损失
    loss = loss_get(args)
    # 摘要
    print('| 训练集:{} | 验证集:{} | 批量{} | 模型:{} | 输入长度:{} | 输出长度:{} | 损失函数:{} | 初始学习率:{} |'
          .format(len(data_dict['train_input']), len(data_dict['val_input']), args.batch, args.model, args.input_size,
                  args.output_size, args.loss, args.lr_start)) if args.local_rank == 0 else None
    # 训练
    train_get(args, data_dict, model_dict, loss)
