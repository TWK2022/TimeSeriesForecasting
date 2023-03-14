import os
import wandb
import torch
import argparse
from block.data_get import data_get
from block.model_get import model_get
from block.loss_get import loss_get
from block.train_get import train_get

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='时间序列预测')
parser.add_argument('--data_path', default=r'./dataset/ETTh.csv', type=str, help='|数据路径|')
parser.add_argument('--divide', default='9,1', type=str, help='|训练集和验证集划分比例|')
parser.add_argument('--weight', default='best.pt', type=str, help='|已有模型的位置，如果没找到模型则会创建新模型|')
parser.add_argument('--save_name', default='best.pt', type=str, help='|保存模型的位置|')
parser.add_argument('--wandb', default=False, type=bool, help='|是否使用wandb可视化|')
parser.add_argument('--wandb_project', default='test', type=str, help='|wandb项目名称|')
parser.add_argument('--wandb_name', default='train', type=str, help='|wandb项目中的训练名称|')
parser.add_argument('--timm', default=False, type=bool, help='|是否使用timm模型|')
parser.add_argument('--model', default='tsf', type=str, help='|模型选择，timm为True时为timm中的模型|')
parser.add_argument('--model_type', default='m', type=str, help='|模型型号参数，部分模型有|')
parser.add_argument('--input_column', default='1,2,3', type=str, help='|选择输入的变量|')
parser.add_argument('--output_column', default='1,2', type=str, help='|选择预测的变量|')
parser.add_argument('--input_size', default=128, type=int, help='|输入的长度|')
parser.add_argument('--output_size', default=32, type=int, help='|输出的长度|')
parser.add_argument('--epoch', default=50, type=int, help='|训练轮数|')
parser.add_argument('--batch', default=64, type=int, help='|训练批量大小|')
parser.add_argument('--loss', default='mse', type=str, help='|损失函数|')
parser.add_argument('--lr', default=0.002, type=int, help='|初始学习率，训练中采用adam算法|')
parser.add_argument('--device', default='cuda', type=str, help='|训练设备|')
parser.add_argument('--latch', default=True, type=bool, help='|模型和数据是否为锁存，True为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU在处理数据时使用的进程数，0表示只有一个主进程，一般为0、2、4、8|')
args = parser.parse_args()
args.divide = list(map(int, args.divide.split(',')))
args.input_column = args.input_column.split(',')
args.output_column = args.output_column.split(',')
print('| args:{} |'.format(args))
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
if args.wandb:
    args.wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.data_path), 'data_path不存在'
if os.path.exists(args.weight):
    print('| 加载已有模型:{} |'.format(args.weight))
elif args.timm:
    import timm

    assert timm.list_models(args.model) != [], 'timm中没有此模型{}'.format(args.model)
    print('| 使用timm创建模型:{} |'.format(args.model))
else:
    assert os.path.exists('model/' + args.model + '.py'), '没有此自定义模型'.format(args.model)
    print('| 创建自定义模型:{} | 型号:{} |'.format(args.model, args.model_type))
if args.device.lower() in ['cuda', 'gpu']:
    assert torch.cuda.is_available(), 'GPU不可用'
    args.device = 'cuda'
else:
    args.device = 'cpu'
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    # 数据
    data_dict = data_get(args)
    # 模型
    model_dict = model_get(args, data_dict)
    # 损失
    loss = loss_get(args)
    # 摘要
    print('| 训练集:{} | 验证集:{} | 模型:{} | 损失函数:{} | 初始学习率:{} |'
          .format(len(data_dict['train_input']), len(data_dict['val_input']), args.model, args.loss, args.lr))
    # 训练(包括图片读取和预处理、训练、验证、保存模型)
    model_dict = train_get(args, data_dict, model_dict, loss)
    # 显示结果
    try:
        print('\n| 最佳结果 | train_loss:{:.4f} val_loss:{:.4f} | val_mae:{:.4f} | val_mse:{:.4f} |\n'
              .format(model_dict['train_loss'], model_dict['val_loss'], model_dict['val_mae'], model_dict['val_mse']))
    except:
        print('\n| !由于指标太低没有保存任何结果! |\n')
