import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.layer import deploy

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|pt模型推理|')
parser.add_argument('--model_path', default='best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--data_path', default=r'./dataset/ETTh1.csv', type=str, help='|数据路径|')
parser.add_argument('--input_column', default='HUFL,HULL,MUFL,MULL,LUFL,LULL,OT', type=str, help='|选择输入的变量|')
parser.add_argument('--output_column', default='HUFL,HULL,MUFL,MULL,LUFL,LULL,OT', type=str, help='|选择预测的变量|')
parser.add_argument('--input_size', default=128, type=int, help='|输入的长度|')
parser.add_argument('--output_size', default=64, type=int, help='|输出的长度|')
parser.add_argument('--batch', default=64, type=int, help='|训练批量大小|')
parser.add_argument('--device', default='cuda', type=str, help='|用CPU/GPU推理|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU在处理数据时使用的进程数，0表示只有一个主进程，一般为0、2、4、8|')
parser.add_argument('--plot_len', default=500, type=int, help='|画图长度|')
args = parser.parse_args()
args.input_column = args.input_column.split(',')
args.output_column = args.output_column.split(',')
args.save_path = 'save_image'
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'没有找到模型{args.model_path}'
assert os.path.exists(args.data_path), f'没有找到预测数据{args.data_path}'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def draw(pred_middle, pred_last, true, middle, last):  # pred为预测值，true为真实值，pred和true长度不相等
    # 画图
    middle_plot = np.zeros(true.shape)
    last_plot = np.zeros(true.shape)
    middle_plot[:, args.input_size + middle - 1:-middle] = pred_middle
    last_plot[:, args.input_size + last - 1:] = pred_last
    for i in range(len(args.output_column)):
        name = f'{args.output_column[i]}_{args.plot_len}'
        plt.title(name)
        plt.plot(true[i, :], color='green', label=f'{args.output_column[i]}_true')
        plt.plot(middle_plot[i, :], color='orange', label=f'{args.output_column[i]}_{middle}')
        plt.plot(last_plot[i, :], color='red', label=f'{args.output_column[i]}_{last}')
        plt.legend()
        plt.savefig(args.save_path + '/' + name + '.jpg')
        plt.close()


def test_pt():
    # 加载模型
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model']
    model = deploy(model, model_dict['mean_input'], model_dict['mean_output'], model_dict['std_input'],
                   model_dict['std_output'])
    model.float().eval().to(args.device)
    print('| 模型加载成功:{} |'.format(args.model_path))
    # 加载数据
    df = pd.read_csv(args.data_path)
    input_data = np.array(df[args.input_column].astype(np.float32)).transpose(1, 0)
    input_data = input_data[:, - args.plot_len:]  # 限定长度方便画图
    output_data = np.array(df[args.output_column].astype(np.float32)).transpose(1, 0)
    output_data = output_data[:, - args.plot_len:]  # 限定长度方便画图
    # 推理
    start_time = time.time()
    middle = args.output_size // 2
    last = args.output_size
    pred_middle = []
    pred_last = []
    dataloader = torch.utils.data.DataLoader(torch_dataset(args, input_data), batch_size=args.batch,
                                             shuffle=False, drop_last=False, pin_memory=False,
                                             num_workers=args.num_worker)
    with torch.no_grad():
        for item, batch in enumerate(dataloader):
            batch = batch.to(args.device)
            pred_batch = model(batch).detach().cpu().numpy()
            pred_middle.append(pred_batch[:, :, middle - 1])
            pred_last.append(pred_batch[:, :, last - 1])
        pred_middle = np.concatenate(pred_middle, axis=0).transpose(1, 0)
        pred_last = np.concatenate(pred_last, axis=0).transpose(1, 0)
    end_time = time.time()
    print('| 数据:{} 批量:{} 平均耗时:{:.4f} |'
          .format(args.data_path, args.batch, (end_time - start_time) / len(pred_middle)))
    # 画图
    draw(pred_middle, pred_last, output_data, middle, last)
    print(f'| 画图保存位置:{args.save_path} |')


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, input_data):
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.input_data = input_data

    def __len__(self):
        return self.input_data.shape[1] - self.input_size - self.output_size + 1

    def __getitem__(self, index):
        boundary = index + self.input_size
        series = self.input_data[:, index:boundary]  # 输入数据
        series = torch.tensor(series, dtype=torch.float32)  # 转换为tensor
        return series


if __name__ == '__main__':
    test_pt()
