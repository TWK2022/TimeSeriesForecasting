import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.layer import deploy
from train_class import train_class

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|pt模型推理|')
parser.add_argument('--model_path', default='best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--data_path', default=r'./dataset/sin_cos.csv', type=str, help='|数据位置|')
parser.add_argument('--input_column', default='sin,cos', type=str, help='|选择输入的变量，可传入.txt|')
parser.add_argument('--output_column', default='mix', type=str, help='|选择预测的变量，可传入.txt|')
parser.add_argument('--input_size', default=96, type=int, help='|输入长度|')
parser.add_argument('--output_size', default=24, type=int, help='|输出长度|')
parser.add_argument('--batch', default=64, type=int, help='|批量大小|')
parser.add_argument('--device', default='cuda', type=str, help='|推理设备|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU处理数据的进程数，0只有一个主进程，一般为0、2、4、8|')
parser.add_argument('--plot_len', default=1000, type=int, help='|画图长度，取数据的倒数plot_len个|')
args = parser.parse_args()
args.input_column = train_class.read_column(args.input_column)  # column处理
args.output_column = train_class.read_column(args.output_column)  # column处理
args.save_path = 'save_image'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# -------------------------------------------------------------------------------------------------------------------- #
def draw(pred_middle, pred_last, true, middle, last):  # pred为预测值，true为真实值，pred和true长度不相等
    # 画图(所有预测中值和预测末值)
    middle_plot = np.zeros(true.shape)
    last_plot = np.zeros(true.shape)
    middle_plot[:, args.input_size + middle - 1:-middle] = pred_middle
    last_plot[:, args.input_size + last - 1:] = pred_last
    input_cut = max(args.input_size, 0)  # 防止输入序列太长时画图不好看
    true = true[:, input_cut:]
    middle_plot = middle_plot[:, input_cut:]
    last_plot = last_plot[:, input_cut:]
    for i in range(len(args.output_column)):
        name = f'{args.output_column[i]}_last{args.plot_len}'
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 使用字体时让坐标轴正常显示负号
        plt.title(name)
        plt.plot(true[i, :], color='green', label=f'{args.output_column[i]}_true')
        plt.plot(middle_plot[i, :], color='orange', label=f'{args.output_column[i]}_{middle}')
        plt.plot(last_plot[i, :], color='red', label=f'{args.output_column[i]}_{last}')
        plt.grid()
        plt.legend()
        plt.savefig(args.save_path + '/' + name + '.jpg')
        plt.close()


def draw_predict(last_data, last_output):
    # 画图(对最后一组数据预测)
    pred = np.zeros(last_data.shape)
    pred[:, -args.output_size:] = last_output
    true = last_data[:, -args.output_size - args.input_size:]
    pred = pred[:, -args.output_size - args.input_size:]
    input_cut = max(args.input_size - 50, 0)  # 防止输入序列太长时画图不好看
    true = true[:, input_cut:]
    pred = pred[:, input_cut:]
    for i in range(len(args.output_column)):
        name = f'{args.output_column[i]}_last_predict'
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 使用字体时让坐标轴正常显示负号
        plt.title(name)
        plt.plot(true[i, :], color='green', label=f'{args.output_column[i]}_true')
        plt.plot(pred[i, :], color='cyan', label=f'{args.output_column[i]}_pred')
        plt.grid()
        plt.legend()
        plt.savefig(args.save_path + '/' + name + '.jpg')
        plt.close()


def predict_pt():
    # 加载模型
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model']
    model = deploy(model, model_dict['mean_input'], model_dict['mean_output'], model_dict['std_input'],
                   model_dict['std_output']).eval().to(args.device)
    model = model.half() if args.device == 'cuda' else model.float()
    epoch = model_dict['epoch_finished']
    rmse = round(model_dict['val_rmse'], 4)
    print(f'| 模型加载成功:{args.model_path} | epoch:{epoch} | rmse:{rmse}|')
    # 加载数据
    try:
        df = pd.read_csv(args.data_path, encoding='utf-8', index_col=0)
    except:
        df = pd.read_csv(args.data_path, encoding='gbk', index_col=0)
    input_data = np.array(df[args.input_column]).astype(np.float32).T
    input_data = input_data[:, -args.plot_len - args.input_size:]  # 限定长度方便画图
    output_data = np.array(df[args.output_column]).astype(np.float32).T
    last_data = output_data[:, -args.input_size - args.output_size:]
    output_data = output_data[:, -args.plot_len - args.input_size:]  # 限定长度方便画图
    # 推理
    middle = args.output_size // 2
    last = args.output_size
    pred_middle = []
    pred_last = []
    dataloader = torch.utils.data.DataLoader(torch_dataset(args, input_data), batch_size=args.batch,
                                             shuffle=False, drop_last=False, pin_memory=False,
                                             num_workers=args.num_worker)
    start_time = time.time()
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            batch = batch.to(args.device)
            pred_batch = model(batch).cpu().numpy()
            pred_middle.append(pred_batch[:, :, middle - 1])
            pred_last.append(pred_batch[:, :, last - 1])
        pred_middle = np.concatenate(pred_middle, axis=0).transpose(1, 0)
        pred_last = np.concatenate(pred_last, axis=0).transpose(1, 0)
        last_output = pred_batch[-1]
    end_time = time.time()
    print('| 数据:{} 批量:{} 平均耗时:{:.4f} |'
          .format(args.data_path, args.batch, (end_time - start_time) / len(pred_middle)))
    # 画图
    draw(pred_middle, pred_last, output_data, middle, last)
    draw_predict(last_data, last_output)
    print(f'| 画图保存位置:{args.save_path} |')


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, input_data):
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.input_data = input_data
        self.device = args.device

    def __len__(self):
        return self.input_data.shape[1] - self.input_size - self.output_size + 1

    def __getitem__(self, index):
        boundary = index + self.input_size
        series = self.input_data[:, index:boundary]  # 输入数据
        series = torch.tensor(series)  # 转换为tensor
        series = series.type(torch.float16) if self.device == 'cuda' else series.type(torch.float32)
        return series


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    predict_pt()
