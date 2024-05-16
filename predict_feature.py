import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.layer import deploy
from block.util import read_column

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|预测未来|')
parser.add_argument('--model_path', default='best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--data_path', default=r'./dataset/sin_cos.csv', type=str, help='|数据位置|')
parser.add_argument('--input_column', default='sin,cos', type=str, help='|选择输入的变量，可传入.txt|')
parser.add_argument('--output_column', default='mix', type=str, help='|选择预测的变量，可传入.txt|')
parser.add_argument('--input_size', default=96, type=int, help='|输入长度|')
parser.add_argument('--output_size', default=24, type=int, help='|输出长度|')
parser.add_argument('--device', default='cpu', type=str, help='|推理设备|')
args = parser.parse_args()
args.input_column = read_column(args.input_column)  # column处理
args.output_column = read_column(args.output_column)  # column处理
args.save_path = 'save_image'
# -------------------------------------------------------------------------------------------------------------------- #
assert os.path.exists(args.model_path), f'! model_path不存在:{args.model_path} !'
assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# -------------------------------------------------------------------------------------------------------------------- #
def draw(output_data, pred):
    for index, (true, out_) in enumerate(zip(output_data, pred)):
        name = args.data_path.split('/')[-1].split('.')[0]
        zero = torch.zeros(len(true))
        out_ = np.concatenate([zero, out_], axis=0)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 使用字体时让坐标轴正常显示负号
        plt.title(name)
        plt.grid()
        plt.plot(true, color='green', label=f'{args.output_column[index]}_in')
        plt.plot(out_, color='cyan', label=f'{args.output_column[index]}_output')
        plt.savefig(args.save_path + f'/{name}_{args.output_column[index]}' + '.jpg')
        plt.close()


def predict_pt():
    # 加载模型
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model']
    model = deploy(model, model_dict['mean_input'], model_dict['mean_output'], model_dict['std_input'],
                   model_dict['std_output'], model_dict['mean_special'],
                   model_dict['std_special']).eval().to(args.device)
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
    input_data = input_data[:, -args.input_size:]
    tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    output_data = np.array(df[args.output_column]).astype(np.float32).T
    output_data = output_data[:, -300:]
    # 推理
    with torch.no_grad():
        pred = model(tensor)[0].cpu().numpy()
    # 画图
    draw(output_data, pred)
    print(f'| 画图保存位置:{args.save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    predict_pt()
