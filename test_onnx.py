import os
import time
import argparse
import onnxruntime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from block.util import read_column

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|onnx模型推理|')
parser.add_argument('--model_path', default='best.onnx', type=str, help='|onnx模型位置|')
parser.add_argument('--data_path', default=r'./dataset/sin_cos.csv', type=str, help='|数据路径|')
parser.add_argument('--input_column', default='sin,cos', type=str, help='|选择输入的变量|')
parser.add_argument('--output_column', default='mix', type=str, help='|选择预测的变量|')
parser.add_argument('--input_size', default=512, type=int, help='|输入长度|')
parser.add_argument('--output_size', default=256, type=int, help='|输出长度|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量，要与导出的模型对应|')
parser.add_argument('--device', default='cuda', type=str, help='|推理设备|')
parser.add_argument('--float16', default=True, type=bool, help='|推理数据类型，要与导出的模型对应，False时为float32|')
parser.add_argument('--plot_len', default=1000, type=int, help='|画图长度，取数据的倒数plot_len个|')
args = parser.parse_args()
args.input_column = read_column(args.input_column)  # column处理
args.output_column = read_column(args.output_column)  # column处理
args.save_path = 'save_image'
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'! model_path不存在:{args.model_path} !'
assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
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
        plt.title(name)
        plt.plot(true[i, :], color='green', label=f'{args.output_column[i]}_true')
        plt.plot(middle_plot[i, :], color='orange', label=f'{args.output_column[i]}_{middle}')
        plt.plot(last_plot[i, :], color='red', label=f'{args.output_column[i]}_{last}')
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
    true = true[input_cut:]
    pred = pred[input_cut:]
    for i in range(len(args.output_column)):
        name = f'{args.output_column[i]}_last_predict'
        plt.title(name)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
        plt.plot(true[i, :], color='green', label=f'{args.output_column[i]}_true')
        plt.plot(pred[i, :], color='cyan', label=f'{args.output_column[i]}_pred')
        plt.legend()
        plt.savefig(args.save_path + '/' + name + '.jpg')
        plt.close()


def test_onnx():
    # 加载模型
    provider = 'CUDAExecutionProvider' if args.device.lower() in ['gpu', 'cuda'] else 'CPUExecutionProvider'
    model = onnxruntime.InferenceSession(args.model_path, providers=[provider])  # 加载模型和框架
    input_name = model.get_inputs()[0].name  # 获取输入名称
    output_name = model.get_outputs()[0].name  # 获取输出名称
    print(f'| 模型加载成功:{args.model_path} |')
    # 加载数据
    try:
        df = pd.read_csv(args.data_path, encoding='utf-8')
    except:
        df = pd.read_csv(args.data_path, encoding='gbk')
    input_data = np.array(df[args.input_column].astype(np.float32)).transpose(1, 0)
    input_data = input_data[:, -args.plot_len:]  # 限定长度方便画图
    output_data = np.array(df[args.output_column].astype(np.float32)).transpose(1, 0)
    last_data = output_data[:, -args.input_size - args.output_size:]
    output_data = output_data[:, -args.plot_len:]  # 限定长度方便画图
    # 数据处理
    input_data = input_data.astype(np.float16 if args.float16 else np.float32)
    input_len = input_data.shape[1] - args.input_size - args.output_size + 1
    input_batch = [input_data[:, _:_ + args.input_size] for _ in range(input_len)]
    input_batch = np.stack(input_batch, axis=0)
    # 推理
    middle = args.output_size // 2
    last = args.output_size
    pred_middle = []
    pred_last = []
    n = input_len // args.batch
    start_time = time.time()
    if n > 0:  # 如果预测数量>=批量(分批预测)
        for i in range(n):
            batch = input_batch[i * args.batch:(i + 1) * args.batch]
            pred_batch = model.run([output_name], {input_name: batch})[0]
            pred_middle.append(pred_batch[:, :, middle - 1])
            pred_last.append(pred_batch[:, :, last - 1])
        if input_len % args.batch > 0:  # 如果图片数量没有刚好满足批量
            batch = input_batch[(i + 1) * args.batch:]
            pred_batch = model.run([output_name], {input_name: batch})[0]
            pred_middle.append(pred_batch[:, :, middle - 1])
            pred_last.append(pred_batch[:, :, last - 1])
    else:  # 如果图片数量<批量(直接预测)
        batch = input_batch
        pred_batch = model.run([output_name], {input_name: batch})[0]
        pred_middle.append(pred_batch[:, :, middle - 1])
        pred_last.append(pred_batch[:, :, last - 1])
    pred_middle = np.concatenate(pred_middle, axis=0).transpose(1, 0)
    pred_last = np.concatenate(pred_last, axis=0).transpose(1, 0)
    last_output = pred_batch[-1]
    end_time = time.time()
    print('| 数据:{} 批量:{} 平均耗时:{:.4f} |'.format(input_len, args.batch, (end_time - start_time) / input_len))
    # 画图
    draw(pred_middle, pred_last, output_data, middle, last)
    draw_predict(last_data, last_output)
    print(f'| 画图保存位置:{args.save_path} |')


if __name__ == '__main__':
    test_onnx()
