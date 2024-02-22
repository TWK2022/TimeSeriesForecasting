import os
import time
import argparse
import tensorrt
import numpy as np
import pandas as pd
import pycuda.autoinit
import pycuda.driver as cuda
import matplotlib.pyplot as plt
from block.util import read_column

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|tensorrt模型推理|')
parser.add_argument('--model_path', default='best.trt', type=str, help='|trt模型位置|')
parser.add_argument('--data_path', default=r'./dataset/sin_cos.csv', type=str, help='|数据路径|')
parser.add_argument('--input_column', default='sin,cos', type=str, help='|选择输入的变量|')
parser.add_argument('--output_column', default='mix', type=str, help='|选择预测的变量|')
parser.add_argument('--input_size', default=512, type=int, help='|输入长度|')
parser.add_argument('--output_size', default=256, type=int, help='|输出长度|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量，要与导出的模型对应，一般为1|')
parser.add_argument('--float16', default=True, type=bool, help='|推理数据类型，要与导出的模型对应，False时为float32|')
parser.add_argument('--plot_len', default=1000, type=int, help='|画图长度，取数据的倒数plot_len个|')
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


def predict_trt():
    # 加载模型
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)  # 创建日志记录信息
    with tensorrt.Runtime(logger) as runtime, open(args.model_path, "rb") as f:
        model = runtime.deserialize_cuda_engine(f.read())  # 读取模型并构建一个对象
    np_type = tensorrt.nptype(model.get_tensor_dtype('input'))  # 获取接口的数据类型并转为np的字符串格式
    h_input = np.zeros(tensorrt.volume(model.get_tensor_shape('input')), dtype=np_type)  # 获取输入的形状(一维)
    h_output = np.zeros(tensorrt.volume(model.get_tensor_shape('output')), dtype=np_type)  # 获取输出的形状(一维)
    d_input = cuda.mem_alloc(h_input.nbytes)  # 分配显存空间
    d_output = cuda.mem_alloc(h_output.nbytes)  # 分配显存空间
    bindings = [int(d_input), int(d_output)]  # 绑定显存输入输出
    stream = cuda.Stream()  # 创建cuda流
    model_context = model.create_execution_context()  # 创建模型推理器
    print(f'| 加载模型成功:{args.model_path} |')
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
    input_batch = np.stack(input_batch, axis=0).reshape(input_len, -1)
    # 推理
    start_time = time.time()
    middle = args.output_size // 2
    last = args.output_size
    pred_middle = [0 for _ in range(input_len)]
    pred_last = [0 for _ in range(input_len)]
    for i in range(input_len):
        cuda.memcpy_htod_async(d_input, input_batch[i], stream)  # 将输入数据从CPU锁存复制到GPU显存
        model_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)  # 执行推理
        cuda.memcpy_dtoh_async(h_output, d_output, stream)  # 将输出数据从GPU显存复制到CPU锁存
        stream.synchronize()  # 同步线程
        pred_batch = h_output.copy().reshape(len(args.output_column), args.output_size)
        pred_middle[i] = pred_batch[:, middle - 1][np.newaxis]
        pred_last[i] = pred_batch[:, last - 1][np.newaxis]
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
    predict_trt()
