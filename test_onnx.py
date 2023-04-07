import os
import time
import argparse
import onnxruntime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='onnx推理')
parser.add_argument('--model_path', default='best.onnx', type=str, help='|onnx模型位置|')
parser.add_argument('--data_path', default=r'./dataset/ETTh1.csv', type=str, help='|数据路径|')
parser.add_argument('--input_column', default='HUFL,HULL,MUFL,MULL,LUFL,LULL,OT', type=str, help='|选择输入的变量|')
parser.add_argument('--output_column', default='HUFL,HULL,MUFL,MULL,LUFL,LULL,OT', type=str, help='|选择预测的变量|')
parser.add_argument('--input_size', default=128, type=int, help='|输入的长度|')
parser.add_argument('--output_size', default=32, type=int, help='|输出的长度|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量，要与导出的模型对应|')
parser.add_argument('--device', default='cuda', type=str, help='|用CPU/GPU推理|')
parser.add_argument('--float16', default=True, type=bool, help='|推理数据类型，要与导出的模型对应，False时为float32|')
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
def draw(pred, true, number):  # pred为模型输出，true为真实数据，pred和true长度不相等
    # 画图
    x = np.arange(len(true) + number)
    y_pred = np.zeros(len(true) + number)
    n = len(x) // 500
    n += 1 if len(x) % 500 else 0
    for i in range(len(args.output_column)):
        y_pred[args.input_size + number - 1:] = pred[:, i]
        y_true = true[:, i]
        for j in range(n):
            name = args.output_column[i] + f'_{500 * j}-{500 * (j + 1)}(number_{number})'
            plt.title(name)
            plt.plot(y_pred[500 * j:500 * (j + 1)], color='cyan')
            plt.plot(y_true[500 * j:500 * (j + 1)], color='green')
            plt.savefig(args.save_path + '/' + name + '.jpg')
            plt.close()


def test_onnx():
    # 加载模型
    provider = 'CUDAExecutionProvider' if args.device.lower() in ['gpu', 'cuda'] else 'CPUExecutionProvider'
    session = onnxruntime.InferenceSession(args.model_path, providers=[provider])  # 加载模型和框架
    input_name = session.get_inputs()[0].name  # 获取输入名称
    output_name = session.get_outputs()[0].name  # 获取输出名称
    print(f'| 模型加载成功:{args.model_path} |')
    # 加载数据
    start_time = time.time()
    df = pd.read_csv(args.data_path)
    input_data = np.array(df[args.input_column].astype(np.float32))[-1000 + args.output_size:]  # 限定长度方便画图
    output_data = np.array(df[args.output_column].astype(np.float32))[-1000 + args.output_size:]  # 限定长度方便画图
    input_data = input_data.astype(np.float16 if args.float16 else np.float32)
    input_len = len(input_data) - args.input_size + 1
    input_batch = [input_data[_:_ + args.input_size] for _ in range(input_len)]
    input_batch = np.stack(input_batch, axis=0).transpose(0, 2, 1)
    end_time = time.time()
    print('| 数据加载成功:{} 耗时:{:.4f} |'.format(args.data_path, end_time - start_time))
    # 推理
    start_time = time.time()
    middle = args.output_size // 2
    last = args.output_size
    result_middle = []
    result_last = []
    n = input_len // args.batch
    if n > 0:  # 如果预测数量>=批量(分批预测)
        for i in range(n):
            batch = input_batch[i * args.batch:(i + 1) * args.batch]
            pred_batch = session.run([output_name], {input_name: batch})[0]
            result_middle.append(pred_batch[:, :, middle - 1])
            result_last.append(pred_batch[:, :, last - 1])
        if input_len % args.batch > 0:  # 如果图片数量没有刚好满足批量
            batch = input_batch[(i + 1) * args.batch:]
            pred_batch = session.run([output_name], {input_name: batch})[0]
            result_middle.append(pred_batch[:, :, middle - 1])
            result_last.append(pred_batch[:, :, last - 1])
    else:  # 如果图片数量<批量(直接预测)
        batch = input_batch
        pred_batch = session.run([output_name], {input_name: batch})[0]
        result_middle.append(pred_batch[:, :, middle - 1])
        result_last.append(pred_batch[:, :, last - 1])
    result_middle = np.concatenate(result_middle, axis=0)
    result_last = np.concatenate(result_last, axis=0)
    end_time = time.time()
    print('| 数据:{} 批量:{} 每次预测耗时:{:.4f} |'.format(input_len, args.batch, (end_time - start_time) / input_len))
    # 画图
    draw(result_middle, output_data, middle)
    draw(result_last, output_data, last)
    print(f'| 画图保存位置:{args.save_path} |')


if __name__ == '__main__':
    test_onnx()
