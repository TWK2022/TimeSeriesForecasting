import os
import argparse
import onnxruntime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
# 模型预测: 用预测的序列的中间值画图
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|模型预测|')
parser.add_argument('--model_path', default='best.onnx', type=str, help='|模型位置|')
parser.add_argument('--data_path', default='dataset/ETTh1.csv', type=str, help='|数据位置|')
parser.add_argument('--input_len', default=96, type=int, help='|输入时间长度|')
parser.add_argument('--output_len', default=24, type=int, help='|输出时间长度|')
parser.add_argument('--input_column', default='HUFL,HULL,MUFL,MULL,LUFL,LULL,OT', type=str, help='|输入变量，可为.txt|')
parser.add_argument('--output_column', default='HUFL,HULL,MUFL,MULL,LUFL,LULL,OT', type=str, help='|输出变量，可为.txt|')
parser.add_argument('--save_dir', default='draw_predict', type=str, help='|画图保存位置|')
parser.add_argument('--pred_len', default=1000, type=int, help='|预测和画图长度|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量|')
parser.add_argument('--device', default='cpu', type=str, help='|设备|')
parser.add_argument('--float16', default=True, type=bool, help='|数据类型|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


# -------------------------------------------------------------------------------------------------------------------- #
class predict_class:
    def __init__(self, args=args):
        args.input_column = predict_class.read_column(args.input_column)  # column处理
        args.output_column = predict_class.read_column(args.output_column)  # column处理
        self.args = args
        self.device = args.device
        self.float16 = args.float16
        provider = 'CUDAExecutionProvider' if args.device.lower() in ['gpu', 'cuda'] else 'CPUExecutionProvider'
        self.model = onnxruntime.InferenceSession(args.model_path, providers=[provider])  # 加载模型和框架
        self.input_name = self.model.get_inputs()[0].name  # 获取输入名称
        self.output_name = self.model.get_outputs()[0].name  # 获取输出名称

    @staticmethod
    def read_column(column_file):  # column处理
        if os.path.exists(column_file):
            with open(column_file, encoding='utf-8') as f:
                column = [_.strip() for _ in f.readlines()]
        else:
            column = column_file.split(',')
        return column

    def predict(self, data_path=args.data_path):
        args = self.args
        # 数据
        df = pd.read_csv(data_path, encoding='utf-8', index_col=0)
        input_data = df[args.input_column].values.astype(np.float32)[-args.pred_len - args.input_len:]
        output_data = df[args.output_column].values.astype(np.float32)[-args.pred_len - args.input_len:]
        # 预测
        label_middle = []
        pred_middle = []
        half = args.output_len // 2
        for index in range(0, len(input_data) - args.input_len - args.output_len + 1):
            input_ = input_data[index:index + args.input_len]
            output_ = output_data[index + args.input_len:index + args.input_len + args.output_len]
            input_ = input_[np.newaxis].astype(np.float16 if self.float16 else np.float32)
            pred = self.model.run([self.output_name], {self.input_name: input_})[0][0]
            label_middle.append(output_[half])
            pred_middle.append(pred[half])
        label_middle = np.array(label_middle)
        pred_middle = np.array(pred_middle)
        self._draw(label_middle, pred_middle)

    def _draw(self, label_middle, pred_middle):
        args = self.args
        for index, name in enumerate(args.output_column):
            save_path = f'{args.save_dir}/{name}.jpg'
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
            plt.rcParams['axes.unicode_minus'] = False  # 使用字体时让坐标轴正常显示负号
            plt.title(name)
            plt.plot(label_middle[:, index], color='black', label='label')
            plt.plot(pred_middle[:, index], color='cyan', label='pred')
            plt.grid()
            plt.legend()
            plt.savefig(save_path)
            plt.close()


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = predict_class()
    model.predict()
