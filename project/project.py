import os
import torch
import argparse
import numpy as np
import pandas as pd
from model.layer import deploy
from block.util import read_column

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|测试|')
parser.add_argument('--model_path', default='best.pt', type=str, help='|pt模型位置|')
parser.add_argument('--data_path', default=r'prepare/dataset/华银电力_add.csv', type=str, help='|数据位置|')
parser.add_argument('--input_column', default='input_column.txt', type=str, help='|选择输入的变量，可传入.txt|')
parser.add_argument('--output_column', default='output_column.txt', type=str, help='|选择预测的变量，可传入.txt|')
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
class project:
    def __init__(self, args):
        self.input_size = args.input_size
        self.output_size = args.output_size
        model_path = args.model_path
        device = args.device
        data_path = args.data_path
        input_column = args.input_column
        output_column = args.output_column
        # 加载模型
        model_dict = torch.load(model_path, map_location='cpu')
        model = model_dict['model']
        model = deploy(model, model_dict['mean_input'], model_dict['mean_output'], model_dict['std_input'],
                       model_dict['std_output']).eval().to(device)
        self.model = model.half() if device == 'cuda' else model.float()
        epoch = model_dict['epoch_finished']
        mse = round(model_dict['val_mse'], 4)
        print(f'| 模型加载成功:{model_path} | epoch:{epoch} | mse:{mse}|')
        # 加载数据
        try:
            df = pd.read_csv(data_path, encoding='utf-8', index_col=0)
        except:
            df = pd.read_csv(data_path, encoding='gbk', index_col=0)
        self.input_data = np.array(df[input_column]).astype(np.float32).T
        self.output_data = np.array(df[output_column]).astype(np.float32).T
        # 其他
        self.state = None

    def predict(self):
        self.state = 0
        with torch.no_grad():
            for index in range(self.input_size, self.input_data.shape[1] - self.output_size):  # index是预测的第一步
                tensor = torch.tensor(self.input_data[:, index - self.input_size:index]).unsqueeze(0)
                true = self.output_data[:, index:index + self.output_size]
                pred = self.model(tensor)[0].cpu().numpy()
                now = self.output_data[index - 1]
                if self._rise(now, pred):  # 预测一直上涨
                    self.state = 1
                    pass
                elif self._decline(now, pred):  # 预测一直下降
                    pass

    def _rise(self, now, pred):
        pass

    def _decline(self, now, pred):
        pass


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = project(args)
    model.predict()
