import onnx
import torch
import argparse
from model.layer import deploy

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|模型转为onnx|')
parser.add_argument('--weight', default='best.pt', type=str, help='|模型位置|')
parser.add_argument('--input_len', default=96, type=int, help='|输入时间长度|')
parser.add_argument('--input_column_num', default=7, type=int, help='|输入变量数量|')
parser.add_argument('--sim', default=True, type=bool, help='|使用onnxsim压缩简化模型|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--float16', default=True, type=bool, help='|数据类型|')
args = parser.parse_args()
args.weight = args.weight.split('.')[0] + '.pt'
args.save_path = args.weight.split('.')[0] + '.onnx'
args.device = 'cpu' if not torch.cuda.is_available() else args.device


# -------------------------------------------------------------------------------------------------------------------- #
def export_onnx(args=args):
    model_dict = torch.load(args.weight, map_location='cpu', weights_only=False)
    model = deploy(model_dict['model'], model_dict['mean_input'], model_dict['mean_output'],
                   model_dict['std_input'], model_dict['std_output'])
    model = model.eval().half().to(args.device) if args.float16 else model.eval().float().to(args.device)
    input_one = torch.rand(1, args.input_len, args.input_column_num,
                           dtype=torch.float16 if args.float16 else torch.float32).to(args.device)
    torch.onnx.export(model, input_one, args.save_path, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f'| onnx模型转换成功:{args.save_path} |')
    if args.sim:
        import onnxsim
        model_onnx = onnx.load(args.save_path)
        model_simplify, check = onnxsim.simplify(model_onnx)
        onnx.save(model_simplify, args.save_path)
        print(f'| onnxsim简化模型成功:{args.save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    export_onnx()
