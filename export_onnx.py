import os
import torch
import argparse
from model.layer import deploy

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='将pt模型字典中的模型转为onnx，同时导出类别信息')
parser.add_argument('--weight', default='best.pt', type=str, help='|模型位置|')
parser.add_argument('--input_size', default=128, type=int, help='|输入的长度|')
parser.add_argument('--input_column_num', default=3, type=int, help='|输入的变量数量|')
parser.add_argument('--batch', default=0, type=int, help='|输入图片批量，0为动态|')
parser.add_argument('--sim', default=True, type=bool, help='|使用onnxsim压缩简化模型|')
parser.add_argument('--device', default='cuda', type=str, help='|在哪个设备上加载模型|')
parser.add_argument('--float16', default=True, type=bool, help='|转换的onnx模型数据类型，需要GPU，False时为float32|')
args = parser.parse_args()
args.weight = args.weight.split('.')[0] + '.pt'
args.save_name = args.weight.split('.')[0] + '.onnx'
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.weight), f'没有找到模型{args.weight}'
if args.float16:
    assert torch.cuda.is_available(), 'cuda不可用，因此无法使用float16'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def export_onnx():
    model_dict = torch.load(args.weight, map_location='cpu')
    model = model_dict['model']
    model = deploy(model, model_dict['input_mean'], model_dict['input_std'], model_dict['output_mean'],
                   model_dict['output_std'])
    model = model.eval().half().to(args.device) if args.float16 else model.eval().float().to(args.device)
    input_shape = torch.rand(1, args.input_column_num, args.input_size,
                             dtype=torch.float16 if args.float16 else torch.float32).to(args.device)
    torch.onnx.export(model, input_shape, args.save_name,
                      opset_version=12, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {args.batch: 'batch_size'}, 'output': {args.batch: 'batch_size'}})
    print(f'| 转为onnx模型成功:{args.save_name} |')
    if args.sim:
        import onnx
        import onnxsim

        model_onnx = onnx.load(args.save_name)
        model_simplify, check = onnxsim.simplify(model_onnx)
        onnx.save(model_simplify, args.save_name)
        print(f'| 使用onnxsim简化模型成功:{args.save_name} |')


if __name__ == '__main__':
    export_onnx()
