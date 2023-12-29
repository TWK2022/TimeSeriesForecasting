## pytorch时间序列预测训练框架(带有测试数据)
>代码兼容性较强，使用的是一些基本的库、基础的函数  
>在argparse中可以选择使用wandb，能在wandb网站中生成可视化的训练过程
### 1，环境
>torch：https://pytorch.org/get-started/previous-versions/
>```
>pip install tqdm wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
>```
### 2，数据格式
>参考dataset中的样例  
>├── 数据集路径：data_path(.csv)
### 3，run.py
>模型训练时运行该文件，argparse中有对每个参数的说明
### 4，test_pt.py
>使用训练好的pt模型预测
### 5. export_onnx.py
>将pt模型导出为onnx模型
### 6. test_onnx.py
>使用导出的onnx模型预测
### 7，export_trt_record
>文档中有onnx模型导出为tensort模型的详细说明
### 8，test_trt.py
>使用导出的trt模型预测
### 其他
>github链接：https://github.com/TWK2022/TimeSeriesForecasting  
>学习笔记：https://github.com/TWK2022/notebook  
>邮箱：1024565378@qq.com  