## pytorch时间序列预测训练框架(带有测试数据)
>代码兼容性较强，使用的是一些基本的库、基础的函数  
>包含分布式训练、混合精度训练、EMA平均指数移动调参、float16精度onnx/trt导出及预测   
>测试/导出模型时，序列处理、结果反处理会包含在模型中  
>在argparse中可以选择使用wandb，能在wandb网站中生成可视化的训练过程
### 数据格式如下  
>├── 数据集路径：data_path(.csv)  
### 1，run.py
>模型训练时运行该文件，argparse中有对每个参数的说明
### 2，test_pt.py
>使用训练好的pt模型预测
### 3，export_onnx.py
>将pt模型导出为onnx模型
### 4，test_onnx.py
>使用导出的onnx模型预测
### 5，export_trt_record
>文档中有onnx模型导出为tensort模型的详细说明
### 6，test_trt.py
>使用导出的trt模型预测
### 其他
>github链接:https://github.com/TWK2022/TimeSeriesForecasting  
>学习笔记:https://github.com/TWK2022/notebook  
>邮箱:1024565378@qq.com