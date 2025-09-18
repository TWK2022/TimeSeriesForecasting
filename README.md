# pytorch时间序列预测训练框架
### 1，环境
>torch: https://pytorch.org/get-started/previous-versions/
>```
>pip install pandas xlrd tqdm wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
>pip install onnx onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
>```
### 2，数据格式
>参考dataset中的样例
### 3，run.py
>模型训练，argparse中有每个参数的说明
### 4，predict.py
>模型预测
### 5，export_onnx.py
>onnx模型导出
### 6，predict_onnx.py
>onnx模型预测
### 其他
>github链接: https://github.com/TWK2022/TimeSeriesForecasting  
>学习笔记: https://github.com/TWK2022/notebook  
>邮箱: 1024565378@qq.com
***
### ETTh1.csv
|    模型(m)     | input_column | output_column | input_size | output_size | divide | train_mse | val_mse |
|:------------:|:------------:|:-------------:|:----------:|:-----------:|:------:|:---------:|:-------:|
|     tsf      |     all      |      all      |     96     |     24      |  19:1  |   0.223   |  0.262  |
|     lstm     |     all      |      all      |     96     |     24      |  19:1  |   0.258   |  0.270  |
|    linear    |     all      |      all      |     96     |     24      |  19:1  |   0.228   |  0.259  |
| itransformer |     all      |      all      |     96     |     24      |  19:1  |   0.234   |  0.260  |
