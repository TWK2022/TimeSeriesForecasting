# economy
### 环境
>```
>pip install finta tushare pyperclip pyautogui opencv-python onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
>```
>### stock_process/industry_choice.py
>根据"dataset/industry"选择行业和股票：dataset/industry_choice.yaml  
### stock_process/tushare_block.py
>根据"dataset/industry_choice.yaml"收集股票数据：dataset/stock  
>需要密钥
### stock_process/data_add.py
>根据"dataset/industry_choice.yaml"和"dataset/stock"补全股票数据：dataset/stock_add
### stock_process/data_screen.py
>根据"dataset/industry_choice.yaml"和"dataset/stock_add"用规则筛选股票：dataset/industry_screen.yaml
### predict/predict.py
>根据"dataset/industry_screen.yaml"和"dataset/stock_add"训练模型筛选股票：dataset/simulate.yaml、dataset/predict.yaml