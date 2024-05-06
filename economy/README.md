## 经济
### 1，economy/tushare/get_data.py
>获取数据
### 2，economy/data_screen.py
>筛选股票
### 3，economy/data_add.py
>补充数据
### 4，run.py
>训练测试模型
>```
>python run.py --data_path economy/dataset/***_add.csv --input_column input_column.txt --output_column output_column.txt --divide 19,1 --model_type l
>```
>训练最终模型
>```
>python run.py --data_path economy/dataset/***_add.csv --input_column input_column.txt --output_column output_column.txt --divide 19,1 --divide_all True --weight_again True --model_type l
>```
### 5，economy/simulate.py
>模拟交易