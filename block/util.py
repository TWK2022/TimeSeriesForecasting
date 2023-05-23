import os


def read_column(column_file):  # column处理
    if os.path.exists(column_file):
        with open(column_file) as f:
            column = [_.strip() for _ in f.readlines()]
        print(f'| 读取column文件:{column_file} |')
    else:
        column = column_file.split(',')
    return column
