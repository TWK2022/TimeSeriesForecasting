import os


def read_column(column_file):  # column处理
    if os.path.exists(column_file):
        with open(column_file, encoding='utf-8') as f:
            column = [_.strip() for _ in f.readlines()]
    else:
        column = column_file.split(',')
    return column
