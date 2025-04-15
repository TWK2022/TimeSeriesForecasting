import os
import pandas as pd


class strategy_class:
    '''
        release: 放量
    '''

    def __init__(self):
        path_dir = os.path.dirname(os.path.dirname(__file__)) + '/dataset/stock_add'
        name_list = os.listdir(path_dir)
        self.path_list = [path_dir + '/' + _ for _ in name_list]

    def count(self):
        self.volume_release()

    def volume_release(self, scale=1.5):
        result_dict = {
            '底部下方放量': [0, 0, 0, 0],
            '底部中间放量': [0, 0, 0, 0],
            '底部上方放量': [0, 0, 0, 0],
            '顶部下方放量': [0, 0, 0, 0],
            '顶部中间放量': [0, 0, 0, 0],
            '顶部上方放量': [0, 0, 0, 0],
        }
        for path in self.path_list:
            df = pd.read_csv(path)
            value = df['均价'].values
            volume = df['成交量'].values
            close = df['收盘价'].values
            close_5 = df['收盘价_EMA_5'].values
            close_10 = df['收盘价_EMA_10'].values
            for index in range(1, len(df) - 1):
                # 放量
                if volume[index] < scale * volume[index - 1]:
                    continue
                # 底部下方放量
                if close[index - 1] < close_5[index - 1] < close_10[index - 1]:
                    result_dict['底部下方放量'][0] += 1
                    if value[index] > close[index - 1] or value[index + 1] > close[index - 1]:
                        result_dict['底部下方放量'][1] += 1
                    if value[index + 1] > value[index]:
                        result_dict['底部下方放量'][2] += 1
                    if close[index + 1] > close_5[index + 1]:
                        result_dict['底部下方放量'][3] += 1
                # 底部中间放量
                elif close_5[index - 1] < close[index - 1] < close_10[index - 1]:
                    result_dict['底部中间放量'][0] += 1
                    if value[index] > close[index - 1] or value[index + 1] > close[index - 1]:
                        result_dict['底部中间放量'][1] += 1
                    if value[index + 1] > value[index]:
                        result_dict['底部中间放量'][2] += 1
                    if close[index + 1] > close_10[index + 1]:
                        result_dict['底部中间放量'][3] += 1
                # 底部上方放量
                elif close_5[index - 1] < close_10[index - 1] < close[index - 1]:
                    result_dict['底部上方放量'][0] += 1
                    if value[index] > close[index - 1] or value[index + 1] > close[index - 1]:
                        result_dict['底部上方放量'][1] += 1
                    if value[index + 1] > value[index]:
                        result_dict['底部上方放量'][2] += 1
                # 顶部下方放量
                elif close[index - 1] < close_10[index - 1] < close_5[index - 1]:
                    result_dict['顶部下方放量'][0] += 1
                    if value[index] > close[index - 1] or value[index + 1] > close[index - 1]:
                        result_dict['顶部下方放量'][1] += 1
                    if value[index + 1] > value[index]:
                        result_dict['顶部下方放量'][2] += 1
                    if close[index + 1] > close_10[index + 1]:
                        result_dict['顶部下方放量'][3] += 1
                # 顶部中间放量
                elif close_10[index - 1] < close[index - 1] < close_5[index - 1]:
                    result_dict['顶部中间放量'][0] += 1
                    if value[index] > close[index - 1] or value[index + 1] > close[index - 1]:
                        result_dict['顶部中间放量'][1] += 1
                    if value[index + 1] > value[index]:
                        result_dict['顶部中间放量'][2] += 1
                    if close[index + 1] > close_5[index + 1]:
                        result_dict['顶部中间放量'][3] += 1
                # 顶部中间放量
                elif close_10[index - 1] < close_5[index - 1] < close[index - 1]:
                    result_dict['顶部上方放量'][0] += 1
                    if value[index] > close[index - 1] or value[index + 1] > close[index - 1]:
                        result_dict['顶部上方放量'][1] += 1
                    if value[index + 1] > value[index]:
                        result_dict['顶部上方放量'][2] += 1
        print(f'| 条件 | 样本数 | 收盘价>昨日 | 均价明日>今日 | 明日上穿5/10日线 |')
        for key in result_dict.keys():
            total, rise, rise_continue, cross = result_dict[key]
            if total:
                print(f'| {key} | {total} | {rise / total:.2f} | {rise_continue / total:.2f} | {cross / total:.2f} |')


if __name__ == '__main__':
    model = strategy_class()
    model.count()
