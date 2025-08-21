import os, datetime, pytz
import numpy as np
import pandas as pd

folder = r'D:\L2_DATA_T0_ETF\his_data'

def fix_raw_data(date, code, result_folder):
    """
    数据预处理
    """

    result_file_name = os.path.join(result_folder, f'{code}_{date}')
    if os.path.exists(result_file_name):
        return

    print(f'fix_raw_data: {date} {code}', flush=True)

    file = os.path.join(folder, date, code, '十档盘口.csv')
    begin_t = os.environ.get('train_data_begin_time', '09:30')
    end_t = os.environ.get('train_data_end_time', '15:00')

    if os.path.exists(file):
        _data = pd.read_csv(file, encoding='gbk')

        # 删除完全重复的行
        _data = _data.drop_duplicates(keep='first')

        # 格式化 时间
        _data['时间'] = pd.to_datetime(_data['时间'])
        # 截取 10:00 - 14:30
        _data = _data[(_data["时间"].dt.time >= pd.to_datetime(begin_t).time()) & (
            _data["时间"].dt.time < pd.to_datetime(end_t).time())].reset_index(drop=True)
        _data = _data[(_data["时间"].dt.time <= pd.to_datetime('11:30:00').time()) | (
            _data["时间"].dt.time > pd.to_datetime('13:00:00').time())].reset_index(drop=True)
        if len(_data) == 0:
            print(f'fix_raw_data: {date} {code} 没有数据')
            return

        # 判断是否有 涨跌停
        # 涨跌停 不使用
        zt = ((_data['卖1价'] == 0) & (_data['卖1量'] == 0)).any()
        dt = ((_data['买1价'] == 0) & (_data['买1量'] == 0)).any()
        if zt or dt:
            print(f'fix_raw_data: {date} {code} 存在涨跌停')
            return

        _data = _data.reset_index(drop=True)

        # 删除列 '卖1价' 和 '买1价' 中存在 NaN 值的行
        # _data = _data.dropna(subset=['卖1价', '买1价']).reset_index(drop=True)
        # 暂时不允许 '卖1价', '买1价' 存在 NaN
        msg = ''
        if _data['卖1价'].isna().any():
            msg += f'{date} {code} 卖1价存在 NaN\n'
        if _data['买1价'].isna().any():
            msg += f'{date} {code} 买1价存在 NaN\n'
        if msg:
            raise Exception(msg)

        # 可以容忍的异常值处理
        # 2-10 档位价格nan填充, 使用上一个档位数据 +-0.001 进行填充
        for i in range(2, 11):
            # 买价
            _data.loc[:, f'买{i}价'] = _data[f'买{i}价'].fillna(_data[f'买{i-1}价'] - 0.001)
            # 卖价
            _data.loc[:, f'卖{i}价'] = _data[f'卖{i}价'].fillna(_data[f'卖{i-1}价'] + 0.001)
        # 盘口量nan与0都用1填充
        vol_cols = [i for i in list(_data) if '量' in i]
        _data[vol_cols] = _data[vol_cols].replace(0, np.nan).fillna(1)

        # 删除 总卖 总买
        _data = _data.drop(columns=['总卖', '总买'])

        return _data

    else:
        print(f'{date} {code} 没有数据, {file} 不存在')
