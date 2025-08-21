# %%
# 读取预测数据
# 读取深度数据
# 真实标签可视化
# 预测标签可视化

import os, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import  tqdm
import pytz, time
import datetime

# %%
train_result_folder = r'D:\code\forecast_model\notebook\20250811_base'

# result_type = 'model_best'
result_type = 'model_final'
# result_type = 'model_dummy'

# test_label = True
test_label = False

# code = 'btcfdusd'
code = '513330'

predict_mode = 'threshold_max'
# predict_mode = ''
# predict_mode = '0.5761168599128723'
# predict_mode = 'threshold_012'

predict_mode = 'label' if test_label else predict_mode

# %%
_train_result_folder = os.path.join(train_result_folder, result_type) if ('model_best' not in train_result_folder) and ('model_final' not in train_result_folder) and ('model_dummy' not in train_result_folder) else train_result_folder
if not os.path.exists(_train_result_folder):
    _train_result_folder = train_result_folder

# 存在回测结果数据的标的
all_files =os.listdir(_train_result_folder)
# back_test_code = [i.split('_')[0] for i in all_files if 'asset_nets' in i and '.csv' in i]

files = [os.path.join(_train_result_folder, i).replace('\\', '/') for i in os.listdir(_train_result_folder) if '.csv' in i and '_' in i and 'asset_nets' not in i and 'trades' not in i ]
# files = [i for i in files if i.split('/')[-1].split('_')[0] in back_test_code]
files

# %% [markdown]
# # 读取预测数据

# %%
file = [i for i in files if code in i][0]
print(file)

symbol, begin, end = file.split('/')[-1][:-4].split('_')
print(symbol, begin, end)

predict_data = pd.read_csv(file)
print(predict_data.dtypes)

predict_data

# %%
if 'predict' not in list(predict_data):
    idx_0 = list(predict_data).index('0')
    y_len = len(list(predict_data)[idx_0:])
    if 'predict' not in list(predict_data) and not test_label:
        if predict_mode == 'threshold_max':
            # 取最大值
            predict_data['predict'] = predict_data.iloc[:, idx_0:].idxmax(axis=1).astype(int)

        else:
            if predict_mode == 'threshold_012':
                # 读取阈值
                print('使用阈值文件')
                threshold_file = file.replace(file.split('/')[-1], 'threshold.txt')
                if not os.path.exists(threshold_file):
                    raise FileNotFoundError('阈值文件不存在')
                threshold = [float(i) for i in open(threshold_file, 'r').readline().strip().split(',')]
            else:
                print('使用阈值字符串')
                threshold_list = predict_mode.split(',')
                while len(threshold_list)!= y_len:
                    threshold_list.append(threshold_list[0])
                threshold = [float(i) for i in threshold_list]

            threshold = threshold[:y_len]
            print(threshold)
            predict_bool = predict_data.iloc[:, idx_0:] >= threshold

            # 定义函数根据条件设置 predict 列的值
            def assign_prediction(row):
                for idx, _is in enumerate(row.to_list()):
                    if _is:
                        return idx
                return y_len - 1

            # 应用函数创建 predict 列
            predict_data['predict'] = predict_bool.apply(assign_prediction, axis=1)

predict_mode = predict_mode if predict_mode == 'threshold_max' else ''

# 回测标签用
if test_label:
    predict_data['predict'] = predict_data['target']

predict_data

# %%
predict_data[predict_data['timestamp'] == 1724995740]

# %%
predict_data['predict'].value_counts()

# %%
predict_data['predict'].value_counts()

# %%
predict_data['time'] = pd.to_datetime(predict_data['timestamp'] + 8*3600, unit='s')
predict_data = predict_data.set_index('time').sort_index()
# predict_data = predict_data.set_index('time')

begin_date = predict_data.index.min().date()
end_date = predict_data.index.max().date()
predict_dates = set([str(i)[:10] for i in predict_data.index.to_list()])
print(begin_date, end_date)
print(predict_dates)

predict_data

# %%
predict_data['2024-09-05 14:35:00':'2024-09-05 14:40:00'].describe()

# %%
# symbol[0] 是否为数字
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

# %% [markdown]
# # 读取深度数据

# %%
# raw文件名
is_t0 = is_number(symbol[0])

raws = []
if is_t0:
    for date in predict_dates:
        # raws.append(rf"Z:\L2_DATA\his_data\{date.replace('-', '')}\{code}\十档盘口.csv")
        raws.append(rf"D:\L2_DATA_T0_ETF\his_data\{date.replace('-', '')}\{code}\十档盘口.csv")

else:
    raw_folder = r'D:\code\featrue_data\binance_datas'
    raw_file_name_begin = f'depth_{symbol.upper()}'
    for i in os.listdir(raw_folder):
        if i.startswith(raw_file_name_begin):
            if not is_t0:
                _,_,_begin, _end = i.replace('.pkl', '').split('_')
                _begin, _end = _begin + '000', _end + '000'
                if not (_begin > end or _end < begin):
                    raws.append(os.path.join(raw_folder, i).replace('\\', '/'))
            else:
                raws.append(os.path.join(raw_folder, i).replace('\\', '/'))
                
raws

# %%
# 读取raw数据，只保留买卖1价
raws_data = pd.DataFrame()
for i in raws:
    if not is_t0:
        _df = pickle.load(open(i, 'rb')).loc[:, ['save_timestamp', '卖1价', '买1价']]
    else:
        _df = pd.read_csv(i, encoding='gbk')
        _df['时间'] = pd.to_datetime(_df['时间'])
        _df = _df.loc[((_df['时间'].dt.time >= datetime.time(9, 30)) & (_df['时间'].dt.time <= datetime.time(11, 30))) | ((_df['时间'].dt.time >= datetime.time(13)) & (_df['时间'].dt.time < datetime.time(14,57))), :].reset_index(drop=True)
        _df['timestamp'] = _df['时间'].apply(lambda x:int((x.timestamp() - 8*3600)))
        _df = _df.loc[:, ['timestamp', '卖1价', '买1价']]

    raws_data = pd.concat([raws_data, _df], ignore_index=True)
raws_data['mid'] = (raws_data['卖1价'] + raws_data['买1价']) / 2

if not is_t0:
    raws_data = raws_data.sort_values('save_timestamp').reset_index(drop=True)
    raws_data['timestamp'] = raws_data['save_timestamp'].apply(lambda x:int((x.timestamp() - 8*3600)*1000))

raws_data# 1714212010531 - 1714341599778

# %%
# 过滤不需要的数据
timestamp_min,timestamp_max = predict_data['timestamp'].min(), predict_data['timestamp'].max()
print(f'预测数据时间戳范围: {timestamp_min} - {timestamp_max}')

raws_data = raws_data.loc[((raws_data['timestamp'] >= timestamp_min) & (raws_data['timestamp'] <= timestamp_max)), :].reset_index(drop=True)

if not is_t0:
    raws_data = raws_data.rename(columns={'save_timestamp': "datetime"})
else:
    # 时间戳转时间格式
    raws_data['datetime'] = pd.to_datetime(raws_data['timestamp'] + 8*3600, unit='s')
raws_data

# %% [markdown]
# # 读取回测数据

# %%
trades_file = f'{symbol}_trades_' if not predict_mode else f'{symbol}_{predict_mode}_trades_'
asset_nets_file = f'{symbol}_asset_nets_' if not predict_mode else f'{symbol}_{predict_mode}_asset_nets_'
for i in os.listdir(_train_result_folder):
    if i.startswith(trades_file):
        trades_file = os.path.join(_train_result_folder, i)
    elif i.startswith(asset_nets_file):
        asset_nets_file = os.path.join(_train_result_folder, i)

print(trades_file)
print(asset_nets_file)

need_plot_sub = trades_file.endswith('.csv') and asset_nets_file.endswith('.csv')
# assert trades_file.endswith('.csv'), f'未找到交易文件'
# assert asset_nets_file.endswith('.csv'), f'未找到资产净值文件'

if need_plot_sub:
    asset_nets = pd.read_csv(asset_nets_file)
    asset_nets['timestamp'] = pd.to_datetime(asset_nets['timestamp'])
    asset_nets = asset_nets.sort_values(by='timestamp').reset_index(drop=True)
    asset_nets = asset_nets.rename(columns={'timestamp': "datetime"})
    asset_nets['asset'] = asset_nets['asset']/asset_nets.iloc[0]['asset']
    asset_nets['benchmark'] = asset_nets['benchmark']/asset_nets.iloc[0]['benchmark']

    # 计算回撤
    # 计算每个时间点的累计最大值
    cumulative_max = asset_nets['asset'].cummax()
    # 计算每个时间点的当前回撤
    drawdown = asset_nets['asset'] - cumulative_max
    # 计算回撤序列
    asset_nets['drawdown'] = drawdown / cumulative_max

    trades = pd.read_csv(trades_file)
    trades['open_deal_timestamp'] = pd.to_datetime(trades['open_deal_timestamp'])
    trades['close_deal_timestamp'] = pd.to_datetime(trades['close_deal_timestamp'])

    # 合并到 asset_nets
    asset_nets['pos'] = 0.0
    for o, c, p in zip(trades['open_deal_timestamp'], trades['close_deal_timestamp'], trades['open_deal_vol']):
        asset_nets.loc[((asset_nets['datetime'] >= o) & (asset_nets['datetime'] < c)), 'pos'] = p

    # 添加 dummy 净值数据
    if result_type != 'model_dummy':
        dummy_asset_nets_file = os.path.join(train_result_folder, 'model_dummy', os.path.basename(asset_nets_file).replace('label_', 'threshold_max_'))
        if os.path.exists(dummy_asset_nets_file):
            dummy_asset_nets = pd.read_csv(dummy_asset_nets_file)
            asset_nets['dummy_asset'] = dummy_asset_nets['asset']/dummy_asset_nets.iloc[0]['asset']

    asset_nets

# %% [markdown]
#  # 合并数据

# %%
# 生成时间序列
start_time = '1990-01-01 00:00:00'
# 设置北京时间时区
tz = pytz.timezone('Asia/Shanghai')
if not is_t0:
    raws_data['time'] = pd.date_range(start=start_time, periods=len(raws_data), freq='S',  tz=tz)
else:
    raws_data['time'] = raws_data['datetime']

raws_data['open'] = raws_data['mid']
raws_data['close'] = raws_data['mid']
raws_data['high'] = raws_data['卖1价']
raws_data['low'] = raws_data['买1价']

# 合并标签数据
raws_data = pd.merge(raws_data, predict_data, on='timestamp', how='inner').sort_values('datetime').reset_index(drop=True)
raws_data

# %%
if need_plot_sub:
    # 合并回测数据
    raws_data = pd.merge(raws_data, asset_nets, on='datetime', how='inner').sort_values('datetime').reset_index(drop=True)
    raws_data

# %%
if need_plot_sub:
    raws_data['hl1'] = 1

# %%
raws_data.set_index('datetime', inplace=True)
raws_data

# %%
import pandas as pd
import asyncio
from datetime import datetime
from lightweight_charts import Chart

plot_dummy = 'dummy_asset' in list(raws_data)

start_time = None
end_time = None
def on_arrow(key):
    global start_time, end_time
    # the amount of time to move on each press (interval * 10)
    td = pd.Timedelta(chart._interval, unit='s') * 10
    if key == 'ArrowLeft':
        start_time -= td
        end_time -= td
    elif key == 'ArrowRight':
        start_time += td
        end_time += td
    chart.set_visible_range(start_time, end_time)
    
data = None
begin_idx, end_idx = 0, -1
def on_button_press(chart):
    """
    在当前时间范围内标注 真实标签/预测标签
    """
    # 删除原有 marks
    chart.clear_markers()

    # 按照范围新增 marks
    """
    chart.marker(
        time: Optional[datetime.datetime] = None,
        position: Literal['above', 'below', 'inside'] = 'below',
        shape: Literal['arrow_up', 'arrow_down', 'circle', 'square'] = 'arrow_up',
        color: str = '#2196F3',
        text: str = '',
    ) -> str
    """
    idx = -1
    marks = []
    t0 = time.time()
    for T, P, t in zip(data['target'], data['predict'], data['time']):
        idx += 1
        if idx < begin_idx:
            continue
        elif idx > end_idx:
            # print(f'{idx} BREAK')
            break

        print(f'{t} {T} {P}', end=' ')
        if test_label:
            if (T==0):
                marks.append({'time': t, 'position': 'below', 'shape': 'arrow_up', 'color':'#CB4335', "text":''})
            elif T==1:
                marks.append({'time': t, 'position': 'above', 'shape': 'arrow_down', 'color':'#27AE60', "text":''})
            elif T==2:
                marks.append({'time': t, 'position': 'below', 'shape': 'square', 'color':'#5d6a74', "text":''})

        else:
            win = 0
            if (T==0):
                # marks.append({'time': t, 'position': 'below', 'shape': 'circle', 'color':'#641E16', "text":''})
                if P==0:
                    # 上涨
                    win = 1
                    marks.append({'time': t, 'position': 'below', 'shape': 'arrow_up', 'color':'#CB4335', "text":''})
                    # chart.marker(t, 'below', 'arrow_up', color='#CB4335')
                    print('win')
                else:
                    marks.append({'time': t, 'position': 'below', 'shape': 'circle', 'color':'#641E16', "text":''})
                    # chart.marker(t, 'below', 'circle', color='#641E16')
                    print('loss')

            elif T==1:
                # marks.append({'time': t, 'position': 'above', 'shape': 'circle', 'color':'#145A32', "text":''})
                if P==1:
                    # 下跌
                    win = 1
                    marks.append({'time': t, 'position': 'above', 'shape': 'arrow_down', 'color':'#27AE60', "text":''})
                    # chart.marker(t, 'above', 'arrow_down', color='#27AE60')
                    print('win')
                else:
                    marks.append({'time': t, 'position': 'above', 'shape': 'circle', 'color':'#145A32', "text":''})
                    # chart.marker(t, 'above', 'circle', color='#145A32')
                    print('loss')

            if win==0:
                if P==0:
                    marks.append({'time': t, 'position': 'below', 'shape': '', 'color':'#641E16', "text":'X'})
                elif P==1:
                    marks.append({'time': t, 'position': 'above', 'shape': '', 'color':'#145A32', "text":'X'})
    
    print(f'cost time {(time.time()-t0):.3f}')
    chart.marker_list(marks)
    print(f'更新标签 {begin_idx} - {end_idx} {len(marks)} {idx}')

chart2 = None
chart2s = {}
each_length = 2000 if not is_t0 else len(raws_data)
data_begin, data_end = 0, each_length
def on_left_right(chart, left=True):
    global start_time, end_time, data, data_begin, data_end
    raw_length = len(raws_data)

    if left:
        if data_begin == 0:
            return
        data_begin -= each_length
        data_end -= each_length
        data_begin = max(data_begin, 0)
        data_end = max(data_end, each_length)
    else:
        if data_end == raw_length:
            return
        data_begin += each_length
        data_end += each_length
        data_end = min(data_end, raw_length)
        data_begin = min(data_begin, raw_length - each_length)

    # 删除所有 marks
    chart.clear_markers()

    data = raws_data.iloc[data_begin:data_end]

    chart.set(data.loc[:, ['time', 'open', 'high', 'low', 'close']])
    start_time = data.iloc[-100]['time']
    end_time = data.iloc[-1]['time']

    if need_plot_sub:
        chart2s['net'].set(data.loc[:, ["time", 'asset']])
        chart2s['benchmark'].set(data.loc[:, ["time", 'benchmark']])
        chart2s['drawdown'].set(data.loc[:, ["time", 'drawdown']])
        chart2s['pos'].set(data.loc[:, ["time", 'pos']])
        chart2s['hl1'].set(data.loc[:, ["time", 'hl1']])
        if plot_dummy:
            chart2s['dummy_asset'].set(data.loc[:, ["time", 'dummy_asset']])

def on_left(chart):
    on_left_right(chart, True)

def on_right(chart):
    on_left_right(chart, False)

def on_range_change(chart, bef, aft):
    # print(f'on_range_change {bef, aft}')

    # 解析时间范围
    global begin_idx, end_idx
    bef = int(max(bef, 0))
    aft = int(max(aft, 0))
    begin_idx, end_idx = bef, len(data) -1 -aft
    # begin, end = int(chart.candle_data.iloc[begin_idx]['time']), int(chart.candle_data.iloc[end_idx]['time'])
    # begin, end = datetime.fromtimestamp(begin), datetime.fromtimestamp(end)
    # print(f'on_range_change {str(begin), str(end)}')

# 主图 标签/预测 中间价格
chart = Chart(toolbox=True,inner_width=1, inner_height=0.7 if need_plot_sub else 1, title=code.upper())
chart.time_scale(seconds_visible=True)
chart.hotkey(None, 'ArrowLeft', func=on_arrow)
chart.hotkey(None, 'ArrowRight', func=on_arrow)
chart.topbar.button('bt', 'mark', func=on_button_press)
chart.topbar.button('left', '   <   ', func=on_left)
chart.topbar.button('right', '   >   ', func=on_right)
chart.legend(True)
chart.precision(4 if not is_t0 else 3)
chart.price_scale(mode='logarithmic')
candle_coler = '#626262'
chart.candle_style(
    up_color = candle_coler, 
    down_color= candle_coler, 
    border_up_color= candle_coler, 
    border_down_color= candle_coler, 
    wick_up_color= candle_coler, 
    wick_down_color= candle_coler, 
)
chart.events.range_change += on_range_change

# data = raws_data.iloc[:160000]
# data = raws_data.iloc[:2000]
# data = raws_data.resample('S').first()
# raw_length = len(data)
# print(f'数据: {raw_length}条')
# if raw_length > 160000:
#     data = data.iloc[:160000]
data = raws_data.iloc[data_begin:data_end]
# data = raws_data.iloc[:300]

chart.set(data.loc[:, ['time', 'open', 'high', 'low', 'close']])

start_time = data.iloc[-100]['time']
end_time = data.iloc[-1]['time']

if need_plot_sub:
    # 附图 策略净值/基准净值/回撤
    chart2 = chart.create_subchart(position='below',width=1, height=0.3, sync=True)
    chart2.time_scale(seconds_visible=True)
    chart2.legend(True)
    chart2.precision(4 if not is_t0 else 3)
    chart2s['drawdown'] = chart2.create_histogram('drawdown', price_line=False, scale_margin_top=0, scale_margin_bottom=0.5, color='rgba(98, 98, 98, 0.7)')
    chart2s['pos'] = chart2.create_histogram('pos', price_line=False, scale_margin_top=0.6, scale_margin_bottom=0, color='rgba(135, 206, 250, 0.3)')
    chart2s['hl1'] = chart2.create_line('hl1', price_line=False, color='white', style='dashed')
    chart2s['net'] = chart2.create_line('asset', price_line=False, color='#CB4335', style='solid')
    chart2s['benchmark'] = chart2.create_line('benchmark', price_line=False, style='solid')
    if plot_dummy:
        chart2s['dummy_asset'] = chart2.create_line('dummy_asset', price_line=False, color='#82a8cd', style='solid')

    # chart2.horizontal_line(1, color=candle_coler, style='dotted', width=5)

    chart2s['net'].set(data.loc[:, ["time", 'asset']])
    chart2s['benchmark'].set(data.loc[:, ["time", 'benchmark']])
    chart2s['drawdown'].set(data.loc[:, ["time", 'drawdown']])
    chart2s['pos'].set(data.loc[:, ["time", 'pos']])
    chart2s['hl1'].set(data.loc[:, ["time", 'hl1']])
    if plot_dummy:
        chart2s['dummy_asset'].set(data.loc[:, ["time", 'dummy_asset']])

await chart.show_async()


