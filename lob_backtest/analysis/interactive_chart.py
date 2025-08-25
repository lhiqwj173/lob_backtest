# -*- coding: utf-8 -*-
"""
交互式图表模块

本模块封装了基于 lightweight_charts 的交互式图表生成功能。
可以将回测过程中的资产、交易、订单簿和预测数据进行可视化展示。
"""

import pandas as pd
import numpy as np
from lightweight_charts import Chart
import asyncio
import time
from typing import List, Dict, Optional


class InteractiveChart:
    """
    交互式图表生成器

    使用 lightweight-charts 库创建可交互的回测结果图表。
    """

    def __init__(self, data: pd.DataFrame, trades: List, symbol: str, output_dir: str):
        """
        初始化交互式图表

        Args:
            data (pd.DataFrame): 包含所有需要绘图的数据的DataFrame。
                                 索引应为 pd.DatetimeIndex。
                                 必须包含以下列:
                                     - 'asset' (float): 策略净值。
                                     - 'benchmark' (float): 基准净值。
                                 可选列:
                                     - '买1价' (float), '卖1价' (float): 用于计算中间价和OHLC。
                                     - 'predict' (int): 模型的预测标签 (例如 0, 1, 2)。
                                     - 'target' (int): 真实的标签。
                                     - 'drawdown' (float): 回撤值。
                                     - 'pos' (float): 持仓量。
                                     - 'hl1' (float): 水平基准线，通常设置为1，用于标识初始净值水平。
            trades (List): 交易记录列表。通常是一个包含交易对象的列表，但在此模块中当前未使用。
            symbol (str): 交易标的名称。
            output_dir (str): 图表文件的输出目录（当前未使用，为未来扩展保留）。
        """
        self.data = data.copy()
        self.trades = trades
        self.symbol = symbol
        self.output_dir = output_dir
        self.chart = None
        self.subchart = None
        self.chart_data = None
        self.begin_idx = 0
        self.end_idx = -1
        self.has_predict_data = 'predict' in self.data.columns and 'target' in self.data.columns
        
        # 日期切换逻辑
        self.unique_dates = sorted(self.data.index.normalize().unique())
        self.current_date_index = len(self.unique_dates) - 1 if self.unique_dates else 0

        # 用于存储图表系列对象
        self.asset_line = None
        self.benchmark_line = None
        self.drawdown_hist = None
        self.pos_hist = None
        self.hl1_line = None


    def _prepare_data(self):
        """
        准备用于图表绘制的数据。
        """
        required_columns = ['asset', 'benchmark', 'drawdown', 'pos']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"错误: 数据中缺少必需的列 '{col}'。")

        # 精度控制
        self.data['drawdown'] = self.data['drawdown'].round(4)

        if '买1价' in self.data.columns and '卖1价' in self.data.columns:
            self.data['mid'] = (self.data['买1价'] + self.data['卖1价']) / 2
        else:
            print("警告: 订单簿数据中缺少'买1价'或'卖1价'，将使用资产净值生成价格。")
            self.data['mid'] = self.data['asset'] * 100

        self.data['open'] = self.data['mid']
        self.data['close'] = self.data['mid']
        self.data['high'] = self.data['mid'] * 1.001
        self.data['low'] = self.data['mid'] * 0.999

    def _on_range_change(self, chart, bef, aft):
        bef = int(max(bef, 0))
        aft = int(max(aft, 0))
        self.begin_idx = bef
        self.end_idx = len(self.chart_data) - 1 - aft

    def _on_button_press(self, chart):
        chart.clear_markers()
        marks = []
        t0 = time.time()
        if self.has_predict_data and self.chart_data is not None:
            data_to_mark = self.chart_data.iloc[self.begin_idx:self.end_idx].reset_index()
            for _, row in data_to_mark.iterrows():
                T = row['target']
                P = row['predict']
                t = row['time']

                if T == 0:
                    if P == 0: marks.append({'time': t, 'position': 'below', 'shape': 'arrow_up', 'color': '#CB4335', "text": ''})
                    else: marks.append({'time': t, 'position': 'below', 'shape': 'circle', 'color': '#641E16', "text": ''})
                elif T == 1:
                    if P == 1: marks.append({'time': t, 'position': 'above', 'shape': 'arrow_down', 'color': '#27AE60', "text": ''})
                    else: marks.append({'time': t, 'position': 'above', 'shape': 'circle', 'color': '#145A32', "text": ''})
                elif T == 2:
                    marks.append({'time': t, 'position': 'below', 'shape': 'square', 'color': '#5d6a74', "text": ''})
        
        print(f'标记更新耗时 {(time.time()-t0):.3f}秒')
        chart.marker_list(marks)
        print(f'更新标签 {self.begin_idx} - {self.end_idx} 共{len(marks)}个标记')

    def _update_charts(self):
        if not self.unique_dates:
            print("错误: 数据中没有有效的日期。")
            return
            
        target_date = self.unique_dates[self.current_date_index]
        # 使用 .copy() 避免 SettingWithCopyWarning
        daily_data = self.data[self.data.index.normalize() == target_date].copy()
        
        self.chart.clear_markers()

        if not daily_data.empty:
            # --- 指标校正 ---
            # 1. 校正净值：将当天的第一个值修正到1
            first_asset = daily_data['asset'].iloc[0]
            if first_asset != 0:
                daily_data['asset'] = daily_data['asset'] / first_asset
            
            first_benchmark = daily_data['benchmark'].iloc[0]
            if first_benchmark != 0:
                daily_data['benchmark'] = daily_data['benchmark'] / first_benchmark

            # 2. 校正回撤：专注在当前日期
            running_max = daily_data['asset'].expanding().max()
            daily_data['drawdown'] = ((daily_data['asset'] - running_max) / running_max).round(4)
            # --- 校正结束 ---

            self.chart_data = daily_data
            df_for_chart = self.chart_data.reset_index().rename(columns={'index': 'time'})
            
            if 'hl1' not in df_for_chart.columns:
                df_for_chart['hl1'] = 1
            
            self.chart.set(df_for_chart.loc[:, ['time', 'open', 'high', 'low', 'close']])
            
            self.asset_line.set(df_for_chart.loc[:, ["time", 'asset']])
            self.benchmark_line.set(df_for_chart.loc[:, ["time", 'benchmark']])
            self.drawdown_hist.set(df_for_chart.loc[:, ["time", 'drawdown']])
            self.pos_hist.set(df_for_chart.loc[:, ["time", 'pos']])
            self.hl1_line.set(df_for_chart.loc[:, ["time", 'hl1']])
            
            start_time = df_for_chart.iloc[0]['time']
            end_time = df_for_chart.iloc[-1]['time']
            self.chart.set_visible_range(start_time, end_time)
        else:
            print(f"日期 {target_date.date()} 没有数据。")
            self.chart_data = pd.DataFrame() # 清空数据
            # 通过设置空数据来清除所有系列
            self.chart.set(pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close']))
            self.asset_line.set(pd.DataFrame(columns=['time', 'asset']))
            self.benchmark_line.set(pd.DataFrame(columns=['time', 'benchmark']))
            self.drawdown_hist.set(pd.DataFrame(columns=['time', 'drawdown']))
            self.pos_hist.set(pd.DataFrame(columns=['time', 'pos']))
            self.hl1_line.set(pd.DataFrame(columns=['time', 'hl1']))

    def _on_left_right(self, chart, left=True):
        if left:
            if self.current_date_index == 0:
                print("已经是第一个交易日。")
                return
            self.current_date_index -= 1
        else:
            if self.current_date_index >= len(self.unique_dates) - 1:
                print("已经是最后一个交易日。")
                return
            self.current_date_index += 1
        
        new_date = self.unique_dates[self.current_date_index].date()
        print(f"图表数据更新: 切换到日期 {new_date}")
        self._update_charts()

    def _on_left(self, chart): self._on_left_right(chart, left=True)
    def _on_right(self, chart): self._on_left_right(chart, left=False)

    def show(self, initial_render=True):
        if initial_render:
            self._prepare_data()
        
        if not self.unique_dates:
            print("错误: 数据为空或不包含有效日期，无法显示图表。")
            return
            
        initial_date = self.unique_dates[self.current_date_index]
        self.chart_data = self.data[self.data.index.normalize() == initial_date].copy()
        
        # --- 指标校正 ---
        if not self.chart_data.empty:
            # 1. 校正净值：将当天的第一个值修正到1
            first_asset = self.chart_data['asset'].iloc[0]
            if first_asset != 0:
                self.chart_data['asset'] = self.chart_data['asset'] / first_asset
            
            first_benchmark = self.chart_data['benchmark'].iloc[0]
            if first_benchmark != 0:
                self.chart_data['benchmark'] = self.chart_data['benchmark'] / first_benchmark

            # 2. 校正回撤：专注在当前日期
            running_max = self.chart_data['asset'].expanding().max()
            self.chart_data['drawdown'] = ((self.chart_data['asset'] - running_max) / running_max).round(4)
        # --- 校正结束 ---

        self.chart = Chart(toolbox=True, inner_width=1, inner_height=0.7, title=self.symbol.upper())
        self.chart.precision(4)
        self.chart.time_scale(seconds_visible=True)
        self.chart.topbar.button('bt', '标记', func=self._on_button_press)
        self.chart.topbar.button('left', '前一天', func=self._on_left)
        self.chart.topbar.button('right', '后一天', func=self._on_right)
        self.chart.legend(True, percent=False)
        self.chart.price_scale(mode='normal')
        self.chart.events.range_change += self._on_range_change

        df_for_chart = self.chart_data.reset_index().rename(columns={'index': 'time'})

        if not df_for_chart.empty:
            candles = self.chart.set(df_for_chart.loc[:, ['time', 'open', 'high', 'low', 'close']])
            if candles:
                candles.precision(4)
            
            candle_color = '#626262'
            self.chart.candle_style(up_color=candle_color, down_color=candle_color, border_up_color=candle_color, border_down_color=candle_color, wick_up_color=candle_color, wick_down_color=candle_color)

            start_time = df_for_chart.iloc[0]['time']
            end_time = df_for_chart.iloc[-1]['time']
            self.chart.set_visible_range(start_time, end_time)
        else:
            print(f"初始日期 {initial_date.date()} 数据为空。")

        self.subchart = self.chart.create_subchart(position='below', width=1, height=0.3, sync=True)
        self.subchart.precision(4)
        self.subchart.time_scale(seconds_visible=True)
        self.subchart.legend(True)
        
        # 无论初始数据是否为空，都创建系列对象
        self.drawdown_hist = self.subchart.create_histogram('drawdown', price_line=False, scale_margin_top=0, scale_margin_bottom=0.5, color='rgba(98, 98, 98, 0.7)')
        self.drawdown_hist.precision(4)
        self.pos_hist = self.subchart.create_histogram('pos', price_line=False, scale_margin_top=0.6, scale_margin_bottom=0, color='rgba(135, 206, 250, 0.3)')
        self.hl1_line = self.subchart.create_line('hl1', price_line=False, color='white', style='dashed')
        self.hl1_line.precision(4)
        self.asset_line = self.subchart.create_line('asset', price_line=False, color='#CB4335', style='solid')
        self.asset_line.precision(4)
        self.benchmark_line = self.subchart.create_line('benchmark', price_line=False, style='solid')
        self.benchmark_line.precision(4)

        if not df_for_chart.empty:
            if 'hl1' not in df_for_chart.columns:
                df_for_chart['hl1'] = 1

            self.asset_line.set(df_for_chart.loc[:, ["time", 'asset']])
            self.benchmark_line.set(df_for_chart.loc[:, ["time", 'benchmark']])
            self.drawdown_hist.set(df_for_chart.loc[:, ["time", 'drawdown']])
            self.pos_hist.set(df_for_chart.loc[:, ["time", 'pos']])
            self.hl1_line.set(df_for_chart.loc[:, ["time", 'hl1']])
        else:
            print("初始图表数据为空，子图表将为空。")

        self.chart.show(block=True)

def plot_interactive_chart(data: pd.DataFrame, trades: List, symbol: str, output_dir: str):
    chart = InteractiveChart(data, trades, symbol, output_dir)
    chart.show()

if __name__ == '__main__':
    print("正在生成模拟数据并启动交互式图表示例...")

    # 生成跨越多天的数据
    num_days = 3
    points_per_day = 1000
    num_points = num_days * points_per_day
    
    start_time = pd.Timestamp('2023-01-01 09:30:00')
    time_deltas = np.concatenate([
        np.arange(points_per_day) * 3 + i * 86400 for i in range(num_days)
    ])
    timestamps = pd.to_datetime(start_time.value + time_deltas * 10**9)
    
    price = 100 + np.random.randn(num_points).cumsum() * 0.1
    
    mock_data = pd.DataFrame({'买1价': price - 0.05, '卖1价': price + 0.05}, index=timestamps)
    mock_data.index.name = 'time'

    mock_data['asset'] = 1 + (price - 100) / 100 + np.random.rand(num_points) * 0.01
    mock_data['benchmark'] = 1 + (price - 100) / 100
    
    running_max = mock_data['asset'].expanding().max()
    mock_data['drawdown'] = (mock_data['asset'] - running_max) / running_max

    mock_data['pos'] = np.random.choice([0, 1, -1], size=num_points, p=[0.6, 0.2, 0.2]).cumsum().clip(0, 5)
    
    mock_data['target'] = np.random.randint(0, 3, size=num_points)
    mock_data['predict'] = mock_data['target'].copy()
    error_indices = np.random.choice(mock_data.index, size=int(num_points * 0.3), replace=False)
    mock_data.loc[error_indices, 'predict'] = np.random.randint(0, 3, size=len(error_indices))

    plot_interactive_chart(data=mock_data, trades=[], symbol="MOCK_STOCK", output_dir="results")

    print("图表已关闭。")