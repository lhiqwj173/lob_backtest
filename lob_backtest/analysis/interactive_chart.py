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
            trades (List): 交易记录列表。通常是一个包含交易对象的列表，但在此模块中当前未使用。
            symbol (str): 交易标的名称。
            output_dir (str): 图表文件的输出目录（当前未使用，为未来扩展保留）。
        """
        self.data = data
        self.trades = trades
        self.symbol = symbol
        self.output_dir = output_dir
        self.chart = None
        self.chart_data = None
        self.begin_idx = 0
        self.end_idx = -1
        self.has_predict_data = 'predict' in self.data.columns and 'target' in self.data.columns

    def _prepare_data(self):
        """
        准备用于图表绘制的数据。

        检查并添加必要的列，计算中间价和OHLC数据。
        """
        # 确保 'time' 列是 Unix 时间戳
        if 'time' not in self.data.columns:
            self.data['time'] = self.data.index.astype(np.int64) // 10**9
            
        required_columns = ['asset', 'benchmark', 'drawdown', 'pos', 'hl1']
        for col in required_columns:
            if col not in self.data.columns:
                print(f"警告: 数据中缺少 '{col}' 列，将使用默认值填充。")
                self.data[col] = 0 if col != 'hl1' else 1

        if '买1价' in self.data.columns and '卖1价' in self.data.columns:
            self.data['mid'] = (self.data['买1价'] + self.data['卖1价']) / 2
        else:
            print("警告: 订单簿数据中缺少'买1价'或'卖1价'，将使用资产净值生成价格。")
            self.data['mid'] = self.data['asset'] * 100  # 假设价格是净值的100倍

        self.data['open'] = self.data['mid']
        self.data['close'] = self.data['mid']
        self.data['high'] = self.data['mid'] * 1.001
        self.data['low'] = self.data['mid'] * 0.999
        
        self.chart_data = self.data

    def _on_range_change(self, chart, bef, aft):
        """
        图表可见范围变化时的回调函数。

        用于更新当前可见数据点的索引范围。
        """
        bef = int(max(bef, 0))
        aft = int(max(aft, 0))
        self.begin_idx = bef
        self.end_idx = len(self.chart_data) - 1 - aft

    def _on_button_press(self, chart):
        """
        “标记”按钮按下的回调函数。

        在当前可见范围内，根据预测和真实标签添加标记。
        """
        chart.clear_markers()
        marks = []
        t0 = time.time()
        if self.has_predict_data:
            for _, row in self.chart_data.iloc[self.begin_idx:self.end_idx].iterrows():
                T = row['target']
                P = row['predict']
                t = row['time']

                if T == 0:  # 真实为上涨
                    if P == 0: # 预测正确
                        marks.append({'time': t, 'position': 'below', 'shape': 'arrow_up', 'color': '#CB4335', "text": ''})
                    else: # 预测错误
                        marks.append({'time': t, 'position': 'below', 'shape': 'circle', 'color': '#641E16', "text": ''})
                elif T == 1:  # 真实为下跌
                    if P == 1: # 预测正确
                        marks.append({'time': t, 'position': 'above', 'shape': 'arrow_down', 'color': '#27AE60', "text": ''})
                    else: # 预测错误
                        marks.append({'time': t, 'position': 'above', 'shape': 'circle', 'color': '#145A32', "text": ''})
                elif T == 2:  # 真实为震荡/无标签
                    marks.append({'time': t, 'position': 'below', 'shape': 'square', 'color': '#5d6a74', "text": ''})
        
        print(f'标记更新耗时 {(time.time()-t0):.3f}秒')
        chart.marker_list(marks)
        print(f'更新标签 {self.begin_idx} - {self.end_idx} 共{len(marks)}个标记')

    def show(self):
        """
        创建并显示交互式图表。
        """
        self._prepare_data()

        # 主图：价格K线和预测标记
        self.chart = Chart(toolbox=True, inner_width=1, inner_height=0.7, title=self.symbol.upper())
        self.chart.time_scale(seconds_visible=True)
        self.chart.topbar.button('bt', '标记', func=self._on_button_press)
        self.chart.legend(True)
        self.chart.price_scale(mode='logarithmic')
        self.chart.events.range_change += self._on_range_change

        if not self.chart_data.empty:
            self.chart.set(self.chart_data.loc[:, ['time', 'open', 'high', 'low', 'close']])
            
            # 设置K线颜色
            candle_color = '#626262'
            self.chart.candle_style(
                up_color=candle_color, down_color=candle_color,
                border_up_color=candle_color, border_down_color=candle_color,
                wick_up_color=candle_color, wick_down_color=candle_color,
            )

            # 设置初始可见范围为最后200个数据点
            start_idx = max(0, len(self.chart_data) - 200)
            start_time = self.chart_data.iloc[start_idx]['time']
            end_time = self.chart_data.iloc[-1]['time']
            self.chart.set_visible_range(start_time, end_time)
        else:
            print("图表数据为空，无法设置主图表。")

        # 附图：净值、回撤、持仓
        subchart = self.chart.create_subchart(position='below', width=1, height=0.3, sync=True)
        subchart.time_scale(seconds_visible=True)
        subchart.legend(True)
        subchart.precision(4)

        if not self.chart_data.empty:
            subchart.create_line('asset', price_line=False, color='#CB4335', style='solid').set(self.chart_data.loc[:, ["time", 'asset']])
            subchart.create_line('benchmark', price_line=False, style='solid').set(self.chart_data.loc[:, ["time", 'benchmark']])
            subchart.create_histogram('drawdown', price_line=False, scale_margin_top=0, scale_margin_bottom=0.5, color='rgba(98, 98, 98, 0.7)').set(self.chart_data.loc[:, ["time", 'drawdown']])
            subchart.create_histogram('pos', price_line=False, scale_margin_top=0.6, scale_margin_bottom=0, color='rgba(135, 206, 250, 0.3)').set(self.chart_data.loc[:, ["time", 'pos']])
            subchart.create_line('hl1', price_line=False, color='white', style='dashed').set(self.chart_data.loc[:, ["time", 'hl1']])
        else:
            print("图表数据为空，无法设置子图表。")

        # 运行图表
        asyncio.run(self.chart.show_async())

def plot_interactive_chart(data: pd.DataFrame, trades: List, symbol: str, output_dir: str):
    """
    便捷函数，用于创建并显示交互式图表。

    Args:
        data (pd.DataFrame): 包含所有绘图数据的DataFrame。
        trades (List): 交易记录列表。
        symbol (str): 交易标的名称。
        output_dir (str): 输出目录。
    """
    chart = InteractiveChart(data, trades, symbol, output_dir)
    chart.show()

if __name__ == '__main__':
    """
    模块独立运行时执行的示例代码。
    生成模拟数据并调用 InteractiveChart 进行可视化展示。
    """
    print("正在生成模拟数据并启动交互式图表示例...")

    # 1. 生成模拟数据
    num_points = 500
    # 创建一个从现在开始，每隔3秒的时间序列
    start_time = int(time.time())
    timestamps_unix = np.arange(start_time, start_time + num_points * 3, 3)
    timestamps = pd.to_datetime(timestamps_unix, unit='s')
    
    # 模拟价格数据
    price = 100 + np.random.randn(num_points).cumsum() * 0.1
    
    # 创建 DataFrame
    mock_data = pd.DataFrame({
        '买1价': price - 0.05,
        '卖1价': price + 0.05,
    }, index=timestamps)

    # 模拟资产和基准净值
    mock_data['asset'] = 1 + (price - 100) / 100 + np.random.rand(num_points) * 0.01
    mock_data['benchmark'] = 1 + (price - 100) / 100
    
    # 模拟回撤
    running_max = mock_data['asset'].expanding().max()
    mock_data['drawdown'] = (mock_data['asset'] - running_max) / running_max

    # 模拟持仓
    mock_data['pos'] = np.random.choice([0, 1, -1], size=num_points, p=[0.6, 0.2, 0.2]).cumsum().clip(0, 5)
    
    # 模拟预测和目标标签
    mock_data['target'] = np.random.randint(0, 3, size=num_points)
    mock_data['predict'] = mock_data['target'].copy()
    # 引入一些预测错误
    error_indices = np.random.choice(mock_data.index, size=int(num_points * 0.3), replace=False)
    mock_data.loc[error_indices, 'predict'] = np.random.randint(0, 3, size=len(error_indices))

    # 2. 准备参数
    mock_trades = []  # 示例中不关注具体交易
    symbol_name = "MOCK_STOCK"
    output_directory = "results"

    # 3. 创建并显示图表
    # 使用便捷函数
    plot_interactive_chart(
        data=mock_data,
        trades=mock_trades,
        symbol=symbol_name,
        output_dir=output_directory
    )

    print("图表已关闭。")