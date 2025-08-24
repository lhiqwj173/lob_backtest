# {{ AURA-X | Action: Add | Reason: 创建可视化模块，基于参考代码实现图表展示 | Approval: Cunzhi(ID:1735632000) }}
"""
可视化模块
基于参考代码实现回测结果的可视化展示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional
from datetime import datetime
import os

# 添加 lightweight_charts 导入
from .interactive_chart import InteractiveChart


class BacktestVisualizer:
    """回测结果可视化器
    
    提供两种可视化方式：
    1. 静态图表：使用matplotlib生成的PNG图片报告
    2. 交互式图表：使用lightweight_charts创建的交互式网页图表
    """
    
    def __init__(self, figsize: tuple = (15, 10), style: str = 'seaborn-v0_8'):
        """
        初始化可视化器
        
        Args:
            figsize: 图表尺寸
            style: 图表样式
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # 中文字体设置
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def create_comprehensive_report(self, asset_history: List[Dict],
                                  trades: List,
                                  metrics: Dict,
                                  output_dir: str = "results",
                                  show_plot: bool = True) -> None:
        """
        创建综合回测报告
        
        Args:
            asset_history: 资产净值历史
            trades: 交易记录
            metrics: 性能指标
            output_dir: 输出目录
            show_plot: 是否显示图表
        """
        if not asset_history:
            print("无资产历史数据，跳过可视化")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换数据
        df = pd.DataFrame(asset_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # 标准化净值
        df['strategy_nav'] = df['asset'] / df['asset'].iloc[0]
        df['benchmark_nav'] = df['benchmark'] / df['benchmark'].iloc[0]
        
        # 计算回撤
        df['drawdown'] = self._calculate_drawdown(df['strategy_nav'])
        
        # 创建主图表
        fig = plt.figure(figsize=self.figsize)
        
        # 子图1: 净值曲线
        ax1 = plt.subplot(3, 2, (1, 2))
        self._plot_nav_curves(ax1, df)
        
        # 子图2: 回撤
        ax2 = plt.subplot(3, 2, (3, 4))
        self._plot_drawdown(ax2, df)
        
        # 子图3: 性能指标表
        ax3 = plt.subplot(3, 2, 5)
        self._plot_metrics_table(ax3, metrics)
        
        # 子图4: 交易分析图
        ax4 = plt.subplot(3, 2, 6)
        # 这里原本是月度收益热力图的位置，现在留空或用于其他用途
        ax4.axis('off')
        ax4.text(0.5, 0.5, '月度收益热力图\n(已移除)', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # 保存图表
        report_path = os.path.join(output_dir, 'backtest_report.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"回测报告已保存: {report_path}")
        
        # 创建交易分析图
        if trades:
            self._create_trade_analysis(trades, output_dir, show_plot)
        
        if show_plot:
            plt.show()
    
    def create_interactive_chart(self, asset_history: List[Dict],
                                trades: List,
                                order_book_data: Optional[pd.DataFrame] = None,
                                predict_data: Optional[pd.DataFrame] = None,
                                symbol: str = "Asset",
                                output_dir: str = "results") -> None:
        """
        创建交互式图表
        
        Args:
            asset_history: 资产净值历史
            trades: 交易记录
            order_book_data: 订单簿数据（可选）
            predict_data: 预测数据（可选）
            symbol: 标的符号
            output_dir: 输出目录
        """
        if not asset_history:
            print("无资产历史数据，跳过可视化")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换资产历史数据
        asset_df = pd.DataFrame(asset_history)
        # 检查 timestamp 列的类型并进行相应处理
        if not pd.api.types.is_datetime64_any_dtype(asset_df['timestamp']):
            asset_df['timestamp'] = pd.to_datetime(asset_df['timestamp'])
        asset_df = asset_df.set_index('timestamp').sort_index()
        
        # 标准化净值
        asset_df['asset'] = asset_df['asset'] / asset_df['asset'].iloc[0]
        asset_df['benchmark'] = asset_df['benchmark'] / asset_df['benchmark'].iloc[0]
        
        # 计算回撤
        asset_df['drawdown'] = self._calculate_drawdown(asset_df['asset'])
        
        # 如果有订单簿数据，则合并数据
        if order_book_data is not None:
            # 确保订单簿数据有时间戳列
            if 'timestamp' in order_book_data.columns:
                # 直接使用时间戳列创建datetime索引，不经过pd.to_datetime转换
                # 将Unix时间戳转换为无时区的datetime对象（本地时间）
                order_book_data['datetime'] = pd.to_datetime(order_book_data['timestamp'], unit='s')
                order_book_data = order_book_data.set_index('datetime').sort_index()
                # 合并数据
                # 使用 merge_asof 解决时间戳不完全对齐的问题
                merged_df = pd.merge_asof(
                    left=order_book_data.sort_index(),
                    right=asset_df.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction='nearest',  # 寻找最近的时间戳
                    tolerance=pd.Timedelta('1s')  # 设置1秒的容忍范围
                )
            else:
                print("订单簿数据缺少时间戳列，无法合并数据")
                merged_df = asset_df.copy()
        else:
            print("订单簿数据为空")
            merged_df = asset_df.copy()
        
        # 添加持仓数据
        merged_df['pos'] = 0.0
        if trades:
            # print(f"交易记录不为空，数量: {len(trades)}")
            for trade in trades:
                if trade.open_deal_timestamp and trade.close_deal_timestamp:
                    # 转换为 pandas Timestamp 对象
                    open_time = pd.Timestamp(trade.open_deal_timestamp)
                    close_time = pd.Timestamp(trade.close_deal_timestamp)
                    
                    # 在持仓期间设置持仓量
                    mask = (merged_df.index >= open_time) & (merged_df.index < close_time)
                    merged_df.loc[mask, 'pos'] = trade.open_deal_vol
        else:
            print("交易记录为空")
        
        # 如果有预测数据，则合并
        if predict_data is not None:
            # 直接使用时间戳列创建datetime索引，不经过pd.to_datetime转换
            # 将Unix时间戳转换为无时区的datetime对象（本地时间）
            predict_data['datetime'] = pd.to_datetime(predict_data['timestamp'], unit='s')
            predict_data = predict_data.set_index('datetime').sort_index()
            # 使用 merge_asof 进行合并
            merged_df = pd.merge_asof(
                left=merged_df.sort_index(),
                right=predict_data.sort_index(),
                left_index=True,
                right_index=True,
                direction='nearest',
                tolerance=pd.Timedelta('1s')
            )
        else:
            print("预测数据为空")
        
        # 添加辅助列
        merged_df['hl1'] = 1
        # 确保 'time' 列是 Unix 时间戳
        merged_df['time'] = merged_df.index.astype(np.int64) // 10**9
        
        # 创建交互式图表
        chart = InteractiveChart(
            data=merged_df,
            trades=trades,
            symbol=symbol,
            output_dir=output_dir
        )
        chart.show()
    def _plot_nav_curves(self, ax, df: pd.DataFrame) -> None:
        """绘制净值曲线"""
        ax.plot(df.index, df['strategy_nav'], label='策略净值', linewidth=2, color='#e74c3c')
        ax.plot(df.index, df['benchmark_nav'], label='基准净值', linewidth=2, color='#3498db')
        
        ax.set_title('策略净值 vs 基准净值', fontsize=14, fontweight='bold')
        ax.set_ylabel('净值', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_drawdown(self, ax, df: pd.DataFrame) -> None:
        """绘制回撤图"""
        ax.fill_between(df.index, df['drawdown'], 0, alpha=0.7, color='#e74c3c')
        ax.plot(df.index, df['drawdown'], color='#c0392b', linewidth=1)
        
        ax.set_title('策略回撤', fontsize=14, fontweight='bold')
        ax.set_ylabel('回撤比例', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 标注最大回撤
        max_dd_idx = df['drawdown'].idxmin()
        max_dd_value = df['drawdown'].min()
        ax.annotate(f'最大回撤: {max_dd_value:.2%}', 
                   xy=(max_dd_idx, max_dd_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    
    def _plot_metrics_table(self, ax, metrics: Dict) -> None:
        """绘制性能指标表"""
        ax.axis('off')
        
        # 选择关键指标
        key_metrics = [
            ('总收益率', f"{metrics.get('total_return', 0):.2%}"),
            ('年化收益率', f"{metrics.get('annual_return', 0):.2%}"),
            ('最大回撤', f"{metrics.get('max_drawdown', 0):.2%}"),
            ('夏普比率', f"{metrics.get('sharpe_ratio', 0):.3f}"),
            ('胜率', f"{metrics.get('win_rate', 0):.2%}"),
            ('总交易次数', f"{metrics.get('total_trades', 0)}"),
            ('盈亏比', f"{metrics.get('profit_factor', 0):.2f}"),
            ('波动率', f"{metrics.get('volatility', 0):.2%}")
        ]
        
        # 创建表格
        table_data = []
        for metric, value in key_metrics:
            table_data.append([metric, value])
        
        table = ax.table(cellText=table_data,
                        colLabels=['指标', '数值'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(key_metrics) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#3498db')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
        
        ax.set_title('关键性能指标', fontsize=14, fontweight='bold')
    
    def _calculate_drawdown(self, nav_series: pd.Series) -> pd.Series:
        """计算回撤序列"""
        running_max = nav_series.expanding().max()
        drawdown = (nav_series - running_max) / running_max
        return drawdown
    
    def _create_trade_analysis(self, trades: List, output_dir: str, show_plot: bool = True) -> None:
        """创建交易分析图表"""
        if not trades:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # 提取交易数据
        profits = [trade.profit for trade in trades]
        holding_periods = []
        
        for trade in trades:
            if trade.open_deal_timestamp and trade.close_deal_timestamp:
                duration = (trade.close_deal_timestamp - trade.open_deal_timestamp).total_seconds() / 3600
                holding_periods.append(duration)
        
        # 1. 盈亏分布
        ax1.hist(profits, bins=20, alpha=0.7, color='#3498db', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('交易盈亏分布', fontsize=12, fontweight='bold')
        ax1.set_xlabel('盈亏金额')
        ax1.set_ylabel('频次')
        
        # 2. 累计盈亏
        cumulative_profits = np.cumsum(profits)
        ax2.plot(range(1, len(cumulative_profits) + 1), cumulative_profits, 
                color='#e74c3c', linewidth=2)
        ax2.set_title('累计盈亏曲线', fontsize=12, fontweight='bold')
        ax2.set_xlabel('交易序号')
        ax2.set_ylabel('累计盈亏')
        ax2.grid(True, alpha=0.3)
        
        # 3. 持仓时间分布
        if holding_periods:
            ax3.hist(holding_periods, bins=15, alpha=0.7, color='#2ecc71', edgecolor='black')
            ax3.set_title('持仓时间分布', fontsize=12, fontweight='bold')
            ax3.set_xlabel('持仓时间(小时)')
            ax3.set_ylabel('频次')
        
        # 4. 月度交易统计
        trade_dates = [trade.open_deal_timestamp.strftime('%Y-%m') for trade in trades 
                      if trade.open_deal_timestamp]
        if trade_dates:
            monthly_counts = pd.Series(trade_dates).value_counts().sort_index()
            ax4.bar(range(len(monthly_counts)), monthly_counts.values, 
                   color='#f39c12', alpha=0.7)
            ax4.set_title('月度交易次数', fontsize=12, fontweight='bold')
            ax4.set_xlabel('月份')
            ax4.set_ylabel('交易次数')
            ax4.set_xticks(range(len(monthly_counts)))
            ax4.set_xticklabels(monthly_counts.index, rotation=45)
        
        plt.tight_layout()
        
        # 保存交易分析图
        trade_analysis_path = os.path.join(output_dir, 'trade_analysis.png')
        plt.savefig(trade_analysis_path, dpi=300, bbox_inches='tight')
        print(f"交易分析图已保存: {trade_analysis_path}")
        
        if show_plot:
            plt.show()
