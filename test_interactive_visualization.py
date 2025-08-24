# {{ AURA-X | Action: Add | Reason: 创建交互式可视化测试脚本 | Approval: Cunzhi(ID:1735632000) }}
"""
交互式可视化测试脚本
演示如何使用新添加的交互式可视化功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# 从lob_backtest包导入BacktestVisualizer类
from lob_backtest.analysis.visualization import BacktestVisualizer


def generate_sample_data():
    """生成示例数据用于测试"""
    # 生成时间序列
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='1min')
    
    # 生成资产净值数据
    asset_history = []
    asset_value = 1000000  # 初始资金100万
    benchmark_value = 1000000
    
    for i, date in enumerate(dates):
        # 模拟收益率
        asset_return = np.random.normal(0.0001, 0.001)  # 平均日收益率0.01%，波动率0.1%
        benchmark_return = np.random.normal(0.00005, 0.0008)  # 基准收益率略低
        
        asset_value *= (1 + asset_return)
        benchmark_value *= (1 + benchmark_return)
        
        asset_history.append({
            'timestamp': date,
            'asset': asset_value,
            'benchmark': benchmark_value
        })
    
    # 生成交易记录
    trades = []
    num_trades = 50
    trade_indices = np.random.choice(len(dates), num_trades, replace=False)
    
    for i, idx in enumerate(trade_indices):
        # 随机生成交易方向
        direction = np.random.choice([-1, 1])
        volume = np.random.randint(1, 10) * 100
        
        # 随机生成开仓和平仓时间（相差1-10小时）
        open_time = dates[idx]
        close_idx = min(idx + np.random.randint(60, 600), len(dates) - 1)  # 1-10小时后平仓
        close_time = dates[close_idx]
        
        # 随机生成价格
        open_price = 100 + np.random.normal(0, 5)
        close_price = open_price + direction * np.random.normal(2, 1)
        
        # 计算收益
        profit = (close_price - open_price) * direction * volume
        
        trades.append(type('Trade', (), {
            'symbol': 'TEST',
            'open_order_timestamp': open_time,
            'open_deal_timestamp': open_time + timedelta(minutes=1),
            'open_order_price': open_price,
            'open_order_vol': volume,
            'open_deal_price': open_price + np.random.normal(0, 0.1),
            'open_deal_vol': volume,
            'open_fee': volume * open_price * 0.00005,
            'close_order_timestamp': close_time,
            'close_deal_timestamp': close_time + timedelta(minutes=1),
            'close_order_price': close_price,
            'close_order_vol': volume,
            'close_deal_price': close_price + np.random.normal(0, 0.1),
            'close_deal_vol': volume,
            'close_fee': volume * close_price * 0.00005,
            'total_fee': volume * (open_price + close_price) * 0.00005,
            'profit': profit,
            'profit_t': profit - volume * (open_price + close_price) * 0.00005
        })())
    
    # 生成订单簿数据（简化版本）
    lob_data = pd.DataFrame({
        'timestamp': dates[::10],  # 每10分钟一个数据点
        '买1价': 100 + np.cumsum(np.random.normal(0, 0.1, len(dates[::10]))),
        '卖1价': 100.1 + np.cumsum(np.random.normal(0, 0.1, len(dates[::10]))),
    })
    
    # 生成预测数据
    predict_data = pd.DataFrame({
        'timestamp': dates[::5],  # 每5分钟一个数据点
        'target': np.random.choice([0, 1, 2], len(dates[::5])),
        'predict': np.random.choice([0, 1, 2], len(dates[::5]))
    })
    
    return asset_history, trades, lob_data, predict_data


def main():
    """主函数"""
    print("生成示例数据...")
    asset_history, trades, lob_data, predict_data = generate_sample_data()
    
    print(f"生成了 {len(asset_history)} 条资产历史数据")
    print(f"生成了 {len(trades)} 条交易记录")
    print(f"生成了 {len(lob_data)} 条订单簿数据")
    print(f"生成了 {len(predict_data)} 条预测数据")
    
    # 创建可视化器
    visualizer = BacktestVisualizer()
    
    # 创建输出目录
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n创建交互式图表...")
    # 创建交互式图表
    visualizer.create_interactive_chart(
        asset_history=asset_history,
        trades=trades,
        order_book_data=lob_data,
        predict_data=predict_data,
        symbol="TEST",
        output_dir=output_dir
    )
    
    print(f"\n交互式图表已创建完成，结果保存在 {output_dir} 目录中")


if __name__ == "__main__":
    main()