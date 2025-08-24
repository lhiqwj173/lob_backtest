# {{ AURA-X | Action: Add | Reason: 创建测试代码，使用一天的模拟数据验证系统功能 | Approval: Cunzhi(ID:1735632000) }}
"""
LOB回测系统测试脚本
使用模拟的一天数据测试系统完整功能
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 从lob_backtest包导入LOBBacktester类
from lob_backtest import LOBBacktester

def generate_sample_lob_data(start_time: datetime, duration_hours: int = 6.5, 
                           interval_seconds: int = 3) -> pd.DataFrame:
    """
    生成模拟的十档盘口数据
    
    Args:
        start_time: 开始时间
        duration_hours: 持续时间（小时）
        interval_seconds: 时间间隔（秒）
        
    Returns:
        模拟的LOB数据DataFrame
    """
    print("正在生成模拟LOB数据...")
    
    # 生成时间序列（本地时间）
    end_time = start_time + timedelta(hours=duration_hours)
    time_range = pd.date_range(start=start_time, end=end_time, freq=f'{interval_seconds}S')
    
    # 基础价格（模拟ETF价格）
    base_price = 1.350
    
    data = []
    current_price = base_price
    
    for i, timestamp in enumerate(time_range):
        # 价格随机游走
        price_change = np.random.normal(0, 0.001)  # 0.1%的波动
        current_price = max(0.1, current_price + price_change)
        
        # 生成买卖价差
        spread = np.random.uniform(0.001, 0.003)  # 0.1%-0.3%的价差
        mid_price = current_price
        
        # 生成十档数据
        row = {
            'datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': int(timestamp.timestamp())  # 使用本地时间生成Unix时间戳
        }
        
        # 卖盘（从低到高）
        for level in range(1, 11):
            ask_price = mid_price + spread/2 + (level-1) * 0.001
            ask_volume = np.random.randint(1000, 50000)
            row[f'卖{level}价'] = round(ask_price, 3)
            row[f'卖{level}量'] = ask_volume
        
        # 买盘（从高到低）
        for level in range(1, 11):
            bid_price = mid_price - spread/2 - (level-1) * 0.001
            bid_volume = np.random.randint(1000, 50000)
            row[f'买{level}价'] = round(bid_price, 3)
            row[f'买{level}量'] = bid_volume
        
        # 添加卖均和买均（可选）
        ask_prices = [row[f'卖{i}价'] for i in range(1, 11)]
        bid_prices = [row[f'买{i}价'] for i in range(1, 11)]
        row['卖均'] = np.mean(ask_prices)
        row['买均'] = np.mean(bid_prices)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"生成了 {len(df)} 条LOB记录")
    return df


def generate_sample_signal_data(start_time: datetime, duration_hours: int = 6.5,
                              signal_interval_seconds: int = 15) -> pd.DataFrame:
    """
    生成模拟的深度学习信号数据
    
    Args:
        start_time: 开始时间
        duration_hours: 持续时间（小时）
        signal_interval_seconds: 信号间隔（秒）
        
    Returns:
        模拟的信号数据DataFrame
    """
    print("正在生成模拟信号数据...")
    
    # 生成信号时间点（本地时间）
    end_time = start_time + timedelta(hours=duration_hours)
    signal_times = pd.date_range(start=start_time, end=end_time,
                                freq=f'{signal_interval_seconds}S')
    
    data = []
    current_position = 0  # 0=无持仓, 1=有持仓
    
    for timestamp in signal_times:
        # 使用本地时间生成Unix时间戳
        utc_timestamp = int(timestamp.timestamp())
        
        # 生成随机信号（模拟策略逻辑）
        # 简单的趋势跟踪策略模拟
        if np.random.random() < 0.3:  # 30%概率产生信号
            if current_position == 0:
                # 无持仓时，可能买入
                target = 0 if np.random.random() < 0.6 else 1  # 60%概率持仓
                if target == 0:
                    current_position = 1
            else:
                # 有持仓时，可能卖出
                target = 1 if np.random.random() < 0.4 else 0  # 40%概率空仓
                if target == 1:
                    current_position = 0
        else:
            # 无信号，保持当前状态
            target = 0 if current_position == 0 else 1
        
        # 生成概率（模拟模型输出）
        if target == 0:
            prob_0 = np.random.uniform(0.55, 0.85)  # 持仓概率
            prob_1 = 1 - prob_0
        else:
            prob_1 = np.random.uniform(0.55, 0.85)  # 空仓概率
            prob_0 = 1 - prob_1
        
        data.append({
            'timestamp': utc_timestamp,
            'target': float(target),
            'has_pos': float(current_position),
            '0': prob_0,
            '1': prob_1
        })
    
    df = pd.DataFrame(data)
    
    # 重命名 'target' 为 'predict' 以满足可视化工具的需求
    df.rename(columns={'target': 'predict'}, inplace=True)
    # 同时创建 'target' 列以满足加载器的需求
    df['target'] = df['predict']
    
    print(f"生成了 {len(df)} 条信号记录")
    return df


def save_sample_data(lob_data: pd.DataFrame, signal_data: pd.DataFrame,
                     symbol: str, start_timestamp: int, end_timestamp: int,
                     output_dir: str = "sample_data") -> tuple:
    """
    保存模拟数据到文件
    
    Args:
        lob_data: LOB数据
        signal_data: 信号数据
        symbol: 交易标的
        start_timestamp: 开始时间戳
        end_timestamp: 结束时间戳
        output_dir: 输出目录
        
    Returns:
        (lob_file_path, signal_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # --- LOB数据保存逻辑修改 ---
    # LOBDataLoader期望的目录结构: output_dir/YYYYMMDD/symbol/十档盘口.csv
    # 从LOB数据中获取日期用于创建目录
    date_str = pd.to_datetime(lob_data['datetime'].iloc[0]).strftime('%Y%m%d')
    lob_target_dir = os.path.join(output_dir, date_str, symbol)
    os.makedirs(lob_target_dir, exist_ok=True)
    
    # 保存LOB数据到指定结构中
    lob_file_path = os.path.join(lob_target_dir, "十档盘口.csv")
    lob_data.to_csv(lob_file_path, index=False, encoding='utf-8-sig')
    print(f"LOB数据已保存到: {lob_file_path}")

    # --- 信号数据保存逻辑保持不变 ---
    # 根据回测器要求格式化信号文件名
    signal_filename = f"{symbol}_{start_timestamp}_{end_timestamp}.csv"
    signal_file = os.path.join(output_dir, signal_filename)
    signal_data.to_csv(signal_file, index=False, encoding='utf-8')
    print(f"信号数据已保存到: {signal_file}")
    
    # 返回LOB数据的基础目录和信号文件的完整路径
    return output_dir, signal_file


def run_sample_test():
    """运行样本数据测试"""
    print("=" * 60)
    print("LOB回测系统 - 样本数据测试")
    print("=" * 60)
    
    # 设置测试时间（模拟一个交易日）
    test_date = datetime(2024, 6, 3, 9, 30, 0)  # 2024年6月3日 9:30
    test_start = test_date
    
    print(f"测试时间: {test_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 生成模拟数据
    print("\n=== 第1步: 生成模拟数据 ===")
    lob_data = generate_sample_lob_data(test_start, duration_hours=6.5, interval_seconds=3)
    signal_data = generate_sample_signal_data(test_start, duration_hours=6.5, signal_interval_seconds=15)
    
    # 2. 保存数据
    print("\n=== 第2步: 保存测试数据 ===")
    symbol = "TESTETF"
    duration_hours = 6.5
    start_timestamp = int(test_start.timestamp())
    end_timestamp = int((test_start + timedelta(hours=duration_hours)).timestamp())
    
    lob_file, signal_file = save_sample_data(lob_data, signal_data, symbol,
                                             start_timestamp, end_timestamp)
    
    # 3. 运行回测
    print("\n=== 第3步: 运行回测测试 ===")
    try:
        # 初始化回测系统
        backtester = LOBBacktester()
        
        # 动态更新配置以适配LOBBacktester
        # 注意：LOB数据路径应为目录，信号数据路径为文件
        # LOB数据路径现在是包含日期子目录的根目录
        backtester.config.set('data.lob_data_path', lob_file)
        backtester.config.set('data.signal_data_path', signal_file)
        
        # 启用Debug模式以进行诊断
        backtester.config.set('backtest.debug', True)
        
        # 运行回测，不传递任何路径或symbol参数
        results = backtester.run_backtest(
            auto_open_interactive=True
        )
        
        # 4. 显示结果
        print("\n=== 第4步: 测试结果 ===")
        if results and 'metrics' in results:
            metrics = results['metrics']
            print("回测成功完成!")
            print(f"关键指标:")
            print(f"   总收益率: {metrics.get('total_return', 0):.2%}")
            print(f"   年化收益率: {metrics.get('annual_return', 0):.2%}")
            print(f"   最大回撤: {metrics.get('max_drawdown', 0):.2%}")
            print(f"   夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   总交易次数: {metrics.get('total_trades', 0)}")
            print(f"   胜率: {metrics.get('win_rate', 0):.2%}")
            
            print(f"\n📁 结果文件已保存到 'results' 目录")
            print(f"   - TEST_ETF_asset_nets.csv: 资产净值历史")
            print(f"   - TEST_ETF_trades.csv: 交易记录")
            print(f"   - TEST_ETF_metrics.json: 性能指标")
            print(f"   - backtest_report.png: 可视化报告")
            
        else:
            print("回测失败")
            
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 测试完成 ===")
    print("您可以查看 'results' 目录中的输出文件")
    print("如需使用真实数据，请修改 main.py 中的数据路径")


def validate_system_components():
    """验证系统各组件功能"""
    print("\n=== 系统组件验证 ===")
    
    try:
        # 测试配置加载
        from lob_backtest.utils.config import BacktestConfig
        config = BacktestConfig()
        print("配置模块正常")
        
        # 测试数据加载器
        from lob_backtest.data.lob_data_loader import LOBDataLoader
        from lob_backtest.data.signal_data_loader import SignalDataLoader
        lob_loader = LOBDataLoader()
        signal_loader = SignalDataLoader()
        print("数据加载模块正常")
        
        # 测试撮合引擎
        from lob_backtest.engine.matching_engine import MatchingEngine
        from lob_backtest.engine.order_book import OrderBook
        engine = MatchingEngine()
        order_book = OrderBook()
        print("撮合引擎模块正常")
        
        # 测试分析模块
        from lob_backtest.analysis.performance_metrics import PerformanceAnalyzer
        from lob_backtest.analysis.visualization import BacktestVisualizer
        analyzer = PerformanceAnalyzer()
        visualizer = BacktestVisualizer()
        print("分析模块正常")
        
        print("所有系统组件验证通过")
        
    except Exception as e:
        print(f"组件验证失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 验证系统组件
    validate_system_components()
    
    # 运行样本测试
    run_sample_test()
