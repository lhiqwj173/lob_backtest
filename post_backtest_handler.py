# {{ AURA-X | Action: Add | Reason: 创建回测完成后处理脚本，自动打开交互式窗口 | Approval: Cunzhi(ID:1735632000) }}
"""
回测完成后处理脚本
用于在回测完成后自动打开交互式窗口显示结果
"""

import os
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from analysis.visualization import BacktestVisualizer


def open_interactive_visualization(asset_history_file: str, trades_file: str, 
                                 lob_data_file: str = None, signal_data_file: str = None,
                                 symbol: str = "Asset", output_dir: str = "results") -> None:
    """
    打开交互式可视化窗口
    
    Args:
        asset_history_file: 资产净值历史文件路径
        trades_file: 交易记录文件路径
        lob_data_file: 订单簿数据文件路径（可选）
        signal_data_file: 信号数据文件路径（可选）
        symbol: 标的符号
        output_dir: 输出目录
    """
    try:
        import pandas as pd
        
        # 加载资产净值历史数据
        asset_history = pd.read_csv(asset_history_file).to_dict('records')
        
        # 加载交易记录数据
        trades_df = pd.read_csv(trades_file)
        
        # 转换交易记录为对象列表
        trades = []
        for _, row in trades_df.iterrows():
            # 创建一个简单的交易对象
            trade = type('Trade', (), {
                'symbol': row.get('symbol', symbol),
                'open_order_timestamp': pd.to_datetime(row.get('open_order_timestamp')),
                'open_deal_timestamp': pd.to_datetime(row.get('open_deal_timestamp')),
                'open_order_price': row.get('open_order_price'),
                'open_order_vol': row.get('open_order_vol'),
                'open_deal_price': row.get('open_deal_price'),
                'open_deal_vol': row.get('open_deal_vol'),
                'open_fee': row.get('open_fee', 0),
                'close_order_timestamp': pd.to_datetime(row.get('close_order_timestamp')),
                'close_deal_timestamp': pd.to_datetime(row.get('close_deal_timestamp')),
                'close_order_price': row.get('close_order_price'),
                'close_order_vol': row.get('close_order_vol'),
                'close_deal_price': row.get('close_deal_price'),
                'close_deal_vol': row.get('close_deal_vol'),
                'close_fee': row.get('close_fee', 0),
                'total_fee': row.get('total_fee', 0),
                'profit': row.get('profit', 0),
                'profit_t': row.get('profit_t', 0)
            })()
            trades.append(trade)
        
        # 加载订单簿数据（如果有提供）
        order_book_data = None
        if lob_data_file and os.path.exists(lob_data_file):
            order_book_data = pd.read_csv(lob_data_file)
        
        # 加载信号数据（如果有提供）
        predict_data = None
        if signal_data_file and os.path.exists(signal_data_file):
            predict_data = pd.read_csv(signal_data_file)
        
        # 创建可视化器并打开交互式图表
        visualizer = BacktestVisualizer()
        visualizer.create_interactive_chart(
            asset_history=asset_history,
            trades=trades,
            order_book_data=order_book_data,
            predict_data=predict_data,
            symbol=symbol,
            output_dir=output_dir
        )
        
        print(f"✅ 交互式可视化窗口已打开: {symbol}")
        
    except Exception as e:
        print(f"❌ 打开交互式可视化窗口时发生错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("回测完成后处理脚本")
    
    # 这里可以添加命令行参数解析
    # 暂时使用默认值进行演示
    symbol = "TEST_ETF"
    output_dir = "results"
    
    # 定义文件路径
    asset_history_file = os.path.join(output_dir, f"{symbol}_asset_nets.csv")
    trades_file = os.path.join(output_dir, f"{symbol}_trades.csv")
    
    # 检查必要的文件是否存在
    if not os.path.exists(asset_history_file):
        print(f"❌ 资产净值历史文件不存在: {asset_history_file}")
        return
    
    if not os.path.exists(trades_file):
        print(f"❌ 交易记录文件不存在: {trades_file}")
        return
    
    print(f"正在为 {symbol} 打开交互式可视化窗口...")
    open_interactive_visualization(
        asset_history_file=asset_history_file,
        trades_file=trades_file,
        symbol=symbol,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()