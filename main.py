# {{ AURA-X | Action: Add | Reason: 创建主入口文件，整合所有模块实现完整回测流程 | Approval: Cunzhi(ID:1735632000) }}
"""
订单簿级别撮合回测系统主入口
LOB (Limit Order Book) Backtesting System Main Entry
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config import BacktestConfig
from data.lob_data_loader import LOBDataLoader
from data.signal_data_loader import SignalDataLoader
from engine.matching_engine import MatchingEngine
from analysis.performance_metrics import PerformanceAnalyzer
from analysis.visualization import BacktestVisualizer


class LOBBacktester:
    """订单簿级别回测系统主类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化回测系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = BacktestConfig(config_path)
        self.lob_loader = LOBDataLoader(self.config.get('data.timezone', 'Asia/Shanghai'))
        self.signal_loader = SignalDataLoader(self.config.get('data.timezone', 'Asia/Shanghai'))
        
        # {{ AURA-X | Action: Modify | Reason: 修复配置值类型转换问题 | Approval: Cunzhi(ID:1735632000) }}
        # 初始化撮合引擎（确保数值类型正确）
        self.engine = MatchingEngine(
            initial_capital=float(self.config.get('trading.initial_capital', 1000000)),
            commission_rate=float(self.config.get('trading.commission_rate', 5e-5)),
            delay_ticks=int(self.config.get('trading.delay_ticks', 0)),
            timezone=self.config.get('data.timezone', 'Asia/Shanghai')
        )
        
        # 分析器
        self.analyzer = PerformanceAnalyzer()
        self.visualizer = BacktestVisualizer()
        
        print("LOB回测系统初始化完成")
    
    def run_backtest(self, lob_data_path: str, signal_data_path: str, 
                    symbol: str = "UNKNOWN") -> dict:
        """
        运行完整回测
        
        Args:
            lob_data_path: 十档盘口数据路径
            signal_data_path: 信号数据路径
            symbol: 交易标的代码
            
        Returns:
            回测结果字典
        """
        print(f"开始回测: {symbol}")
        print(f"LOB数据: {lob_data_path}")
        print(f"信号数据: {signal_data_path}")
        
        # 1. 加载数据
        print("\n=== 数据加载阶段 ===")
        lob_data = self._load_lob_data(lob_data_path)
        if lob_data is None:
            print("LOB数据加载失败")
            return {}
        
        signal_data = self._load_signal_data(signal_data_path)
        if signal_data is None:
            print("信号数据加载失败")
            return {}
        
        print(f"LOB数据: {len(lob_data)} 条记录")
        print(f"信号数据: {len(signal_data)} 条记录")
        
        # 2. 数据对齐
        print("\n=== 数据对齐阶段 ===")
        aligned_data = self._align_data(lob_data, signal_data)
        if aligned_data is None:
            print("数据对齐失败")
            return {}
        
        print(f"对齐后数据: {len(aligned_data)} 条记录")
        
        # 3. 执行回测
        print("\n=== 回测执行阶段 ===")
        self._execute_backtest(aligned_data, symbol)
        
        # 4. 性能分析
        print("\n=== 性能分析阶段 ===")
        metrics = self.analyzer.calculate_all_metrics(
            self.engine.asset_history,
            self.engine.trades,
            self.engine.initial_capital
        )
        
        # 5. 结果输出
        print("\n=== 结果输出阶段 ===")
        output_dir = self.config.get('analysis.output_dir', 'results')
        self._save_results(symbol, output_dir, metrics)
        
        # 6. 可视化
        if self.config.get('visualization.enable', True):
            print("\n=== 可视化阶段 ===")
            self.visualizer.create_comprehensive_report(
                self.engine.asset_history,
                self.engine.trades,
                metrics,
                output_dir
            )
        
        print(f"\n=== 回测完成 ===")
        self._print_summary(metrics)
        
        return {
            'metrics': metrics,
            'trades': self.engine.trades,
            'asset_history': self.engine.asset_history
        }
    
    def _load_lob_data(self, file_path: str) -> pd.DataFrame:
        """加载LOB数据"""
        try:
            return self.lob_loader.load_data(file_path)
        except Exception as e:
            print(f"LOB数据加载错误: {e}")
            return None
    
    def _load_signal_data(self, file_path: str) -> pd.DataFrame:
        """加载信号数据"""
        try:
            data = self.signal_loader.load_signal_data(file_path)
            if data is not None:
                # 生成交易信号
                data = self.signal_loader.generate_signals(
                    data,
                    source=self.config.get('signal.source', 'predict'),
                    threshold_strategy=self.config.get('signal.predict_threshold', 'max')
                )
            return data
        except Exception as e:
            print(f"信号数据加载错误: {e}")
            return None
    
    def _align_data(self, lob_data: pd.DataFrame, signal_data: pd.DataFrame) -> pd.DataFrame:
        """数据时间对齐"""
        try:
            # 找到时间交集
            lob_timestamps = set(lob_data['timestamp'])
            signal_timestamps = set(signal_data['timestamp'])
            
            # 使用LOB数据的时间戳为基准，寻找最近的信号
            aligned_records = []
            
            for _, lob_row in lob_data.iterrows():
                lob_ts = lob_row['timestamp']
                
                # 找到最近的信号
                signal_info = self.signal_loader.get_signal_at_timestamp(signal_data, lob_ts)
                
                if signal_info:
                    # 获取LOB快照
                    lob_snapshot = self.lob_loader.get_orderbook_snapshot(lob_data, lob_ts)
                    
                    aligned_records.append({
                        'timestamp': lob_ts,
                        'lob_snapshot': lob_snapshot,
                        'signal_info': signal_info
                    })
            
            return pd.DataFrame(aligned_records)
            
        except Exception as e:
            print(f"数据对齐错误: {e}")
            return None
    
    def _execute_backtest(self, aligned_data: pd.DataFrame, symbol: str) -> None:
        """执行回测主循环"""
        total_records = len(aligned_data)
        
        for i, row in aligned_data.iterrows():
            timestamp = row['timestamp']
            lob_snapshot = row['lob_snapshot']
            signal_info = row['signal_info']
            
            # 更新市场数据
            self.engine.update_market_data(lob_snapshot, timestamp)
            
            # 处理交易信号
            if not pd.isna(signal_info['signal']):
                signal = int(signal_info['signal'])
                self.engine.submit_signal(signal, signal_info, timestamp)
            
            # 进度显示
            if (i + 1) % 1000 == 0 or i == total_records - 1:
                progress = (i + 1) / total_records * 100
                print(f"回测进度: {progress:.1f}% ({i+1}/{total_records})")
        
        # 强制平仓
        if aligned_data.iloc[-1]['timestamp']:
            self.engine.force_close_position(aligned_data.iloc[-1]['timestamp'])
        
        # 设置交易记录的symbol
        for trade in self.engine.trades:
            trade.symbol = symbol
    
    def _save_results(self, symbol: str, output_dir: str, metrics: dict) -> None:
        """保存回测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存资产净值历史
        if self.engine.asset_history:
            asset_df = pd.DataFrame(self.engine.asset_history)
            asset_file = os.path.join(output_dir, f"{symbol}_asset_nets.csv")
            asset_df.to_csv(asset_file, index=False)
            print(f"资产净值已保存: {asset_file}")
        
        # 保存交易记录
        if self.engine.trades:
            trades_data = []
            for trade in self.engine.trades:
                trades_data.append({
                    'symbol': trade.symbol,
                    'open_order_timestamp': trade.open_order_timestamp,
                    'open_deal_timestamp': trade.open_deal_timestamp,
                    'open_order_price': trade.open_order_price,
                    'open_order_vol': trade.open_order_vol,
                    'open_deal_price': trade.open_deal_price,
                    'open_deal_vol': trade.open_deal_vol,
                    'open_fee': trade.open_fee,
                    'close_order_timestamp': trade.close_order_timestamp,
                    'close_deal_timestamp': trade.close_deal_timestamp,
                    'close_order_price': trade.close_order_price,
                    'close_order_vol': trade.close_order_vol,
                    'close_deal_price': trade.close_deal_price,
                    'close_deal_vol': trade.close_deal_vol,
                    'close_fee': trade.close_fee,
                    'total_fee': trade.total_fee,
                    'profit': trade.profit,
                    'profit_t': trade.profit_t
                })
            
            trades_df = pd.DataFrame(trades_data)
            trades_file = os.path.join(output_dir, f"{symbol}_trades.csv")
            trades_df.to_csv(trades_file, index=False)
            print(f"交易记录已保存: {trades_file}")
        
        # 保存性能指标
        metrics_file = os.path.join(output_dir, f"{symbol}_metrics.json")
        import json
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"性能指标已保存: {metrics_file}")
    
    def _print_summary(self, metrics: dict) -> None:
        """打印回测摘要"""
        print(f"总收益率: {metrics.get('total_return', 0):.2%}")
        print(f"年化收益率: {metrics.get('annual_return', 0):.2%}")
        print(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"总交易次数: {metrics.get('total_trades', 0)}")
        print(f"胜率: {metrics.get('win_rate', 0):.2%}")
        print(f"盈亏比: {metrics.get('profit_factor', 0):.2f}")


def main():
    """主函数"""
    # 示例用法
    backtester = LOBBacktester()
    
    # 这里需要用户提供实际的数据路径
    lob_data_path = "path/to/lob_data.csv"  # 十档盘口数据
    signal_data_path = "path/to/signal_data.csv"  # 信号数据
    symbol = "EXAMPLE"
    
    print("请在代码中设置正确的数据路径后运行")
    print(f"当前LOB数据路径: {lob_data_path}")
    print(f"当前信号数据路径: {signal_data_path}")
    
    # 取消注释以下行来运行回测
    # results = backtester.run_backtest(lob_data_path, signal_data_path, symbol)


if __name__ == "__main__":
    main()
