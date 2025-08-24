# {{ AURA-X | Action: Add | Reason: 创建回测器模块，将LOBBacktester类移至库目录 | Approval: Cunzhi(ID:1735632000) }}
"""
订单簿级别回测系统主类
LOB (Limit Order Book) Backtesting System Main Class
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from .utils.config import BacktestConfig
from .data.lob_data_loader import LOBDataLoader
from .data.signal_data_loader import SignalDataLoader
from .engine.matching_engine import MatchingEngine
from .analysis.performance_metrics import PerformanceAnalyzer
from .analysis.visualization import BacktestVisualizer


class LOBBacktester:
    """订单簿级别回测系统主类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化回测系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = BacktestConfig(config_path)
        self.lob_loader = LOBDataLoader()
        self.signal_loader = SignalDataLoader()
        
        # 仅初始化加载器
        print("LOB回测系统初始化完成")
    
    def run_backtest(self, auto_open_interactive: bool = False) -> dict:
        """
        运行完整回测
        
        Args:
            auto_open_interactive: 是否在回测完成后自动打开交互式窗口
            
        Returns:
            回测结果字典
        """
        # --- 在每次运行时重置引擎和分析器 ---
        self.engine = MatchingEngine(
            initial_capital=float(self.config.get('trading.initial_capital', 1000000)),
            commission_rate=float(self.config.get('trading.commission_rate', 5e-5)),
            delay_ticks=int(self.config.get('trading.delay_ticks', 0)),
        )
        self.analyzer = PerformanceAnalyzer()
        self.visualizer = BacktestVisualizer()
        self.debug_mode = self.config.get('backtest.debug', False)
        self.debug_records = []
        
        # 从配置中获取参数
        lob_data_path = self.config.get('data.lob_data_path')
        signal_data_path = self.config.get('data.signal_data_path')

        # 从信号文件名解析参数
        try:
            filename = Path(signal_data_path).stem
            parts = filename.split('_')
            stock_id, start_timestamp_str, end_timestamp_str = parts[0], parts[1], parts[2]
            start_timestamp = int(start_timestamp_str)
            end_timestamp = int(end_timestamp_str)
        except (IndexError, ValueError) as e:
            print(f"无法从信号文件名解析参数: {signal_data_path}")
            print("文件名格式应为: <stock_id>_<start_timestamp>_<end_timestamp>.csv")
            return {}

        print(f"开始回测: {stock_id}")
        print(f"LOB数据目录: {lob_data_path}")
        print(f"信号数据文件: {signal_data_path}")
        
        # 1. 加载数据
        print("\n=== 数据加载阶段 ===")
        lob_data = self._load_lob_data(lob_data_path, stock_id, start_timestamp, end_timestamp)
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
        self._execute_backtest(aligned_data, stock_id)
        
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
        self._save_results(stock_id, output_dir, metrics)
        
        # 6. 可视化
        if self.config.get('visualization.enable', True):
            print("\n=== 可视化阶段 ===")
            self.visualizer.create_comprehensive_report(
                self.engine.asset_history,
                self.engine.trades,
                metrics,
                output_dir,
                show_plot=False
            )
            
            # 如果启用交互式可视化或需要自动打开交互式窗口，则创建交互式图表
            if self.config.get('visualization.interactive', False) or auto_open_interactive:
                print("\n=== 交互式可视化阶段 ===")
                # 获取LOB数据用于交互式可视化
                lob_data = self._load_lob_data(lob_data_path, stock_id, start_timestamp, end_timestamp)
                # 获取信号数据用于交互式可视化
                signal_data = self._load_signal_data(signal_data_path)
                
                self.visualizer.create_interactive_chart(
                    self.engine.asset_history,
                    self.engine.trades,
                    lob_data,
                    signal_data,
                    stock_id,
                    output_dir
                )
        
        print(f"\n=== 回测完成 ===")
        self._print_summary(metrics)
        
        return {
            'metrics': metrics,
            'trades': self.engine.trades,
            'asset_history': self.engine.asset_history
        }
    
    def _load_lob_data(self, data_path: str, stock_id: str, start_timestamp: int, end_timestamp: int) -> pd.DataFrame:
        """加载LOB数据"""
        try:
            return self.lob_loader.load_data(data_path, stock_id, start_timestamp, end_timestamp)
        except Exception as e:
            import traceback
            print("LOB数据加载错误")
            print(f"异常堆栈信息:\n{traceback.format_exc()}")
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
            import traceback
            print("信号数据加载错误")
            print(f"异常堆栈信息:\n{traceback.format_exc()}")
            return None
    
    def _align_data(self, lob_data: pd.DataFrame, signal_data: pd.DataFrame) -> pd.DataFrame:
        """数据时间对齐"""
        try:
            # 确保数据按时间戳排序
            lob_data = lob_data.sort_values('timestamp').reset_index(drop=True)
            signal_data = signal_data.sort_values('timestamp').reset_index(drop=True)

            # 1. 将信号按 has_pos 分离
            signal_pos_0 = signal_data[signal_data['has_pos'] == 0].copy()
            signal_pos_1 = signal_data[signal_data['has_pos'] == 1].copy()

            # 2. 分别进行对齐
            aligned_df_0 = pd.merge_asof(
                lob_data,
                signal_pos_0.add_suffix('_pos0'),
                left_on='timestamp',
                right_on='timestamp_pos0',
                direction='nearest'
            )
            
            aligned_df_1 = pd.merge_asof(
                lob_data,
                signal_pos_1.add_suffix('_pos1'),
                left_on='timestamp',
                right_on='timestamp_pos1',
                direction='nearest'
            )

            # 3. 合并对齐结果
            aligned_df = pd.merge(aligned_df_0, aligned_df_1, on=list(lob_data.columns))

            # 将LOB和信号数据组织成回测循环所需的格式
            aligned_records = []
            for _, row in aligned_df.iterrows():
                lob_ts = row['timestamp']
                
                # 获取LOB快照
                lob_snapshot = self.lob_loader.get_orderbook_snapshot(lob_data, lob_ts)
                
                # 构建信号信息字典
                signal_info_pos0 = {
                    'timestamp': row.get('timestamp_pos0'),
                    'signal': row.get('signal_pos0', np.nan),
                    'target': row.get('target_pos0'),
                    'has_pos': 0,
                    'prob_hold': row.get('0_pos0'),
                    'prob_close': row.get('1_pos0'),
                    'confidence': max(row.get('0_pos0', 0), row.get('1_pos0', 0))
                }
                
                signal_info_pos1 = {
                    'timestamp': row.get('timestamp_pos1'),
                    'signal': row.get('signal_pos1', np.nan),
                    'target': row.get('target_pos1'),
                    'has_pos': 1,
                    'prob_hold': row.get('0_pos1'),
                    'prob_close': row.get('1_pos1'),
                    'confidence': max(row.get('0_pos1', 0), row.get('1_pos1', 0))
                }

                aligned_records.append({
                    'timestamp': lob_ts,
                    'lob_snapshot': lob_snapshot,
                    'signal_info_pos0': signal_info_pos0,
                    'signal_info_pos1': signal_info_pos1
                })
            
            return pd.DataFrame(aligned_records)

        except Exception as e:
            import traceback
            print("数据对齐错误")
            print(f"异常堆栈信息:\n{traceback.format_exc()}")
            return None
    
    def _execute_backtest(self, aligned_data: pd.DataFrame, symbol: str) -> None:
        """执行回测主循环"""
        total_records = len(aligned_data)
        
        rest_force_close = self.config.get('trading.rest_force_close', True)
        close_force_close = self.config.get('trading.close_force_close', True)

        for i, row in aligned_data.iterrows():
            timestamp = row['timestamp']
            lob_snapshot = row['lob_snapshot']
            signal_info_pos0 = row['signal_info_pos0']
            signal_info_pos1 = row['signal_info_pos1']

            # 更新市场数据
            self.engine.update_market_data(lob_snapshot, timestamp)

            # 强制平仓逻辑
            # datetime.utcfromtimestamp() 将时间戳视为UTC时间，
            # 避免重复应用本地时区转换。
            current_time = datetime.utcfromtimestamp(timestamp)

            # 每日中午收盘前10秒
            if rest_force_close and (current_time.hour == 11 and current_time.minute == 29 and current_time.second >= 50):
                self.engine.force_close_position(timestamp)

            # 每日下午收盘前10秒
            elif close_force_close and (current_time.hour == 14 and current_time.minute == 59 and current_time.second >= 50):
                self.engine.force_close_position(timestamp)

            # 正常处理交易信号
            else:
                has_position = self.engine.position.volume > 0
                
                if has_position:
                    # 有持仓，使用 pos=1 的信号
                    signal_info = signal_info_pos1
                else:
                    # 无持仓，使用 pos=0 的信号
                    signal_info = signal_info_pos0
                
                if not pd.isna(signal_info.get('signal')):
                    signal = int(signal_info['signal'])
                    self.engine.submit_signal(signal, signal_info, timestamp)

            # Debug 模式记录
            if self.debug_mode:
                engine_has_pos = 1 if self.engine.position.volume > 0 else 0
                chosen_signal_info = signal_info_pos1 if engine_has_pos else signal_info_pos0
                
                self.debug_records.append({
                    'datetime': datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'timestamp': timestamp,
                    'engine_has_pos': engine_has_pos,
                    'signal': chosen_signal_info.get('signal', -1),
                    'signal_has_pos': chosen_signal_info.get('has_pos', -1),
                    'ask1_price': lob_snapshot['asks'][0][0] if lob_snapshot['asks'] else 0,
                    'bid1_price': lob_snapshot['bids'][0][0] if lob_snapshot['bids'] else 0,
                })
            
            # 进度显示
            if (i + 1) % 1000 == 0 or i == total_records - 1:
                progress = (i + 1) / total_records * 100
                print(f"回测进度: {progress:.1f}% ({i+1}/{total_records})")
        
        # 结束时强制平仓
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

        # 保存Debug记录
        if self.debug_mode and self.debug_records:
            debug_df = pd.DataFrame(self.debug_records)
            debug_file = os.path.join(output_dir, f"{symbol}_debug_records.csv")
            debug_df.to_csv(debug_file, index=False)
            print(f"Debug记录已保存: {debug_file}")
    
    def _print_summary(self, metrics: dict) -> None:
        """打印回测摘要"""
        print(f"总收益率: {metrics.get('total_return', 0):.2%}")
        print(f"年化收益率: {metrics.get('annual_return', 0):.2%}")
        print(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"总交易次数: {metrics.get('total_trades', 0)}")
        print(f"胜率: {metrics.get('win_rate', 0):.2%}")
        print(f"盈亏比: {metrics.get('profit_factor', 0):.2f}")