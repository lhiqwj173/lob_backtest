# {{ AURA-X | Action: Add | Reason: 创建撮合引擎核心，实现高性能回测逻辑 | Approval: Cunzhi(ID:1735632000) }}
"""
撮合引擎核心模块
实现订单簿级别的撮合回测，支持全仓模式和延迟处理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pytz

from .order_book import OrderBook


@dataclass
class Position:
    """持仓信息"""
    symbol: str = ""
    volume: float = 0.0  # 持仓数量
    avg_price: float = 0.0  # 平均成本价
    open_time: Optional[datetime] = None
    total_cost: float = 0.0  # 总成本


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    open_order_timestamp: datetime
    open_deal_timestamp: datetime
    open_order_price: float  # 市价单用极大值表示
    open_order_vol: float
    open_deal_price: float
    open_deal_vol: float
    open_fee: float
    close_order_timestamp: Optional[datetime] = None
    close_deal_timestamp: Optional[datetime] = None
    close_order_price: float = 0.0
    close_order_vol: float = 0.0
    close_deal_price: float = 0.0
    close_deal_vol: float = 0.0
    close_fee: float = 0.0
    total_fee: float = 0.0
    profit: float = 0.0
    profit_t: float = 0.0  # 累计利润


class MatchingEngine:
    """撮合引擎"""
    
    def __init__(self, initial_capital: float = 1000000.0,
                 commission_rate: float = 5e-5,
                 delay_ticks: int = 0):
        """
        初始化撮合引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            delay_ticks: 延迟tick数
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.delay_ticks = delay_ticks
        
        # 交易状态
        self.position = Position()
        self.order_book = OrderBook()
        self.pending_orders = []  # 延迟订单队列
        
        # 记录
        self.trades: List[Trade] = []
        self.asset_history: List[Dict] = []
        self.current_trade: Optional[Trade] = None
        
        # 统计
        self.total_trades = 0
        self.winning_trades = 0
        self.total_commission = 0.0
    
    def update_market_data(self, lob_snapshot: Dict, timestamp: int) -> None:
        """
        更新市场数据
        
        Args:
            lob_snapshot: 订单簿快照
            timestamp: 时间戳
        """
        # 更新订单簿
        self.order_book.update_snapshot(lob_snapshot)
        
        # 处理延迟订单
        self._process_pending_orders(timestamp)
        
        # 更新资产净值
        self._update_asset_value(timestamp)
    
    def submit_signal(self, signal: int, signal_info: Dict, timestamp: int) -> bool:
        """
        提交交易信号
        
        Args:
            signal: 交易信号 (0=持仓, 1=空仓)
            signal_info: 信号详细信息
            timestamp: 信号时间戳
            
        Returns:
            是否成功提交订单
        """
        # datetime.fromtimestamp() 会隐式使用本地系统时区（北京时间），
        # 这符合将Unix时间戳转换为本地时间的要求。
        current_time = datetime.fromtimestamp(timestamp)
        
        # 检查信号有效性
        if not self._is_valid_signal(signal, signal_info):
            return False
        
        # 根据当前持仓状态和信号决定动作
        if self.position.volume == 0 and signal == 0:
            # 无持仓且信号为持仓 -> 买入
            return self._submit_buy_order(timestamp, current_time)
        elif self.position.volume > 0 and signal == 1:
            # 有持仓且信号为空仓 -> 卖出
            return self._submit_sell_order(timestamp, current_time)
        
        return False
    
    def _is_valid_signal(self, signal: int, signal_info: Dict) -> bool:
        """验证信号有效性"""
        # 检查信号值
        if signal not in [0, 1]:
            return False
        
        # 检查市场数据有效性
        if not self.order_book.get_best_bid() or not self.order_book.get_best_ask():
            return False
        
        return True

    def _submit_buy_order(self, timestamp: int, current_time: datetime) -> bool:
        """提交买入订单（全仓模式）"""
        # 计算可买入数量（全仓）
        best_ask = self.order_book.get_best_ask()
        if not best_ask:
            return False

        # 全仓买入：使用所有可用资金
        available_capital = self.current_capital
        estimated_volume = available_capital / best_ask[0]  # 粗略估算

        if estimated_volume <= 0:
            return False

        # 创建订单
        order = {
            'type': 'buy',
            'volume': estimated_volume,
            'timestamp': timestamp,
            'order_time': current_time,
            'ticks_since_submit': 0
        }

        self.pending_orders.append(order)
        return True

    def _submit_sell_order(self, timestamp: int, current_time: datetime) -> bool:
        """提交卖出订单（全部持仓）"""
        if self.position.volume <= 0:
            return False

        # 创建订单
        order = {
            'type': 'sell',
            'volume': self.position.volume,
            'timestamp': timestamp,
            'order_time': current_time,
            'ticks_since_submit': 0
        }

        self.pending_orders.append(order)
        return True

    def _process_pending_orders(self, current_timestamp: int) -> None:
        """处理延迟订单队列"""
        # {{ AURA-X | Action: Modify | Reason: 使用基于tick的延迟机制，更精确地模拟延迟 | Approval: Cunzhi(ID:1735632000) }}
        if not self.pending_orders:
            return

        executable_orders = []
        remaining_orders = []

        # 增加所有待处理订单的tick计数器
        for order in self.pending_orders:
            order['ticks_since_submit'] = order.get('ticks_since_submit', 0) + 1
            # delay_ticks=0 也默认存在一个tick 的延迟
            if order['ticks_since_submit'] > self.delay_ticks:
                executable_orders.append(order)
            else:
                remaining_orders.append(order)
        
        self.pending_orders = remaining_orders

        # 执行达到延迟条件的订单
        for order in executable_orders:
            if order['type'] == 'buy':
                self._execute_buy_order(order, current_timestamp)
            elif order['type'] == 'sell':
                self._execute_sell_order(order, current_timestamp)

    def _execute_buy_order(self, order: Dict, execution_timestamp: int) -> None:
        """执行买入订单"""
        # 重新计算可买入数量（考虑资金变化）
        available_capital = self.current_capital

        # 执行市价买入
        result = self.order_book.execute_market_buy(order['volume'])

        if not result['success'] or result['filled_volume'] <= 0:
            return

        # {{ AURA-X | Action: Modify | Reason: 简化类型转换，移除调试信息 | Approval: Cunzhi(ID:1735632000) }}
        # 实际成交金额（确保为数值类型）
        actual_cost = float(result['total_cost'])
        actual_volume = float(result['filled_volume'])
        avg_price = float(result['avg_price'])

        # 计算手续费
        commission = actual_cost * self.commission_rate
        total_cost = actual_cost + commission

        # 检查资金是否足够
        if total_cost > available_capital:
            # 资金不足，按比例调整
            scale = available_capital / total_cost
            actual_volume *= scale
            actual_cost *= scale
            commission *= scale
            total_cost = available_capital

        # 更新持仓
        self.position.volume = actual_volume
        self.position.avg_price = avg_price
        self.position.open_time = order['order_time']
        self.position.total_cost = actual_cost

        # 更新资金
        self.current_capital -= total_cost
        self.total_commission += commission

        # 创建交易记录
        deal_time = datetime.fromtimestamp(execution_timestamp)

        self.current_trade = Trade(
            symbol="",  # 将在外部设置
            open_order_timestamp=order['order_time'],
            open_deal_timestamp=deal_time,
            open_order_price=1.79769e+308,  # 市价单标记
            open_order_vol=actual_volume,
            open_deal_price=avg_price,
            open_deal_vol=actual_volume,
            open_fee=commission
        )

        self.total_trades += 1

    def _execute_sell_order(self, order: Dict, execution_timestamp: int) -> None:
        """执行卖出订单"""
        if self.position.volume <= 0 or not self.current_trade:
            return

        # 执行市价卖出
        result = self.order_book.execute_market_sell(order['volume'])

        if not result['success'] or result['filled_volume'] <= 0:
            return

        # {{ AURA-X | Action: Modify | Reason: 修复类型错误，确保数值类型正确 | Approval: Cunzhi(ID:1735632000) }}
        # 实际成交（确保为数值类型）
        actual_volume = float(result['filled_volume'])
        avg_price = float(result['avg_price'])
        actual_proceeds = float(result['total_cost'])

        # 计算手续费
        commission = actual_proceeds * self.commission_rate
        net_proceeds = actual_proceeds - commission

        # 计算盈亏
        cost_basis = self.position.avg_price * actual_volume
        profit = net_proceeds - cost_basis

        # 更新资金
        self.current_capital += net_proceeds
        self.total_commission += commission

        # 完成交易记录
        deal_time = datetime.fromtimestamp(execution_timestamp)

        self.current_trade.close_order_timestamp = order['order_time']
        self.current_trade.close_deal_timestamp = deal_time
        self.current_trade.close_order_price = 0.0  # 市价单
        self.current_trade.close_order_vol = actual_volume
        self.current_trade.close_deal_price = avg_price
        self.current_trade.close_deal_vol = actual_volume
        self.current_trade.close_fee = commission
        self.current_trade.total_fee = self.current_trade.open_fee + commission
        self.current_trade.profit = profit

        # 计算累计利润
        total_profit = sum(trade.profit for trade in self.trades) + profit
        self.current_trade.profit_t = total_profit

        # 统计
        if profit > 0:
            self.winning_trades += 1

        # 保存交易记录
        self.trades.append(self.current_trade)

        # 清空持仓
        self.position = Position()
        self.current_trade = None

    def _update_asset_value(self, timestamp: int) -> None:
        """更新资产净值记录"""
        current_time = datetime.fromtimestamp(timestamp)

        # 计算当前总资产
        total_asset = self.current_capital

        # 如果有持仓，计算持仓市值
        if self.position.volume > 0:
            mid_price = self.order_book.get_mid_price()
            if mid_price:
                position_value = self.position.volume * mid_price
                total_asset += position_value

        # 计算基准（买入并持有）
        benchmark_value = self._calculate_benchmark(timestamp)

        # 记录资产净值
        self.asset_history.append({
            'timestamp': current_time,
            'asset': total_asset,
            'benchmark': benchmark_value
        })

    def _calculate_benchmark(self, timestamp: int) -> float:
        """计算买入并持有基准"""
        if not hasattr(self, '_benchmark_start_price'):
            # 首次调用，记录起始价格
            mid_price = self.order_book.get_mid_price()
            if mid_price:
                self._benchmark_start_price = mid_price
                self._benchmark_shares = self.initial_capital / mid_price
                return self.initial_capital
            else:
                return self.initial_capital

        # 计算当前基准价值
        current_price = self.order_book.get_mid_price()
        if current_price and hasattr(self, '_benchmark_shares'):
            return self._benchmark_shares * current_price
        else:
            return self.initial_capital

    def force_close_position(self, timestamp: int) -> None:
        """强制平仓（回测结束时调用）"""
        if self.position.volume > 0:
            current_time = datetime.fromtimestamp(timestamp)

            # 创建强制平仓订单
            order = {
                'type': 'sell',
                'volume': self.position.volume,
                'timestamp': timestamp,
                'order_time': current_time,
            }

            self._execute_sell_order(order, timestamp)

    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.trades:
            return {}

        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        win_rate = self.winning_trades / len(self.trades) if self.trades else 0

        profits = [trade.profit for trade in self.trades]
        avg_profit = np.mean(profits) if profits else 0

        return {
            'total_trades': len(self.trades),
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_commission': self.total_commission,
            'avg_profit_per_trade': avg_profit,
            'final_capital': self.current_capital
        }
