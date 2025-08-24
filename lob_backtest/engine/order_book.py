# {{ AURA-X | Action: Add | Reason: 创建订单簿模拟器，支持十档深度和市价单撮合 | Approval: Cunzhi(ID:1735632000) }}
"""
订单簿模拟器
模拟十档盘口的订单簿，支持市价单撮合和滑点计算
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from numba import jit
import numba


@jit(nopython=True)
def calculate_market_impact(levels: np.ndarray, volumes: np.ndarray, 
                          order_volume: float) -> Tuple[float, float]:
    """
    计算市价单的市场冲击和平均成交价格
    使用Numba JIT编译以获得最高性能
    
    Args:
        levels: 价格档位数组 [price1, price2, ...]
        volumes: 对应数量数组 [vol1, vol2, ...]
        order_volume: 订单数量
        
    Returns:
        (平均成交价格, 实际成交数量)
    """
    if order_volume <= 0 or len(levels) == 0:
        return 0.0, 0.0
    
    total_cost = 0.0
    remaining_volume = order_volume
    filled_volume = 0.0
    
    for i in range(len(levels)):
        if remaining_volume <= 0:
            break
            
        available_volume = volumes[i]
        if available_volume <= 0:
            continue
            
        # 本档位可成交数量
        fill_volume = min(remaining_volume, available_volume)
        
        # 累计成本和数量
        total_cost += fill_volume * levels[i]
        filled_volume += fill_volume
        remaining_volume -= fill_volume
    
    if filled_volume > 0:
        avg_price = total_cost / filled_volume
        return avg_price, filled_volume
    else:
        return 0.0, 0.0


class OrderBook:
    """订单簿模拟器"""
    
    def __init__(self):
        """初始化订单簿"""
        # 使用Numpy数组存储订单簿，格式为 [price, volume]
        self.bids = np.empty((0, 2), dtype=np.float64)
        self.asks = np.empty((0, 2), dtype=np.float64)
        self.last_update_time = 0
    
    def update_snapshot(self, snapshot: Dict) -> None:
        """
        更新订单簿快照
        
        Args:
            snapshot: 包含bids和asks的快照数据
        """
        bids_data = snapshot.get('bids', [])
        asks_data = snapshot.get('asks', [])
        
        self.bids = np.array(bids_data, dtype=np.float64)
        self.asks = np.array(asks_data, dtype=np.float64)
        self.last_update_time = snapshot.get('timestamp', 0)
        
        # 使用Numpy进行高效排序
        if self.bids.size > 0:
            self.bids = self.bids[np.argsort(self.bids[:, 0])[::-1]]
        if self.asks.size > 0:
            self.asks = self.asks[np.argsort(self.asks[:, 0])]
    
    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        """获取最优买价"""
        return (self.bids[0, 0], self.bids[0, 1]) if self.bids.size > 0 else None
    
    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        """获取最优卖价"""
        return (self.asks[0, 0], self.asks[0, 1]) if self.asks.size > 0 else None
    
    def get_mid_price(self) -> Optional[float]:
        """获取中间价"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid[0] + best_ask[0]) / 2.0
        return None
    
    def get_spread(self) -> Optional[float]:
        """获取买卖价差"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        return None
    
    def execute_market_buy(self, volume: float) -> Dict:
        """
        执行市价买单
        
        Args:
            volume: 买入数量
            
        Returns:
            成交结果字典
        """
        if self.asks.size == 0 or volume <= 0:
            return {
                'success': False,
                'filled_volume': 0.0,
                'avg_price': 0.0,
                'total_cost': 0.0,
                'remaining_volume': volume
            }

        # 直接使用内部的Numpy数组调用Numba优化函数
        avg_price, filled_volume = calculate_market_impact(
            self.asks[:, 0], self.asks[:, 1], float(volume)
        )

        # 整理返回结果
        if filled_volume > 0:
            total_cost = avg_price * filled_volume
            success = True
        else:
            total_cost = 0.0
            avg_price = 0.0
            success = False

        return {
            'success': success,
            'filled_volume': filled_volume,
            'avg_price': avg_price,
            'total_cost': total_cost,
            'remaining_volume': volume - filled_volume,
            'slippage': self._calculate_slippage(avg_price, 'buy')
        }
    
    def execute_market_sell(self, volume: float) -> Dict:
        """
        执行市价卖单
        
        Args:
            volume: 卖出数量
            
        Returns:
            成交结果字典
        """
        if self.bids.size == 0 or volume <= 0:
            return {
                'success': False,
                'filled_volume': 0.0,
                'avg_price': 0.0,
                'total_cost': 0.0,
                'remaining_volume': volume
            }

        # 直接使用内部的Numpy数组调用Numba优化函数
        avg_price, filled_volume = calculate_market_impact(
            self.bids[:, 0], self.bids[:, 1], float(volume)
        )

        # 整理返回结果
        if filled_volume > 0:
            total_cost = avg_price * filled_volume
            success = True
        else:
            total_cost = 0.0
            avg_price = 0.0
            success = False

        return {
            'success': success,
            'filled_volume': filled_volume,
            'avg_price': avg_price,
            'total_cost': total_cost,
            'remaining_volume': volume - filled_volume,
            'slippage': self._calculate_slippage(avg_price, 'sell')
        }
    
    def _calculate_slippage(self, execution_price: float, side: str) -> float:
        """
        计算滑点
        
        Args:
            execution_price: 实际成交价格
            side: 交易方向 ('buy' 或 'sell')
            
        Returns:
            滑点（相对于中间价的偏差）
        """
        mid_price = self.get_mid_price()
        if not mid_price or execution_price <= 0:
            return 0.0
        
        if side == 'buy':
            # 买入滑点：实际价格高于中间价的比例
            return (execution_price - mid_price) / mid_price
        else:
            # 卖出滑点：中间价高于实际价格的比例
            return (mid_price - execution_price) / mid_price
    
    def get_market_depth(self, side: str, max_levels: int = 10) -> List[Tuple[float, float]]:
        """
        获取市场深度
        
        Args:
            side: 'bid' 或 'ask'
            max_levels: 最大档位数
            
        Returns:
            价格和数量的列表
        """
        if side == 'bid':
            return [tuple(row) for row in self.bids[:max_levels]]
        elif side == 'ask':
            return [tuple(row) for row in self.asks[:max_levels]]
        else:
            raise ValueError("side must be 'bid' or 'ask'")
    
    def get_total_volume(self, side: str, max_levels: int = 10) -> float:
        """
        获取指定方向的总挂单量
        
        Args:
            side: 'bid' 或 'ask'
            max_levels: 最大档位数
            
        Returns:
            总挂单量
        """
        depth = self.get_market_depth(side, max_levels)
        return sum(volume for _, volume in depth)
