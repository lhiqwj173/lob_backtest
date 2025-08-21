# {{ AURA-X | Action: Add | Reason: 创建性能指标计算模块，实现业界标准评价指标 | Approval: Cunzhi(ID:1735632000) }}
"""
性能指标计算模块
实现业界常用的量化交易评价指标
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化性能分析器
        
        Args:
            risk_free_rate: 无风险利率（年化）
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(self, asset_history: List[Dict], 
                            trades: List, 
                            initial_capital: float) -> Dict:
        """
        计算所有性能指标
        
        Args:
            asset_history: 资产净值历史
            trades: 交易记录
            initial_capital: 初始资金
            
        Returns:
            包含所有指标的字典
        """
        if not asset_history:
            return {}
        
        # 转换为DataFrame
        df = pd.DataFrame(asset_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # 计算收益率序列
        df['strategy_return'] = df['asset'].pct_change()
        df['benchmark_return'] = df['benchmark'].pct_change()
        
        # 计算累计收益率
        df['strategy_cumret'] = (df['asset'] / initial_capital) - 1
        df['benchmark_cumret'] = (df['benchmark'] / df['benchmark'].iloc[0]) - 1
        
        # 基础指标
        metrics = self._calculate_basic_metrics(df, trades, initial_capital)
        
        # 风险指标
        risk_metrics = self._calculate_risk_metrics(df)
        metrics.update(risk_metrics)
        
        # 回撤指标
        drawdown_metrics = self._calculate_drawdown_metrics(df)
        metrics.update(drawdown_metrics)
        
        # 交易指标
        trade_metrics = self._calculate_trade_metrics(trades)
        metrics.update(trade_metrics)
        
        return metrics
    
    def _calculate_basic_metrics(self, df: pd.DataFrame, 
                               trades: List, initial_capital: float) -> Dict:
        """计算基础收益指标"""
        if len(df) < 2:
            return {}
        
        # 时间相关
        start_date = df.index[0]
        end_date = df.index[-1]
        total_days = (end_date - start_date).days
        trading_days = len(df)
        
        # 最终收益
        final_asset = df['asset'].iloc[-1]
        final_benchmark = df['benchmark'].iloc[-1]
        
        # 总收益率
        total_return = (final_asset - initial_capital) / initial_capital
        benchmark_return = (final_benchmark - df['benchmark'].iloc[0]) / df['benchmark'].iloc[0]
        
        # 年化收益率
        years = total_days / 365.25
        annual_return = (final_asset / initial_capital) ** (1/years) - 1 if years > 0 else 0
        benchmark_annual = (final_benchmark / df['benchmark'].iloc[0]) ** (1/years) - 1 if years > 0 else 0
        
        # 超额收益
        excess_return = total_return - benchmark_return
        excess_annual = annual_return - benchmark_annual
        
        return {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'total_days': total_days,
            'trading_days': trading_days,
            'initial_capital': initial_capital,
            'final_asset': final_asset,
            'total_return': total_return,
            'annual_return': annual_return,
            'benchmark_return': benchmark_return,
            'benchmark_annual': benchmark_annual,
            'excess_return': excess_return,
            'excess_annual': excess_annual
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """计算风险指标"""
        if len(df) < 2:
            return {}
        
        strategy_returns = df['strategy_return'].dropna()
        benchmark_returns = df['benchmark_return'].dropna()
        
        if len(strategy_returns) == 0:
            return {}
        
        # 波动率（年化）
        strategy_vol = strategy_returns.std() * np.sqrt(252)
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        
        # 夏普比率
        excess_returns = strategy_returns - self.risk_free_rate/252
        sharpe_ratio = excess_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        # 信息比率（相对基准）
        active_returns = strategy_returns - benchmark_returns
        info_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
        
        # Beta和Alpha
        if len(benchmark_returns) > 1 and benchmark_returns.std() > 0:
            beta = np.cov(strategy_returns, benchmark_returns)[0,1] / np.var(benchmark_returns)
            alpha = strategy_returns.mean() - beta * benchmark_returns.mean()
            alpha_annual = alpha * 252
        else:
            beta = 0
            alpha_annual = 0
        
        # VaR和CVaR (95%置信度)
        var_95 = np.percentile(strategy_returns, 5)
        cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
        
        return {
            'volatility': strategy_vol,
            'benchmark_volatility': benchmark_vol,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': info_ratio,
            'beta': beta,
            'alpha_annual': alpha_annual,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _calculate_drawdown_metrics(self, df: pd.DataFrame) -> Dict:
        """计算回撤指标"""
        if len(df) < 2:
            return {}
        
        # 计算回撤序列
        cumulative = df['asset']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # 最大回撤
        max_drawdown = drawdown.min()
        
        # 最大回撤持续时间
        dd_duration = self._calculate_drawdown_duration(drawdown)
        
        # 回撤恢复时间
        recovery_time = self._calculate_recovery_time(drawdown)
        
        # Calmar比率 (年化收益/最大回撤)
        annual_return = df['strategy_cumret'].iloc[-1] * 252 / len(df) if len(df) > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': dd_duration,
            'avg_recovery_time': recovery_time,
            'calmar_ratio': calmar_ratio
        }
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """计算最大回撤持续时间"""
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> float:
        """计算平均回撤恢复时间"""
        recovery_times = []
        in_drawdown = False
        drawdown_start = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                # 开始回撤
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                # 回撤结束
                recovery_times.append(i - drawdown_start)
                in_drawdown = False
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def _calculate_trade_metrics(self, trades: List) -> Dict:
        """计算交易相关指标"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_holding_period': 0
            }
        
        profits = [trade.profit for trade in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        # 基础统计
        total_trades = len(trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # 盈亏统计
        avg_profit = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # 盈亏比
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf')
        
        # 平均持仓时间
        holding_periods = []
        for trade in trades:
            if trade.open_deal_timestamp and trade.close_deal_timestamp:
                duration = (trade.close_deal_timestamp - trade.open_deal_timestamp).total_seconds() / 3600  # 小时
                holding_periods.append(duration)
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding_period
        }
