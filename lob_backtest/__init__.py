# {{ AURA-X | Action: Add | Reason: 创建src包初始化文件 | Approval: Cunzhi(ID:1735632000) }}
"""
订单簿级别撮合回测系统
LOB (Limit Order Book) Backtesting System
"""

__version__ = "1.0.0"
__author__ = "Deep Learning Quantitative Engineer"

# 导出主要的类和函数，方便用户直接从包中导入
from .utils.config import BacktestConfig
from .data.lob_data_loader import LOBDataLoader
from .data.signal_data_loader import SignalDataLoader
from .engine.matching_engine import MatchingEngine
from .analysis.performance_metrics import PerformanceAnalyzer
from .analysis.visualization import BacktestVisualizer
from .backtester import LOBBacktester

# 为了向后兼容，保留原来的导入方式
__all__ = [
    "BacktestConfig",
    "LOBDataLoader",
    "SignalDataLoader",
    "MatchingEngine",
    "PerformanceAnalyzer",
    "BacktestVisualizer",
    "LOBBacktester"
]
