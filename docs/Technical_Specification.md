# 技术规格说明

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    LOB回测系统架构                           │
├─────────────────────────────────────────────────────────────┤
│  用户接口层 (User Interface Layer)                          │
│  ├── main.py (主入口)                                       │
│  ├── test_with_sample_data.py (测试脚本)                    │
│  └── config/backtest_config.yaml (配置文件)                 │
├─────────────────────────────────────────────────────────────┤
│  数据处理层 (Data Processing Layer)                         │
│  ├── LOBDataLoader (十档盘口数据加载)                       │
│  ├── SignalDataLoader (信号数据加载)                        │
│  └── DataValidator (数据验证) [可选扩展]                     │
├─────────────────────────────────────────────────────────────┤
│  核心引擎层 (Core Engine Layer)                             │
│  ├── OrderBook (订单簿模拟器)                               │
│  ├── MatchingEngine (撮合引擎)                              │
│  └── PerformanceMonitor (性能监控) [可选扩展]               │
├─────────────────────────────────────────────────────────────┤
│  分析输出层 (Analysis & Output Layer)                       │
│  ├── PerformanceAnalyzer (性能分析)                         │
│  ├── BacktestVisualizer (可视化)                           │
│  └── ErrorHandler (错误处理) [可选扩展]                     │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件说明

#### 1. 数据处理层
- **LOBDataLoader**: 处理十档盘口CSV数据，支持时间过滤、数据清理、格式转换
- **SignalDataLoader**: 处理深度学习信号数据，支持多种阈值策略
- **时区处理**: 强制使用北京时间 (Asia/Shanghai)

#### 2. 核心引擎层
- **OrderBook**: 模拟十档订单簿，支持市价单撮合
- **MatchingEngine**: 核心撮合逻辑，支持全仓模式、延迟处理
- **Numba优化**: 关键算法使用JIT编译提升性能

#### 3. 分析输出层
- **PerformanceAnalyzer**: 计算20+业界标准指标
- **BacktestVisualizer**: 生成专业回测报告
- **多格式输出**: CSV、JSON、PNG格式

## 数据流程

### 1. 数据加载流程
```
原始数据文件 → 数据验证 → 格式转换 → 时间对齐 → 内存优化
```

### 2. 回测执行流程
```
信号生成 → 订单提交 → 延迟处理 → 撮合执行 → 持仓更新 → 记录保存
```

### 3. 结果输出流程
```
原始数据 → 指标计算 → 可视化生成 → 文件保存 → 报告输出
```

## 性能规格

### 处理能力
- **数据吞吐量**: ~1000 ticks/秒
- **内存使用**: ~2GB (一天数据)
- **撮合延迟**: <1ms (单次)
- **支持数据量**: 100万+ tick数据

### 系统要求
- **CPU**: Intel i5+ 或同等性能
- **内存**: 8GB+ (推荐16GB)
- **存储**: 10GB+ 可用空间
- **Python**: 3.8+

### 优化特性
- **Numba JIT**: 核心算法编译优化
- **向量化计算**: Pandas/NumPy优化
- **内存管理**: 自动垃圾回收
- **并行处理**: 支持多进程扩展

## 算法实现

### 1. 市价单撮合算法

```python
@jit(nopython=True)
def calculate_market_impact(levels, volumes, order_volume):
    """
    高性能市价单撮合算法
    - 使用Numba JIT编译
    - O(n)时间复杂度
    - 支持部分成交
    """
    total_cost = 0.0
    remaining_volume = order_volume
    filled_volume = 0.0
    
    for i in range(len(levels)):
        if remaining_volume <= 0:
            break
        
        available_volume = volumes[i]
        fill_volume = min(remaining_volume, available_volume)
        
        total_cost += fill_volume * levels[i]
        filled_volume += fill_volume
        remaining_volume -= fill_volume
    
    avg_price = total_cost / filled_volume if filled_volume > 0 else 0.0
    return avg_price, filled_volume
```

### 2. 滑点计算模型

```python
def calculate_slippage(execution_price, mid_price, side):
    """
    基于中间价的滑点计算
    - 买入滑点: (成交价 - 中间价) / 中间价
    - 卖出滑点: (中间价 - 成交价) / 中间价
    """
    if side == 'buy':
        return (execution_price - mid_price) / mid_price
    else:
        return (mid_price - execution_price) / mid_price
```

### 3. 性能指标算法

#### 夏普比率
```python
sharpe_ratio = (mean_return - risk_free_rate) / std_return * sqrt(252)
```

#### 最大回撤
```python
running_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - running_max) / running_max
max_drawdown = drawdown.min()
```

#### 信息比率
```python
active_returns = strategy_returns - benchmark_returns
information_ratio = active_returns.mean() / active_returns.std() * sqrt(252)
```

## 数据结构设计

### 1. 订单簿快照
```python
{
    'timestamp': int,           # 时间戳
    'bids': [(price, volume)],  # 买盘 [(价格, 数量)]
    'asks': [(price, volume)]   # 卖盘 [(价格, 数量)]
}
```

### 2. 交易记录
```python
@dataclass
class Trade:
    symbol: str                    # 标的代码
    open_order_timestamp: datetime # 开仓委托时间
    open_deal_timestamp: datetime  # 开仓成交时间
    open_deal_price: float        # 开仓成交价
    open_deal_vol: float          # 开仓成交量
    open_fee: float               # 开仓手续费
    close_order_timestamp: datetime # 平仓委托时间
    close_deal_timestamp: datetime  # 平仓成交时间
    close_deal_price: float       # 平仓成交价
    close_deal_vol: float         # 平仓成交量
    close_fee: float              # 平仓手续费
    profit: float                 # 单笔盈亏
    profit_t: float               # 累计盈亏
```

### 3. 性能指标
```python
{
    'total_return': float,        # 总收益率
    'annual_return': float,       # 年化收益率
    'max_drawdown': float,        # 最大回撤
    'sharpe_ratio': float,        # 夏普比率
    'volatility': float,          # 波动率
    'win_rate': float,            # 胜率
    'profit_factor': float,       # 盈亏比
    'total_trades': int,          # 总交易次数
    'calmar_ratio': float,        # 卡玛比率
    'information_ratio': float    # 信息比率
}
```

## 配置系统

### 配置文件结构
```yaml
# 数据源配置
data:
  lob_data_path: ""
  signal_data_path: ""
  timezone: "Asia/Shanghai"

# 信号配置
signal:
  source: "predict"           # target/predict
  predict_threshold: "max"    # max/数值

# 交易配置
trading:
  delay_ticks: 0             # 延迟tick数
  commission_rate: 5e-5      # 手续费率
  all_in_mode: true          # 全仓模式
  initial_capital: 1000000   # 初始资金

# 撮合引擎配置
matching:
  order_type: "market"           # 订单类型
  slippage_model: "market_impact" # 滑点模型
  use_all_levels: true           # 使用全部档位

# 性能分析配置
analysis:
  benchmark: "buy_and_hold"  # 基准策略
  output_dir: "results"      # 输出目录

# 可视化配置
visualization:
  enable: true
  save_plots: true
  plot_format: "png"
```

### 配置管理类
```python
class BacktestConfig:
    def __init__(self, config_path=None)
    def get(self, key, default=None)      # 获取配置值
    def set(self, key, value)             # 设置配置值
    def save(self, path=None)             # 保存配置
    def data_config(self)                 # 数据配置属性
    def trading_config(self)              # 交易配置属性
```

## 错误处理机制

### 异常类型层次
```python
BacktestError                 # 基础异常
├── DataLoadError            # 数据加载异常
├── DataValidationError      # 数据验证异常
├── MatchingEngineError      # 撮合引擎异常
└── ConfigurationError       # 配置异常
```

### 错误恢复策略
1. **数据错误**: 跳过异常数据，记录警告
2. **撮合错误**: 跳过当前tick，继续处理
3. **配置错误**: 使用默认值，发出警告
4. **系统错误**: 保存中间结果，安全退出

## 扩展接口

### 1. 自定义信号策略接口
```python
class CustomSignalStrategy:
    def generate_signals(self, data, params):
        # 实现自定义信号逻辑
        return signals
```

### 2. 自定义性能指标接口
```python
class CustomMetric:
    def calculate(self, returns, trades):
        # 实现自定义指标计算
        return metric_value
```

### 3. 自定义可视化接口
```python
class CustomVisualizer:
    def create_chart(self, data, config):
        # 实现自定义图表
        return chart
```

## 测试框架

### 单元测试覆盖
- 数据加载模块测试
- 撮合引擎逻辑测试
- 性能指标计算测试
- 配置管理测试

### 集成测试
- 端到端回测流程测试
- 大数据量性能测试
- 异常情况处理测试

### 性能基准测试
- 处理速度基准
- 内存使用基准
- 准确性验证基准

---

本技术规格说明提供了系统的详细技术实现细节，为开发者理解和扩展系统提供参考。
