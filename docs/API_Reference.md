# API参考文档

## 核心类和方法

### LOBBacktester

主回测系统类，提供完整的回测流程。

#### 初始化
```python
LOBBacktester(config_path: str = None)
```

**参数:**
- `config_path`: 配置文件路径，默认使用 `config/backtest_config.yaml`

#### 主要方法

##### run_backtest()
```python
run_backtest(lob_data_path: str, signal_data_path: str, symbol: str = "UNKNOWN") -> dict
```

运行完整回测流程。

**参数:**
- `lob_data_path`: 十档盘口数据文件路径
- `signal_data_path`: 信号数据文件路径  
- `symbol`: 交易标的代码

**返回:**
```python
{
    'metrics': dict,           # 性能指标
    'trades': List[Trade],     # 交易记录
    'asset_history': List[dict] # 资产净值历史
}
```

### LOBDataLoader

十档盘口数据加载器。

#### 初始化
```python
LOBDataLoader(timezone: str = "Asia/Shanghai")
```

#### 主要方法

##### load_data()
```python
load_data(file_path: str, begin_time: str = "09:30", end_time: str = "15:00") -> pd.DataFrame
```

加载并清理十档盘口数据。

**参数:**
- `file_path`: CSV文件路径
- `begin_time`: 开始时间 (HH:MM格式)
- `end_time`: 结束时间 (HH:MM格式)

##### get_orderbook_snapshot()
```python
get_orderbook_snapshot(data: pd.DataFrame, timestamp: int) -> dict
```

获取指定时间戳的订单簿快照。

### SignalDataLoader

深度学习信号数据加载器。

#### 主要方法

##### load_signal_data()
```python
load_signal_data(file_path: str) -> pd.DataFrame
```

##### generate_signals()
```python
generate_signals(data: pd.DataFrame, source: str = "predict", 
                threshold_strategy: Union[str, float] = "max") -> pd.DataFrame
```

生成交易信号。

**参数:**
- `source`: 信号源 ("target" 或 "predict")
- `threshold_strategy`: 阈值策略 ("max" 或具体数值)

### MatchingEngine

撮合引擎核心类。

#### 初始化
```python
MatchingEngine(initial_capital: float = 1000000.0,
               commission_rate: float = 5e-5,
               delay_ticks: int = 0,
               timezone: str = "Asia/Shanghai")
```

#### 主要方法

##### update_market_data()
```python
update_market_data(lob_snapshot: Dict, timestamp: int) -> None
```

更新市场数据。

##### submit_signal()
```python
submit_signal(signal: int, signal_info: Dict, timestamp: int) -> bool
```

提交交易信号。

**参数:**
- `signal`: 交易信号 (0=持仓, 1=空仓)
- `signal_info`: 信号详细信息
- `timestamp`: 信号时间戳

##### force_close_position()
```python
force_close_position(timestamp: int) -> None
```

强制平仓（回测结束时调用）。

### OrderBook

订单簿模拟器。

#### 主要方法

##### update_snapshot()
```python
update_snapshot(snapshot: Dict) -> None
```

更新订单簿快照。

##### execute_market_buy()
```python
execute_market_buy(volume: float) -> Dict
```

执行市价买单。

**返回:**
```python
{
    'success': bool,
    'filled_volume': float,
    'avg_price': float,
    'total_cost': float,
    'remaining_volume': float,
    'slippage': float
}
```

##### execute_market_sell()
```python
execute_market_sell(volume: float) -> Dict
```

执行市价卖单，返回格式同上。

### PerformanceAnalyzer

性能分析器。

#### 主要方法

##### calculate_all_metrics()
```python
calculate_all_metrics(asset_history: List[Dict], trades: List, 
                     initial_capital: float) -> Dict
```

计算所有性能指标。

**返回指标包括:**
- `total_return`: 总收益率
- `annual_return`: 年化收益率
- `max_drawdown`: 最大回撤
- `sharpe_ratio`: 夏普比率
- `volatility`: 波动率
- `win_rate`: 胜率
- `profit_factor`: 盈亏比

### BacktestVisualizer

回测结果可视化器。

#### 主要方法

##### create_comprehensive_report()
```python
create_comprehensive_report(asset_history: List[Dict], trades: List, 
                          metrics: Dict, output_dir: str = "results") -> None
```

创建综合回测报告。

## 数据结构

### Trade类
```python
@dataclass
class Trade:
    symbol: str
    open_order_timestamp: datetime
    open_deal_timestamp: datetime
    open_order_price: float
    open_order_vol: float
    open_deal_price: float
    open_deal_vol: float
    open_fee: float
    close_order_timestamp: datetime
    close_deal_timestamp: datetime
    close_order_price: float
    close_order_vol: float
    close_deal_price: float
    close_deal_vol: float
    close_fee: float
    total_fee: float
    profit: float
    profit_t: float  # 累计利润
```

### Position类
```python
@dataclass
class Position:
    symbol: str = ""
    volume: float = 0.0
    avg_price: float = 0.0
    open_time: datetime = None
    total_cost: float = 0.0
```

## 配置参数

### 数据配置
```yaml
data:
  lob_data_path: ""           # 十档盘口数据路径
  signal_data_path: ""        # 信号数据路径
  timezone: "Asia/Shanghai"   # 时区
```

### 信号配置
```yaml
signal:
  source: "predict"           # target/predict
  predict_threshold: "max"    # max/数值
```

### 交易配置
```yaml
trading:
  delay_ticks: 0             # 延迟tick数
  commission_rate: 5e-5      # 手续费率
  all_in_mode: true          # 全仓模式
  initial_capital: 1000000   # 初始资金
```

### 撮合引擎配置
```yaml
matching:
  order_type: "market"           # 订单类型
  slippage_model: "market_impact" # 滑点模型
  use_all_levels: true           # 使用全部档位
```

## 异常处理

### 常见异常类型

- `FileNotFoundError`: 数据文件不存在
- `ValueError`: 数据格式错误
- `KeyError`: 缺少必要的数据列
- `MemoryError`: 内存不足

### 错误处理建议

1. **数据文件检查**: 确保文件路径正确且文件存在
2. **数据格式验证**: 使用系统提供的验证功能
3. **内存管理**: 大数据集考虑分批处理
4. **日志查看**: 检查系统生成的日志文件

## 性能优化

### 数据处理优化
- 使用向量化操作替代循环
- 预先过滤不需要的数据
- 合理设置数据类型

### 撮合引擎优化
- Numba JIT编译已自动优化
- 避免频繁的对象创建
- 使用适当的数据结构

### 内存优化
- 及时释放不需要的变量
- 使用生成器处理大数据集
- 监控内存使用情况
