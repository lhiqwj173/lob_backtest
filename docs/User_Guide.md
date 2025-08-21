# 用户使用指南

## 系统概述

LOB回测系统是一个专业的订单簿级别撮合回测平台，专门用于量化交易策略的精确回测。系统支持十档盘口数据处理、深度学习信号集成和高精度撮合模拟。

## 安装与配置

### 1. 环境要求

- Python 3.8+
- 内存: 8GB+ (推荐16GB)
- 硬盘: 10GB+ 可用空间

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python test_with_sample_data.py
```

如果看到"✅ 所有系统组件验证通过"，说明安装成功。

## 数据准备

### 十档盘口数据格式

系统需要CSV格式的十档盘口数据，包含以下列：

**必需列:**
- `时间`: 时间戳，格式如 "2025-06-03 09:30:05"
- `卖1价` 到 `卖10价`: 卖盘十档价格
- `卖1量` 到 `卖10量`: 卖盘十档数量
- `买1价` 到 `买10价`: 买盘十档价格  
- `买1量` 到 `买10量`: 买盘十档数量

**可选列:**
- `卖均`, `买均`: 平均价格（系统会自动计算）

**示例数据:**
```csv
时间,卖10价,卖10量,卖9价,卖9量,...,卖1价,卖1量,买1价,买1量,...,买10价,买10量
2025-06-03 09:30:05,1.370,12868,1.369,8999,...,1.361,16615,1.360,4394,...,1.351,6624
2025-06-03 09:30:08,1.369,9368,1.368,8482,...,1.360,3655,1.359,107851,...,1.350,19461
```

### 深度学习信号数据格式

信号数据需要包含以下列：

**必需列:**
- `timestamp`: UTC时间戳（秒级）
- `target`: 正确标签 (0=持仓, 1=空仓)
- `has_pos`: 执行前持仓状态 (0=无持仓, 1=有持仓)
- `0`: 模型预测持仓概率
- `1`: 模型预测空仓概率

**示例数据:**
```csv
timestamp,target,has_pos,0,1
1725865015,1.0,0,0.5339759,0.46602404
1725865015,1.0,1,0.50600046,0.49399954
```

## 基础使用

### 1. 快速开始

```python
from main import LOBBacktester

# 初始化回测系统
backtester = LOBBacktester()

# 运行回测
results = backtester.run_backtest(
    lob_data_path="data/lob_data.csv",
    signal_data_path="data/signal_data.csv",
    symbol="ETF_513330"
)

# 查看结果
print(f"总收益率: {results['metrics']['total_return']:.2%}")
```

### 2. 使用测试数据

如果没有真实数据，可以使用系统生成的测试数据：

```bash
python test_with_sample_data.py
```

这会生成模拟的一天交易数据并运行完整回测。

## 配置详解

### 配置文件位置

主配置文件: `config/backtest_config.yaml`

### 主要配置项

#### 数据配置
```yaml
data:
  timezone: "Asia/Shanghai"  # 强制使用北京时间
```

#### 信号配置
```yaml
signal:
  source: "predict"          # 信号源: target(真实标签) / predict(模型预测)
  predict_threshold: "max"   # 阈值策略: max(最大概率) / 数值(固定阈值)
```

**阈值策略说明:**
- `"max"`: 选择概率最大的动作
- `0.6`: 当概率≥0.6时选择该动作，否则选择概率最大的

#### 交易配置
```yaml
trading:
  delay_ticks: 0            # 延迟tick数 (0=下一tick成交)
  commission_rate: 5e-5     # 手续费率 (万分之五)
  all_in_mode: true         # 全仓模式
  initial_capital: 1000000  # 初始资金
```

#### 撮合引擎配置
```yaml
matching:
  order_type: "market"           # 市价单
  slippage_model: "market_impact" # 滑点模型
  use_all_levels: true           # 使用全部档位撮合
```

## 高级功能

### 1. 自定义配置

```python
from utils.config import BacktestConfig

# 加载配置
config = BacktestConfig()

# 修改配置
config.set('trading.commission_rate', 1e-4)  # 调整手续费为万分之一
config.set('signal.predict_threshold', 0.7)  # 设置70%阈值

# 保存配置
config.save()

# 使用自定义配置
backtester = LOBBacktester()
```

### 2. 批量回测

```python
symbols = ["ETF_513330", "ETF_159941", "ETF_510300"]
results = {}

for symbol in symbols:
    lob_file = f"data/{symbol}_lob.csv"
    signal_file = f"data/{symbol}_signals.csv"
    
    result = backtester.run_backtest(lob_file, signal_file, symbol)
    results[symbol] = result
    
    print(f"{symbol} 回测完成: {result['metrics']['total_return']:.2%}")
```

### 3. 结果分析

```python
# 获取详细指标
metrics = results['metrics']

print("=== 收益指标 ===")
print(f"总收益率: {metrics['total_return']:.2%}")
print(f"年化收益率: {metrics['annual_return']:.2%}")
print(f"基准收益率: {metrics['benchmark_return']:.2%}")

print("=== 风险指标 ===")
print(f"最大回撤: {metrics['max_drawdown']:.2%}")
print(f"波动率: {metrics['volatility']:.2%}")
print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")

print("=== 交易指标 ===")
print(f"总交易次数: {metrics['total_trades']}")
print(f"胜率: {metrics['win_rate']:.2%}")
print(f"盈亏比: {metrics['profit_factor']:.2f}")
```

## 输出文件说明

回测完成后，系统会在 `results/` 目录生成以下文件：

### 1. 资产净值文件 (`{symbol}_asset_nets.csv`)
```csv
timestamp,asset,benchmark
2024-08-29 09:35:12.000,1000000,1.0555
2024-08-29 09:35:15.000,1001250,1.0558
```

### 2. 交易记录文件 (`{symbol}_trades.csv`)
```csv
symbol,open_order_timestamp,open_deal_timestamp,open_order_price,open_order_vol,open_deal_price,open_deal_vol,open_fee,close_order_timestamp,close_deal_timestamp,close_order_price,close_order_vol,close_deal_price,close_deal_vol,close_fee,total_fee,profit,profit_t
159941,2024-08-29 09:35:48.000,2024-08-29 09:35:51.000,1.79769e+308,900,1.056,900,0.04752,2024-08-29 09:43:39.000,2024-08-29 09:43:42.000,0,900,1.056,900,0.04752,0.09504,-0.009504,-0.009504
```

### 3. 性能指标文件 (`{symbol}_metrics.json`)
包含所有计算的性能指标，JSON格式。

### 4. 可视化报告 (`backtest_report.png`)
包含净值曲线、回撤图、月度收益热力图和关键指标表。

### 5. 交易分析图 (`trade_analysis.png`)
包含盈亏分布、累计盈亏、持仓时间分布和月度交易统计。

## 常见问题

### Q1: 数据加载失败
**A:** 检查以下项目：
- 文件路径是否正确
- 文件编码是否为UTF-8或GBK
- 数据格式是否符合要求
- 必需列是否完整

### Q2: 内存不足
**A:** 解决方案：
- 减少数据量（按时间段分割）
- 增加系统内存
- 关闭其他程序释放内存

### Q3: 回测速度慢
**A:** 优化建议：
- 减少数据精度（如3秒改为5秒间隔）
- 使用更快的硬盘（SSD）
- 确保Numba正确安装

### Q4: 结果异常
**A:** 检查项目：
- 信号数据的时间对齐
- 手续费设置是否合理
- 滑点模型是否适当
- 数据中是否有异常值

## 性能优化建议

### 1. 数据预处理
- 提前清理和验证数据
- 使用合适的数据类型
- 删除不必要的列

### 2. 系统配置
- 调整采样间隔
- 优化内存使用
- 使用SSD存储

### 3. 参数调优
- 合理设置延迟参数
- 优化信号阈值
- 调整手续费模型

## 扩展开发

### 1. 添加新指标
在 `src/analysis/performance_metrics.py` 中添加：

```python
def calculate_custom_metric(self, returns):
    # 实现自定义指标
    return custom_value
```

### 2. 自定义信号策略
在 `src/data/signal_data_loader.py` 中扩展：

```python
def custom_signal_strategy(self, data, params):
    # 实现自定义信号逻辑
    return signals
```

### 3. 新的可视化
在 `src/analysis/visualization.py` 中添加：

```python
def create_custom_chart(self, data):
    # 实现自定义图表
    pass
```

## 技术支持

如需技术支持，请提供：
1. 错误信息和堆栈跟踪
2. 数据样本（脱敏后）
3. 配置文件内容
4. 系统环境信息

---

更多详细信息请参考 [API参考文档](API_Reference.md)。
