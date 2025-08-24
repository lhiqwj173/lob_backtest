# 用户使用指南

## 系统概述

LOB回测系统是一个专业的订单簿级别撮合回测平台，专门用于量化交易策略的精确回测。系统支持十档盘口数据处理、深度学习信号集成和高精度撮合模拟。

## 安装与配置

请参考详细的 [安装指南](Installation_Guide.md)。

## 数据准备

### 十档盘口数据格式

系统需要CSV格式的十档盘口数据，包含以下列：

**必需列:**
- `datetime`: 时间字符串，格式如 "2025-06-03 09:30:05"
- `卖1价` 到 `卖10价`: 卖盘十档价格
- `卖1量` 到 `卖10量`: 卖盘十档数量
- `买1价` 到 `买10价`: 买盘十档价格  
- `买1量` 到 `买10量`: 买盘十档数量

**可选列:**
- `卖均`, `买均`: 平均价格（系统会自动计算）

**示例数据:**
```csv
datetime,卖10价,卖10量,卖9价,卖9量,...,卖1价,卖1量,买1价,买1量,...,买10价,买10量
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

# 运行回测 (参数从配置文件读取)
results = backtester.run_backtest()

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

所有配置项均在此文件中设置，以下为详细说明。

#### 1. 数据源配置 (`data`)

```yaml
data:
  lob_data_path: "D:/L2_DATA_T0_ETF/his_data"      # 十档盘口数据根目录
  signal_data_path: "513180_1725845725_1727161015.csv" # 深度学习信号数据路径
```

**详细说明:**
- `lob_data_path`: **必需**。指向十档盘口数据的 **根目录**。
- `signal_data_path`: **必需**。指向深度学习信号数据的 **CSV文件路径**。**文件名必须遵循 `<stock_id>_<start_timestamp>_<end_timestamp>.csv` 格式**，系统将从此文件名中解析出回测所需的标的、开始时间和结束时间。

**数据目录结构要求:**
`lob_data_path` 必须遵循以下结构，以便系统能根据从信号文件名中解析出的 `stock_id` 找到对应的数据：
```
<lob_data_path>/
└── <YYYYMMDD>/
    ├── <stock_id>/
    │   └── 十档盘口.csv
    ├── <stock_id>/
    │   └── 十档盘口.csv
    └── ...
```
**示例:**
```
D:/L2_DATA_T0_ETF/his_data/
└── 20240909/
    ├── 513180/
    │   └── 十档盘口.csv
    └── 513000/
        └── 十档盘口.csv
```

#### 2. 信号配置 (`signal`)
```yaml
signal:
  source: "predict"          # 信号源: target(真实标签) / predict(模型预测)
  predict_threshold: "max"   # 阈值策略: max(最大概率) / 数值(固定阈值)
```

**详细说明:**
- `source`:
    - `"target"`: 使用数据中的 `target` 列作为交易信号，用于验证理想情况下的策略表现。
    - `"predict"`: 使用模型预测的概率作为交易信号，更贴近实盘。
- `predict_threshold`:
    - `"max"`: 选择模型输出概率最高的类别作为信号。
    - `数值` (例如 `0.6`): 当某一类别的预测概率超过此阈值时，才触发交易信号。

#### 3. 交易配置 (`trading`)
```yaml
trading:
  delay_ticks: 0            # 延迟tick数 (0=下一tick成交)
  commission_rate: 0.00005  # 手续费率 (万分之五)
  all_in_mode: true         # 全仓模式
  initial_capital: 1000000  # 初始资金
  rest_force_close: true    # 中午收盘前10s强制平仓
  close_force_close: true   # 下午收盘前10s强制平仓
```
**详细说明:**
- `delay_ticks`: 信号触发后，延迟多少个tick再执行交易。`0` 表示在下一个tick立即执行。
- `commission_rate`: 单边交易手续费率。
- `all_in_mode`: 是否采用全仓交易模式。`true` 表示每次交易都投入全部可用资金。
- `initial_capital`: 回测的初始资金。
- `rest_force_close`: `true` 表示在每个交易日午休收盘前10秒（11:29:50之后）强制平掉所有持仓。
- `close_force_close`: `true` 表示在每个交易日下午收盘前10秒（14:59:50之后）强制平掉所有持仓。

#### 4. 回测配置 (`backtest`)
```yaml
backtest:
  debug: false # debug模式开关
```
**详细说明:**
- `debug`: `true` 表示开启Debug模式。在该模式下，系统会记录每个时间点的详细状态（如引擎持仓、信号、盘口价格等），并保存到`results`目录下的`<stock_id>_debug_records.csv`文件中，便于深度分析和调试。该文件包含 `datetime` (可读日期时间), `timestamp` (时间戳), `engine_has_pos` (引擎持仓状态), `signal` (交易信号), `signal_has_pos` (信号对应的持仓状态), `ask1_price` (卖一价), 和 `bid1_price` (买一价) 等字段。

#### 5. 撮合引擎配置 (`matching`)
```yaml
matching:
  order_type: "market"           # 订单类型，目前仅支持市价单
  slippage_model: "market_impact" # 滑点模型，目前仅支持市场冲击模型
  use_all_levels: true           # 是否使用全部档位进行撮合
```
**详细说明:**
- `order_type`: **订单类型**。目前固定为 `"market"` (市价单)，系统会立即以市场上最优的价格进行交易。
- `slippage_model`: **滑点模型**。目前固定为 `"market_impact"` (市场冲击模型)。这意味着系统会根据您的订单大小和当前订单簿的流动性深度来计算成交价格。如果您的订单量超过了第一档的挂单量，系统会自动“吃掉”后续档位的流动性，直到订单完全成交，这会导致最终成交均价变差，从而真实地模拟滑点。
- `use_all_levels`: **是否使用所有档位**。`true` 表示撮合时会利用订单簿中所有可用的对手方报价（即全部十档）来完成您的订单。
- 更多技术细节请参考 [技术规格说明](Technical_Specification.md)。

#### 6. 性能分析配置 (`analysis`)
```yaml
analysis:
  benchmark: "buy_and_hold"  # 基准策略
  output_dir: "results"      # 结果输出目录
```
**详细说明:**
- `benchmark`: 用于与策略表现进行对比的基准。目前支持 `"buy_and_hold"` (买入并持有)。
- `output_dir`: 保存回测结果（如性能指标CSV、图表等）的目录。

#### 7. 可视化配置 (`visualization`)
```yaml
visualization:
  enable: true
  save_plots: true
  plot_format: "png"
  interactive: false  # 是否启用交互式可视化
```
**详细说明:**
- `enable`: 总开关，控制是否生成任何可视化图表。
- `save_plots`: 当 `enable` 为 `true` 时，控制是否将生成的图表保存到 `output_dir`。
- `plot_format`: 保存图表的文件格式。
- `interactive`: 如果为 `true`，会尝试以交互模式显示图表（例如，使用 `matplotlib` 的交互式后端）。

## 结果解读

### 性能指标解读

- **总收益率 (Total Return)**: 整个回测期间的累计收益率。
- **年化收益率 (Annual Return)**: 将回测期间的收益率转换为年度标准。
- **基准收益率 (Benchmark Return)**: 买入并持有策略的收益率，用于对比。
- **最大回撤 (Max Drawdown)**: 衡量策略可能面临的最大亏损。
- **夏普比率 (Sharpe Ratio)**: 衡量每单位风险所获得的超额回报。越高越好。
- **胜率 (Win Rate)**: 盈利交易次数占总交易次数的比例。
- **盈亏比 (Profit Factor)**: 总盈利与总亏损的比值。大于1表示盈利。

### 可视化报告

- **净值曲线**: 直观展示策略资产随时间的变化。
- **回撤图**: 显示策略净值的回撤情况，帮助评估风险。
- **交易分析图**: 分析交易行为，如盈亏分布、持仓时间等。

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

config = BacktestConfig()
for symbol in symbols:
    # 假设时间戳范围对于所有标的都是相同的
    start_ts = 1725845725
    end_ts = 1727161015
    signal_file = f"{symbol}_{start_ts}_{end_ts}.csv"
    
    config.set('data.signal_data_path', signal_file)
    config.save()
    
    backtester = LOBBacktester()
    result = backtester.run_backtest()
    results[symbol] = result
    
    print(f"{symbol} 回测完成: {result['metrics']['total_return']:.2%}")
```

### 3. 参数网格搜索

```python
from utils.config import BacktestConfig
import pandas as pd

thresholds = [0.55, 0.6, 0.65]
delays = [0, 1, 2]
all_results = []

for threshold in thresholds:
    for delay in delays:
        config = BacktestConfig()
        config.set('signal.predict_threshold', threshold)
        config.set('trading.delay_ticks', delay)
        config.save()

        backtester = LOBBacktester()
        result = backtester.run_backtest()
        
        metrics = result['metrics']
        metrics['threshold'] = threshold
        metrics['delay'] = delay
        all_results.append(metrics)

results_df = pd.DataFrame(all_results)
print(results_df[['threshold', 'delay', 'total_return', 'sharpe_ratio']])
```

## 参数调优建议

### 信号阈值 (`predict_threshold`)
- **过高**: 可能导致交易次数过少，错过机会。
- **过低**: 可能导致交易过于频繁，增加交易成本和噪音。
- **建议**: 从 `0.5` 到 `0.7` 区间进行测试，找到最佳平衡点。

### 延迟成交 (`delay_ticks`)
- 模拟真实世界中的网络和处理延迟。
- **增加延迟**: 通常会降低收益，但更接近真实情况。
- **建议**: 根据您的交易环境，设置为 `0` 到 `5` ticks 进行测试。

### 手续费 (`commission_rate`)
- 对高频策略影响巨大。
- **建议**: 设置为您的券商提供的真实费率。

## 常见问题

### Q1: 数据加载失败
**A:** 检查文件路径、编码和格式是否符合要求。

### Q2: 内存不足
**A:** 减少数据量、增加系统内存或关闭其他程序。

### Q3: 回测速度慢
**A:** 减少数据采样频率、使用SSD硬盘、确保Numba正确安装。

---

更多详细信息请参考 [API参考文档](API_Reference.md)。
