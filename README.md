# LOB回测系统 (Limit Order Book Backtesting System)

## 项目简介

这是一个专业的订单簿级别撮合回测系统，专门用于对交易信号进行高精度的回测分析。系统支持十档盘口数据处理、深度学习信号集成、实时撮合模拟和全面的性能分析。

## 核心功能

### 🔄 数据处理模块
- **十档盘口数据加载**: 支持CSV格式的秒级盘口数据，强制使用北京时间
- **深度学习信号处理**: 处理模型预测结果，支持多种阈值策略
- **数据清理与验证**: 自动处理涨跌停、异常值和缺失数据

### ⚡ 高性能撮合引擎
- **Numba JIT优化**: 核心撮合算法使用JIT编译，确保最高性能
- **全档位撮合**: 真实模拟市场深度消耗和滑点影响
- **延迟模拟**: 可配置的tick延迟，模拟真实交易环境
- **全仓模式**: 支持All-in交易策略

### 📊 性能分析与可视化
- **业界标准指标**: 夏普比率、最大回撤、胜率、盈亏比等
- **基准对比**: 与买入并持有策略对比
- **交互式图表**: 净值曲线、回撤图、交易分析
- **详细报告**: 自动生成PDF格式的回测报告

## 项目结构

```
lob_backtest/
├── src/                          # 源代码目录
│   ├── data/                     # 数据处理模块
│   │   ├── lob_data_loader.py    # 十档盘口数据加载器
│   │   └── signal_data_loader.py # 信号数据加载器
│   ├── engine/                   # 撮合引擎模块
│   │   ├── order_book.py         # 订单簿模拟器
│   │   └── matching_engine.py    # 撮合引擎核心
│   ├── analysis/                 # 分析模块
│   │   ├── performance_metrics.py # 性能指标计算
│   │   └── visualization.py      # 可视化模块
│   └── utils/                    # 工具模块
│       └── config.py             # 配置管理
├── config/                       # 配置文件
│   └── backtest_config.yaml      # 回测配置
├── docs/                         # 完整文档
│   ├── README.md                 # 文档中心
│   ├── Installation_Guide.md     # 安装指南
│   ├── User_Guide.md             # 用户指南
│   ├── API_Reference.md          # API参考
│   ├── Technical_Specification.md # 技术规格
│   └── FAQ.md                    # 常见问题
├── 参考/                         # 参考代码
│   ├── 格式化十档盘口数据.py      # 数据格式化参考
│   └── 回测结果可视化.py          # 可视化参考
├── test_with_sample_data.py      # 测试脚本（模拟数据）
├── main.py                       # 主入口文件
├── requirements.txt              # 依赖包列表
└── README.md                     # 项目文档
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 运行测试

系统提供了完整的测试脚本，使用模拟数据验证所有功能：

```bash
# 使用模拟数据测试系统
python test_with_sample_data.py
```

测试脚本会自动：
- 生成一天的模拟十档盘口数据（6.5小时交易时间）
- 生成对应的深度学习信号数据
- 运行完整的回测流程
- 输出性能指标和可视化报告

**预期输出示例：**
```
✅ 所有系统组件验证通过
正在生成模拟LOB数据...
生成了 7800 条LOB记录
正在生成模拟信号数据...
生成了 1560 条信号记录
✅ 回测成功完成!
📊 关键指标:
   总收益率: 2.34%
   年化收益率: 15.67%
   最大回撤: -1.23%
   夏普比率: 1.456
   总交易次数: 23
   胜率: 65.22%
```

### 3. 使用真实数据

```python
from main import LOBBacktester

# 初始化回测系统
backtester = LOBBacktester()

# 运行回测
results = backtester.run_backtest(
    lob_data_path="path/to/lob_data.csv",
    signal_data_path="path/to/signal_data.csv",
    symbol="YOUR_SYMBOL"
)
```

## 数据格式要求

### 十档盘口数据格式
```csv
时间,卖10价,卖10量,卖9价,卖9量,...,卖1价,卖1量,买1价,买1量,...,买10价,买10量,卖均,买均
2025-06-03 09:30:05,1.370,12868,1.369,8999,...,1.361,16615,1.360,4394,...,1.351,6624,1.389,1.330
```

### 深度学习信号数据格式
```csv
timestamp,target,has_pos,0,1
1725865015,1.0,0,0.5339759,0.46602404
1725865015,1.0,1,0.50600046,0.49399954
```

## 配置说明

主要配置项在 `config/backtest_config.yaml` 中：

```yaml
# 信号配置
signal:
  source: "predict"              # target/predict
  predict_threshold: "max"       # max/数值

# 交易配置  
trading:
  delay_ticks: 0                # 延迟tick数
  commission_rate: 5e-5         # 手续费率
  all_in_mode: true             # 全仓模式
  initial_capital: 1000000      # 初始资金

# 撮合引擎配置
matching:
  order_type: "market"          # 市价单
  slippage_model: "market_impact" # 滑点模型
  use_all_levels: true          # 使用全部档位
```

## 输出文件

系统会在 `results/` 目录下生成以下文件：

- `{symbol}_asset_nets.csv`: 资产净值历史
- `{symbol}_trades.csv`: 详细交易记录  
- `{symbol}_metrics.json`: 性能指标
- `backtest_report.png`: 可视化报告
- `trade_analysis.png`: 交易分析图

## 核心技术特性

### 高性能计算
- 使用Numba JIT编译优化撮合算法
- 向量化数据处理，支持大规模数据集
- 内存优化，支持长时间回测

### 精确撮合模拟
- 全档位市价单撮合
- 基于订单簿深度的动态滑点模型
- 真实的手续费和延迟模拟

### 专业分析
- 20+业界标准性能指标
- 风险调整收益分析
- 交易行为统计分析

## 使用示例

### 基础回测
```python
from main import LOBBacktester

backtester = LOBBacktester()
results = backtester.run_backtest(
    lob_data_path="data/lob.csv",
    signal_data_path="data/signals.csv", 
    symbol="ETF_513330"
)

print(f"总收益率: {results['metrics']['total_return']:.2%}")
print(f"夏普比率: {results['metrics']['sharpe_ratio']:.3f}")
```

### 自定义配置
```python
from utils.config import BacktestConfig

# 修改配置
config = BacktestConfig()
config.set('trading.commission_rate', 1e-4)  # 调整手续费
config.set('signal.predict_threshold', 0.6)  # 调整信号阈值
config.save()

# 使用自定义配置
backtester = LOBBacktester()
```

## 性能基准

在标准测试环境下（Intel i7, 16GB RAM）：
- 处理速度: ~1000 ticks/秒
- 内存使用: ~2GB（一天数据）
- 撮合延迟: <1ms（单次）

## 注意事项

1. **时区处理**: 所有时间数据强制使用北京时间（Asia/Shanghai）
2. **数据质量**: 系统会自动检测和处理数据异常，但建议预先清理数据
3. **内存管理**: 大数据集建议分批处理或增加系统内存
4. **精度设置**: 价格精度默认为3位小数，可根据标的调整

## 扩展开发

### 添加新的信号策略
```python
# 在 signal_data_loader.py 中扩展
def custom_threshold_strategy(self, data, custom_params):
    # 实现自定义阈值逻辑
    pass
```

### 添加新的性能指标
```python
# 在 performance_metrics.py 中扩展
def calculate_custom_metric(self, returns):
    # 实现自定义指标计算
    pass
```

## 技术支持

如遇到问题，请检查：
1. 数据格式是否符合要求
2. 配置文件是否正确
3. 依赖包是否完整安装
4. 系统资源是否充足

## 版本历史

- v1.0.0: 初始版本，支持基础回测功能
- 核心功能: LOB数据处理、信号集成、撮合引擎、性能分析

---

**开发团队**: 深度学习金融量化工程师  
**技术栈**: Python, Numba, Pandas, NumPy, Matplotlib  
**许可证**: MIT License
