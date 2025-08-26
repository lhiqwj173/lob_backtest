# System Architecture

## 1. 核心组件

LOB回测系统采用模块化设计，主要由以下几个核心组件构成：

- **数据处理模块 (`lob_backtest/data/`)**: 负责加载、解析和预处理十档盘口（LOB）数据和外部信号数据。
- **撮合引擎模块 (`lob_backtest/engine/`)**: 模拟交易所的撮合逻辑，是整个回测系统的核心。
- **性能分析模块 (`lob_backtest/analysis/`)**: 计算各种性能指标，并提供可视化图表。
- **工具模块 (`lob_backtest/utils/`)**: 提供配置管理等通用功能。
- **主入口 (`main.py`)**: 系统的启动文件，负责协调各个模块完成回测流程。

## 2. 源代码路径与关键文件

- `main.py`: 项目主入口，负责初始化和运行回测。
- `lob_backtest/backtester.py`: 核心回测器类，整合了数据加载、撮合和分析的完整流程。
- `lob_backtest/data/lob_data_loader.py`: 十档盘口数据加载器。
- `lob_backtest/data/signal_data_loader.py`: 交易信号数据加载器。
- `lob_backtest/engine/matching_engine.py`: 核心撮合引擎，使用 Numba 进行了性能优化。
- `lob_backtest/engine/order_book.py`: 订单簿模拟器。
- `lob_backtest/analysis/performance_metrics.py`: 性能指标计算。
- `lob_backtest/analysis/visualization.py`: 静态图表生成。
- `lob_backtest/analysis/interactive_chart.py`: 交互式图表生成。
- `lob_backtest/utils/config.py`: 配置文件 `backtest_config.yaml` 的加载和管理。
- `config/backtest_config.yaml`: 全局配置文件，用于调整回测参数。
- `参考/`: 包含用于数据格式化和可视化的参考脚本。

## 3. 关键技术决策与设计模式

- **配置驱动**: 整个回测流程由 `config/backtest_config.yaml` 文件驱动，实现了代码与参数的分离，便于调优。
- **模块化设计**: 各个核心功能（数据、引擎、分析）被拆分为独立的模块，提高了代码的可维护性和可扩展性。
- **JIT编译优化**: 核心的撮合循环 (`matching_engine.py`) 使用 Numba 的 `@jit` 装饰器进行即时编译，大幅提升了计算性能。
- **向量化操作**: 优先使用 Pandas 和 NumPy 的向量化操作处理数据，避免了低效的循环。

## 4. 数据流

1.  **启动**: `main.py` 启动，加载 `config/backtest_config.yaml` 配置。
2.  **数据加载**: `LOBBacktester` 调用 `lob_data_loader` 和 `signal_data_loader` 加载并预处理数据。
3.  **撮合循环**: `LOBBacktester` 将处理好的数据送入 `matching_engine`。引擎逐个tick模拟交易，更新订单簿状态和账户净值。
4.  **结果分析**: 撮合结束后，`LOBBacktester` 将交易记录和净值数据传递给 `performance_metrics` 计算指标。
5.  **可视化**: `visualization` 和 `interactive_chart` 模块根据分析结果生成静态或交互式图表。
6.  **输出**: 所有结果（CSV, JSON, PNG）被保存到 `results/` 目录下。