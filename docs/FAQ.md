# 常见问题解答 (FAQ)

## 安装相关问题

### Q1: 安装依赖包时出现编译错误
**A:** 这通常是由于缺少编译工具导致的。

**解决方案:**
- **Windows**: 安装 Visual Studio Build Tools
- **macOS**: 运行 `xcode-select --install`
- **Linux**: 安装 `build-essential` 包

```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev

# CentOS/RHEL  
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

### Q2: Numba安装失败，提示LLVM相关错误
**A:** Numba需要LLVM支持，可能需要降级或使用conda安装。

```bash
# 方案1: 使用conda安装
conda install numba

# 方案2: 安装特定版本
pip install numba==0.58.1

# 方案3: 从源码编译
pip install numba --no-binary numba
```

### Q3: 在Apple M1/M2芯片上安装失败
**A:** M1/M2芯片需要特殊处理。

```bash
# 使用Rosetta模式
arch -x86_64 pip install -r requirements.txt

# 或使用conda-forge
conda install -c conda-forge numba pandas numpy
```

## 数据相关问题

### Q4: 数据加载时提示编码错误
**A:** 中文数据文件可能使用GBK编码。

```python
# 在LOBDataLoader中指定编码
data = pd.read_csv(file_path, encoding='gbk')

# 或转换文件编码
import codecs
with codecs.open('input.csv', 'r', 'gbk') as f:
    content = f.read()
with codecs.open('output.csv', 'w', 'utf-8') as f:
    f.write(content)
```

### Q5: 时间戳格式不匹配
**A:** 确保时间戳格式正确。

```python
# LOB数据时间格式
"2025-06-03 09:30:05"  # 正确格式

# 信号数据时间戳
1725865015  # UTC秒级时间戳

# 转换示例
import pandas as pd
df['时间'] = pd.to_datetime(df['时间'], format='%Y-%m-%d %H:%M:%S')
```

### Q6: 数据量太大导致内存不足
**A:** 采用分批处理策略。

```python
# 分时间段处理
def process_by_chunks(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # 处理每个chunk
        yield process_chunk(chunk)

# 减少数据精度
df = df.astype({
    '卖1价': 'float32',
    '买1价': 'float32',
    '卖1量': 'int32',
    '买1量': 'int32'
})
```

## 回测相关问题

### Q7: 回测结果不合理，收益率异常高或异常低
**A:** 检查以下配置参数。

```yaml
# 检查手续费设置
trading:
  commission_rate: 5e-5  # 万分之五，不是5%

# 检查初始资金
trading:
  initial_capital: 1000000  # 100万，不是100元

# 检查信号阈值
signal:
  predict_threshold: "max"  # 或0.6等合理数值
```

### Q8: 交易次数为0，没有产生任何交易
**A:** 检查信号生成逻辑。

```python
# 检查信号数据
print(signal_data['signal'].value_counts())
print(signal_data[['0', '1']].describe())

# 检查时间对齐
print(f"LOB时间范围: {lob_data['timestamp'].min()} - {lob_data['timestamp'].max()}")
print(f"信号时间范围: {signal_data['timestamp'].min()} - {signal_data['timestamp'].max()}")

# 调整阈值策略
config.set('signal.predict_threshold', 0.5)  # 降低阈值
```

### Q9: 回测速度很慢
**A:** 优化性能配置。

```python
# 减少数据频率
lob_data = lob_data[::3]  # 每3条取1条

# 使用更少的档位
# 只使用前5档数据进行撮合

# 启用Numba缓存
import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
```

### Q10: 撮合结果不准确
**A:** 检查订单簿数据质量。

```python
# 检查买卖价格关系
invalid_spread = data['卖1价'] <= data['买1价']
print(f"价格倒挂数量: {invalid_spread.sum()}")

# 检查数量为0的情况
zero_volume = (data['卖1量'] == 0) | (data['买1量'] == 0)
print(f"零数量记录: {zero_volume.sum()}")

# 检查价格跳跃
price_change = data['卖1价'].pct_change().abs()
print(f"异常价格变动: {(price_change > 0.1).sum()}")
```

## 配置相关问题

### Q11: 配置文件修改后不生效
**A:** 确保配置文件路径和格式正确。

```python
# 检查配置文件路径
config = BacktestConfig('config/backtest_config.yaml')

# 验证配置加载
print(config.get('trading.commission_rate'))

# 保存修改后的配置
config.save()
```

### Q12: 自定义配置参数不被识别
**A:** 确保配置键名正确。

```yaml
# 正确的配置结构
trading:
  commission_rate: 5e-5    # 正确
  
# 错误的配置
trading:
  commision_rate: 5e-5     # 拼写错误
```

## 输出相关问题

### Q13: 可视化图表显示异常或不显示
**A:** 检查matplotlib后端配置。

```python
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 或安装GUI后端
pip install PyQt5  # 或 tkinter
```

### Q14: 输出文件权限错误
**A:** 检查输出目录权限。

```bash
# 创建输出目录
mkdir -p results
chmod 755 results

# 或修改输出路径
config.set('analysis.output_dir', '/tmp/results')
```

### Q15: CSV文件中文显示乱码
**A:** 指定正确的编码格式。

```python
# 保存时指定编码
df.to_csv('output.csv', encoding='utf-8-sig', index=False)

# 读取时指定编码
df = pd.read_csv('output.csv', encoding='utf-8-sig')
```

## 性能相关问题

### Q16: 系统运行时CPU使用率过高
**A:** 调整并行处理参数。

```python
import os
# 限制Numba线程数
os.environ['NUMBA_NUM_THREADS'] = '2'

# 限制NumPy线程数
os.environ['OMP_NUM_THREADS'] = '2'
```

### Q17: 内存使用持续增长
**A:** 启用垃圾回收和内存监控。

```python
import gc

# 定期清理内存
if i % 1000 == 0:
    gc.collect()

# 监控内存使用
import psutil
process = psutil.Process()
print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

## 扩展开发问题

### Q18: 如何添加自定义性能指标？
**A:** 在PerformanceAnalyzer中扩展。

```python
class CustomPerformanceAnalyzer(PerformanceAnalyzer):
    def calculate_custom_metric(self, returns):
        # 实现自定义指标
        return custom_value
    
    def calculate_all_metrics(self, asset_history, trades, initial_capital):
        metrics = super().calculate_all_metrics(asset_history, trades, initial_capital)
        metrics['custom_metric'] = self.calculate_custom_metric(returns)
        return metrics
```

### Q19: 如何修改撮合逻辑？
**A:** 继承MatchingEngine类。

```python
class CustomMatchingEngine(MatchingEngine):
    def _execute_buy_order(self, order):
        # 实现自定义撮合逻辑
        # 调用父类方法或完全重写
        return super()._execute_buy_order(order)
```

### Q20: 如何集成其他数据源？
**A:** 创建新的数据加载器。

```python
class CustomDataLoader:
    def load_data(self, source_config):
        # 实现自定义数据加载逻辑
        # 返回标准格式的DataFrame
        return standardized_data
```

## 错误诊断

### 常见错误代码
- **ERR_001**: 数据文件不存在
- **ERR_002**: 数据格式错误
- **ERR_003**: 内存不足
- **ERR_004**: 配置参数错误
- **ERR_005**: 撮合引擎异常

### 调试技巧
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用调试器
import pdb; pdb.set_trace()

# 保存中间结果
intermediate_data.to_csv('debug_data.csv')
```

## 获取帮助

### 1. 查看日志文件
```bash
# 检查系统日志
tail -f logs/backtest_*.log

# 检查错误日志
grep ERROR logs/backtest_*.log
```

### 2. 收集系统信息
```python
import sys, platform, pandas, numpy, numba

print(f"Python版本: {sys.version}")
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Pandas版本: {pandas.__version__}")
print(f"NumPy版本: {numpy.__version__}")
print(f"Numba版本: {numba.__version__}")
```

### 3. 提供错误信息
在寻求帮助时，请提供：
- 完整的错误堆栈跟踪
- 系统环境信息
- 数据样本（脱敏后）
- 配置文件内容
- 重现步骤

---

如果以上FAQ没有解决您的问题，请联系技术支持或查阅详细的技术文档。
