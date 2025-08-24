# 常见问题解答 (FAQ)

## 安装相关问题

### Q1: 安装依赖包时出现编译错误
**A:** 这通常是由于缺少编译工具导致的。

**解决方案:**
- **Windows**: 安装 Visual Studio Build Tools
- **macOS**: 运行 `xcode-select --install`
- **Linux**: 安装 `build-essential` 包

### Q2: Numba安装失败，提示LLVM相关错误
**A:** Numba需要LLVM支持，可以尝试使用 `conda` 安装或指定兼容的版本。

```bash
# 方案1: 使用conda安装 (推荐)
conda install numba

# 方案2: 安装特定版本
pip install numba==0.58.1
```

### Q3: 在Apple M1/M2芯片上安装失败
**A:** M1/M2芯片建议使用 `miniforge` 环境进行安装。

```bash
# 使用 miniforge 安装
conda install -c conda-forge numba pandas numpy
```

## 数据相关问题

### Q4: 数据加载时提示编码错误
**A:** 确保数据文件为 `UTF-8` 编码。如果为其他编码，请先进行转换。

```python
# 转换文件编码示例 (GBK to UTF-8)
import codecs
with codecs.open('input.csv', 'r', 'gbk') as f:
    content = f.read()
with codecs.open('output.csv', 'w', 'utf-8') as f:
    f.write(content)
```

### Q5: 时间戳格式不匹配
**A:** 确保LOB数据的时间格式为 `%Y-%m-%d %H:%M:%S`，信号数据的时间戳为秒级UTC时间戳。

### Q6: 数据量太大导致内存不足
**A:** 对于大规模数据，建议：
- **分批处理**: 将数据按天或按周分割成小文件进行回测。
- **减少精度**: 将 `float64` 转换为 `float32`，`int64` 转换为 `int32`。
- **增加内存**: 升级硬件或使用更高配置的云服务器。

## 回测相关问题

### Q7: 回测结果不合理，收益率异常
**A:** 请检查以下配置：
- `commission_rate`: 手续费是否设置得过低或过高。
- `initial_capital`: 初始资金是否符合实际情况。
- `predict_threshold`: 信号阈值是否过于宽松或严格。

### Q8: 交易次数为0
**A:** 请检查：
- **时间对齐**: LOB数据和信号数据的时间范围是否有重叠。
- **信号质量**: 信号的预测概率是否能够超过设定的阈值。
- **资金不足**: 初始资金是否足够执行第一笔交易。

### Q9: 回测速度很慢
**A:** 优化建议：
- **减少数据频率**: 如果不需要秒级精度，可以对数据进行降采样。
- **启用Numba缓存**: 设置 `NUMBA_CACHE_DIR` 环境变量可以加速二次运行。
- **硬件升级**: 使用更快的CPU和SSD硬盘。

### Q10: 回测结果与实盘差异大
**A:** 这是常见现象，可能的原因包括：
- **滑点模型**: 真实滑点比模型更复杂。
- **延迟**: 真实世界的网络延迟和交易延迟是变化的。
- **数据质量**: 回测数据可能无法完全反映真实市场的流动性。
- **过拟合**: 策略可能在历史数据上表现很好，但在未来数据上表现不佳。

## 可视化相关问题

### Q11: 静态图表中文显示乱码
**A:** 确保 `matplotlib` 配置了支持中文的字体。

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
```

### Q12: 交互式图表无法显示
**A:** `lightweight-charts` 需要在支持JavaScript的环境中运行。
- **检查浏览器**: 确保您的默认浏览器支持最新的JavaScript特性。
- **查看日志**: 检查运行时是否有任何与图表相关的错误日志。
- **网络问题**: 如果图表需要加载外部资源，请检查网络连接。

---

如果以上FAQ没有解决您的问题，请联系技术支持或查阅详细的技术文档。
