# 安装指南

## 系统要求

### 硬件要求
- **CPU**: Intel i5 或 AMD Ryzen 5 及以上
- **内存**: 8GB RAM (推荐 16GB)
- **存储**: 10GB 可用磁盘空间
- **网络**: 用于下载依赖包

### 软件要求
- **操作系统**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8 - 3.11 (推荐 3.9)
- **Git**: 用于代码管理 (可选)

## 安装步骤

### 1. Python环境准备

#### Windows系统
```bash
# 下载并安装Python 3.9
# 访问 https://www.python.org/downloads/
# 确保勾选 "Add Python to PATH"

# 验证安装
python --version
pip --version
```

#### macOS系统
```bash
# 使用Homebrew安装
brew install python@3.9

# 或下载官方安装包
# 访问 https://www.python.org/downloads/

# 验证安装
python3 --version
pip3 --version
```

#### Linux系统
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-pip python3.9-venv

# CentOS/RHEL
sudo yum install python39 python39-pip

# 验证安装
python3.9 --version
pip3.9 --version
```

### 2. 获取项目代码

#### 方式一: 直接下载
1. 下载项目压缩包
2. 解压到目标目录
3. 进入项目目录

#### 方式二: Git克隆 (如果有Git仓库)
```bash
git clone <repository-url>
cd lob_backtest
```

### 3. 创建虚拟环境 (推荐)

```bash
# 创建虚拟环境
python -m venv lob_env

# 激活虚拟环境
# Windows
lob_env\Scripts\activate

# macOS/Linux
source lob_env/bin/activate

# 验证虚拟环境
which python  # 应该指向虚拟环境中的python
```

### 4. 安装依赖包

```bash
# 确保在项目根目录下
cd lob_backtest

# 升级pip
python -m pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

### 5. 验证安装

```bash
# 运行测试脚本
python test_with_sample_data.py
```

如果看到以下输出，说明安装成功：
```
✅ 所有系统组件验证通过
正在生成模拟LOB数据...
正在生成模拟信号数据...
✅ 回测成功完成!
```

## 依赖包说明

### 核心依赖
```
pandas>=2.0.0          # 数据处理
numpy>=1.24.0           # 数值计算
pytz>=2023.3            # 时区处理
numba>=0.58.0           # JIT编译优化
PyYAML>=6.0             # 配置文件解析
matplotlib>=3.7.0       # 基础绘图
```

### 可视化依赖
```
lightweight-charts>=2.0.0  # 交互式图表
seaborn>=0.12.0            # 统计图表 (可选)
plotly>=5.15.0             # 交互式图表 (可选)
```

### 开发依赖
```
pytest>=7.4.0          # 单元测试
black>=23.0.0           # 代码格式化
flake8>=6.0.0           # 代码检查
jupyter>=1.0.0          # 笔记本环境 (可选)
```

## 常见安装问题

### Q1: pip安装失败
**问题**: `pip install` 命令执行失败

**解决方案**:
```bash
# 升级pip
python -m pip install --upgrade pip

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 或使用阿里云镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### Q2: Numba安装失败
**问题**: Numba编译器安装失败

**解决方案**:
```bash
# Windows: 安装Visual Studio Build Tools
# 下载地址: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# macOS: 安装Xcode命令行工具
xcode-select --install

# Linux: 安装编译工具
sudo apt install build-essential  # Ubuntu/Debian
sudo yum groupinstall "Development Tools"  # CentOS/RHEL

# 重新安装numba
pip install numba --no-cache-dir
```

### Q3: 内存不足
**问题**: 运行时内存不足

**解决方案**:
1. 关闭其他程序释放内存
2. 减少测试数据量
3. 增加虚拟内存
4. 升级物理内存

### Q4: 权限问题
**问题**: 没有写入权限

**解决方案**:
```bash
# Windows: 以管理员身份运行命令提示符
# macOS/Linux: 使用sudo (不推荐) 或修改目录权限
chmod 755 /path/to/lob_backtest
```

### Q5: Python版本不兼容
**问题**: Python版本过低或过高

**解决方案**:
```bash
# 使用pyenv管理多个Python版本
# 安装pyenv
curl https://pyenv.run | bash

# 安装Python 3.9
pyenv install 3.9.18
pyenv local 3.9.18

# 验证版本
python --version
```

## 性能优化配置

### 1. Numba优化
```bash
# 设置Numba缓存目录
export NUMBA_CACHE_DIR=/tmp/numba_cache

# 启用并行计算
export NUMBA_NUM_THREADS=4
```

### 2. 内存优化
```bash
# 设置Pandas内存使用限制
export PANDAS_MEMORY_LIMIT=4GB

# 启用内存映射
export PANDAS_USE_MEMORY_MAP=1
```

### 3. 多进程配置
```python
# 在配置文件中设置
import multiprocessing
max_workers = multiprocessing.cpu_count() - 1
```

## 开发环境配置

### 1. IDE配置
推荐使用以下IDE:
- **PyCharm**: 专业Python IDE
- **VS Code**: 轻量级编辑器
- **Jupyter Lab**: 交互式开发

### 2. 代码格式化
```bash
# 安装开发工具
pip install black flake8 isort

# 格式化代码
black .
isort .
flake8 .
```

### 3. 调试配置
```python
# 在代码中添加调试断点
import pdb; pdb.set_trace()

# 或使用IDE的调试功能
```

## 部署配置

### 1. 生产环境
```bash
# 创建生产环境配置
cp config/backtest_config.yaml config/production_config.yaml

# 修改生产环境参数
# - 调整数据路径
# - 设置日志级别
# - 配置输出目录
```

### 2. 容器化部署
```dockerfile
# Dockerfile示例
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### 3. 服务化部署
```bash
# 创建systemd服务文件
sudo nano /etc/systemd/system/lob-backtest.service

# 启动服务
sudo systemctl enable lob-backtest
sudo systemctl start lob-backtest
```

## 卸载指南

### 1. 删除虚拟环境
```bash
# 退出虚拟环境
deactivate

# 删除虚拟环境目录
rm -rf lob_env
```

### 2. 清理项目文件
```bash
# 删除项目目录
rm -rf lob_backtest

# 清理pip缓存
pip cache purge
```

### 3. 清理系统配置
```bash
# 删除环境变量 (如果设置了)
unset NUMBA_CACHE_DIR
unset PANDAS_MEMORY_LIMIT

# 清理临时文件
rm -rf /tmp/numba_cache
```

---

如果在安装过程中遇到其他问题，请参考 [常见问题文档](FAQ.md) 或联系技术支持。
