# {{ AURA-X | Action: Modify | Reason: 重构主入口文件，实现配置驱动和命令行参数支持 | Approval: Cunzhi(ID:1735632000) }}
"""
订单簿级别撮合回测系统主入口
LOB (Limit Order Book) Backtesting System Main Entry
"""

import argparse
import logging
import sys
from lob_backtest.backtester import LOBBacktester
from lob_backtest.utils.config import BacktestConfig

def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def main():
    """主函数"""
    setup_logging()

    parser = argparse.ArgumentParser(description="订单簿级别回测系统")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='指定配置文件路径。如果未提供，则使用默认路径 (config/backtest_config.yaml)。'
    )
    args = parser.parse_args()

    try:
        logging.info("初始化回测系统...")
        # 使用指定的配置文件路径（如果提供的话）初始化回测器
        backtester = LOBBacktester(config_path=args.config)
        config = backtester.config

        # 从配置中获取数据路径和交易对信息
        # 检查必要的配置
        if not config.get('data.lob_data_path') or not config.get('data.signal_data_path'):
            logging.error("配置文件中缺少 'lob_data_path' 或 'signal_data_path'。")
            sys.exit(1)

        logging.info(f"成功加载配置: {config.config_path}")

        # 运行回测 (无需传递参数)
        backtester.run_backtest()

    except FileNotFoundError as e:
        logging.error(f"配置或数据文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"回测过程中发生未知错误: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
