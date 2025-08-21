# {{ AURA-X | Action: Add | Reason: 创建配置管理模块，支持YAML配置文件加载 | Approval: Cunzhi(ID:1735632000) }}
"""
配置管理模块
负责加载和管理回测系统的所有配置参数
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class BacktestConfig:
    """回测配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认使用项目根目录下的config/backtest_config.yaml
        """
        if config_path is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "backtest_config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，支持 'data.timezone' 格式
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            path: 保存路径，默认覆盖原文件
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """数据配置"""
        return self.get('data', {})
    
    @property
    def signal_config(self) -> Dict[str, Any]:
        """信号配置"""
        return self.get('signal', {})
    
    @property
    def trading_config(self) -> Dict[str, Any]:
        """交易配置"""
        return self.get('trading', {})
    
    @property
    def matching_config(self) -> Dict[str, Any]:
        """撮合引擎配置"""
        return self.get('matching', {})
    
    @property
    def analysis_config(self) -> Dict[str, Any]:
        """性能分析配置"""
        return self.get('analysis', {})
    
    @property
    def visualization_config(self) -> Dict[str, Any]:
        """可视化配置"""
        return self.get('visualization', {})
