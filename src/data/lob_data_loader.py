# {{ AURA-X | Action: Add | Reason: 创建十档盘口数据加载器，基于参考代码进行优化 | Approval: Cunzhi(ID:1735632000) }}
"""
十档盘口数据加载器
负责加载、清理和格式化十档盘口CSV数据
"""

import os
import pandas as pd
import numpy as np
import pytz
from typing import Optional, List, Tuple
from pathlib import Path


class LOBDataLoader:
    """十档盘口数据加载器"""
    
    def __init__(self, timezone: str = "Asia/Shanghai"):
        """
        初始化数据加载器
        
        Args:
            timezone: 时区设置，强制使用北京时间
        """
        self.timezone = pytz.timezone(timezone)
    
    def load_data(self, file_path: str, 
                  begin_time: str = "09:30", 
                  end_time: str = "15:00") -> Optional[pd.DataFrame]:
        """
        加载十档盘口数据
        
        Args:
            file_path: CSV文件路径
            begin_time: 开始时间 (HH:MM格式)
            end_time: 结束时间 (HH:MM格式)
            
        Returns:
            清理后的DataFrame，如果数据无效则返回None
        """
        if not os.path.exists(file_path):
            print(f"数据文件不存在: {file_path}")
            return None
        
        try:
            # {{ AURA-X | Action: Modify | Reason: 增强编码兼容性，自动尝试多种编码格式 | Approval: Cunzhi(ID:1735632000) }}
            # 尝试多种编码格式读取CSV数据
            encodings = ['gbk', 'utf-8', 'utf-8-sig', 'gb2312']
            data = None

            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, encoding=encoding)
                    print(f"成功使用 {encoding} 编码读取数据")
                    break
                except UnicodeDecodeError:
                    continue

            if data is None:
                raise ValueError(f"无法使用任何支持的编码格式读取文件: {file_path}")
            
            # 删除完全重复的行
            data = data.drop_duplicates(keep='first')
            
            # 格式化时间列，强制使用北京时间
            data['时间'] = pd.to_datetime(data['时间'])
            data['时间'] = data['时间'].dt.tz_localize(self.timezone, ambiguous='infer')
            
            # 时间过滤：交易时间段
            data = self._filter_trading_hours(data, begin_time, end_time)
            
            if len(data) == 0:
                print(f"时间过滤后无数据: {file_path}")
                return None
            
            # 涨跌停检查
            if self._check_limit_up_down(data):
                print(f"存在涨跌停，跳过: {file_path}")
                return None
            
            # 数据质量检查
            self._validate_data_quality(data)
            
            # 数据清理
            data = self._clean_data(data)
            
            # 添加时间戳列（UTC秒级）
            data['timestamp'] = data['时间'].apply(
                lambda x: int(x.timestamp())
            )
            
            return data.reset_index(drop=True)
            
        except Exception as e:
            print(f"加载数据失败 {file_path}: {e}")
            return None
    
    def _filter_trading_hours(self, data: pd.DataFrame, 
                            begin_time: str, end_time: str) -> pd.DataFrame:
        """过滤交易时间段"""
        # 基本时间过滤
        begin_dt = pd.to_datetime(begin_time).time()
        end_dt = pd.to_datetime(end_time).time()
        
        data = data[
            (data["时间"].dt.time >= begin_dt) & 
            (data["时间"].dt.time < end_dt)
        ].reset_index(drop=True)
        
        # 排除午休时间 (11:30-13:00)
        data = data[
            (data["时间"].dt.time <= pd.to_datetime('11:30:00').time()) | 
            (data["时间"].dt.time > pd.to_datetime('13:00:00').time())
        ].reset_index(drop=True)
        
        return data
    
    def _check_limit_up_down(self, data: pd.DataFrame) -> bool:
        """
        检查是否存在涨跌停
        
        Returns:
            True if 存在涨跌停
        """
        # 涨停：卖1价为0且卖1量为0
        limit_up = ((data['卖1价'] == 0) & (data['卖1量'] == 0)).any()
        # 跌停：买1价为0且买1量为0  
        limit_down = ((data['买1价'] == 0) & (data['买1量'] == 0)).any()
        
        return limit_up or limit_down
    
    def _validate_data_quality(self, data: pd.DataFrame) -> None:
        """验证数据质量，买卖1价不允许NaN"""
        error_msgs = []
        
        if data['卖1价'].isna().any():
            error_msgs.append("卖1价存在 NaN")
        if data['买1价'].isna().any():
            error_msgs.append("买1价存在 NaN")
            
        if error_msgs:
            raise ValueError("; ".join(error_msgs))
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清理"""
        # 2-10档位价格NaN填充
        for i in range(2, 11):
            # 买价：使用上一档位价格-0.001填充
            data.loc[:, f'买{i}价'] = data[f'买{i}价'].fillna(
                data[f'买{i-1}价'] - 0.001
            )
            # 卖价：使用上一档位价格+0.001填充
            data.loc[:, f'卖{i}价'] = data[f'卖{i}价'].fillna(
                data[f'卖{i-1}价'] + 0.001
            )
        
        # 盘口量：NaN和0都用1填充
        vol_cols = [col for col in data.columns if '量' in col]
        data[vol_cols] = data[vol_cols].replace(0, np.nan).fillna(1)
        
        # 删除不需要的列
        if '总卖' in data.columns:
            data = data.drop(columns=['总卖'])
        if '总买' in data.columns:
            data = data.drop(columns=['总买'])
        
        return data
    
    def get_orderbook_snapshot(self, data: pd.DataFrame, timestamp: int) -> Optional[dict]:
        """
        获取指定时间戳的订单簿快照
        
        Args:
            data: LOB数据
            timestamp: 时间戳
            
        Returns:
            订单簿快照字典，包含买卖各10档价格和数量
        """
        # 找到最接近的时间戳
        closest_idx = (data['timestamp'] - timestamp).abs().idxmin()
        row = data.iloc[closest_idx]
        
        snapshot = {
            'timestamp': row['timestamp'],
            'bids': [],  # [(price, volume), ...]
            'asks': []   # [(price, volume), ...]
        }
        
        # 构建买盘（从高到低）
        for i in range(1, 11):
            price = row[f'买{i}价']
            volume = row[f'买{i}量']
            if pd.notna(price) and pd.notna(volume) and volume > 0:
                snapshot['bids'].append((float(price), float(volume)))
        
        # 构建卖盘（从低到高）
        for i in range(1, 11):
            price = row[f'卖{i}价']
            volume = row[f'卖{i}量']
            if pd.notna(price) and pd.notna(volume) and volume > 0:
                snapshot['asks'].append((float(price), float(volume)))
        
        return snapshot
