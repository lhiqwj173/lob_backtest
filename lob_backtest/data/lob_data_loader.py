# {{ AURA-X | Action: Add | Reason: 创建十档盘口数据加载器，基于参考代码进行优化 | Approval: Cunzhi(ID:1735632000) }}
"""
十档盘口数据加载器
负责加载、清理和格式化十档盘口CSV数据
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path


class LOBDataLoader:
    """十档盘口数据加载器"""
    
    def load_data(self, data_path: str, stock_id: str, start_timestamp: int, end_timestamp: int) -> Optional[pd.DataFrame]:
        """
        从指定目录加载多日十档盘口数据

        Args:
            data_path: 数据根目录 (e.g., "D:/L2_DATA_T0_ETF/his_data")
            stock_id: 标的代码 (e.g., "513180")
            start_timestamp: 开始时间戳
            end_timestamp: 结束时间戳

        Returns:
            合并并清理后的DataFrame，如果无有效数据则返回None
        """
        data_root = Path(data_path)
        if not data_root.is_dir():
            print(f"数据根目录不存在: {data_root}")
            return None

        all_data = []
        start_date = pd.to_datetime(start_timestamp, unit='s').date()
        end_date = pd.to_datetime(end_timestamp, unit='s').date()

        for date_dir in sorted(data_root.iterdir()):
            if not date_dir.is_dir():
                continue
            
            try:
                dir_date = pd.to_datetime(date_dir.name, format='%Y%m%d').date()
                if not (start_date <= dir_date <= end_date):
                    continue
            except ValueError:
                continue

            file_path = date_dir / stock_id / "十档盘口.csv"
            if not file_path.exists():
                continue

            print(f"正在加载数据: {file_path}")
            daily_data = self._load_single_day_data(str(file_path))
            if daily_data is not None:
                all_data.append(daily_data)

        if not all_data:
            print("在指定时间范围内未找到有效数据")
            return None

        # 合并所有数据
        full_data = pd.concat(all_data, ignore_index=True)
        full_data = full_data.sort_values('时间').reset_index(drop=True)

        # 转换为时间戳并进行最终过滤
        # 移除毫秒，确保时间戳与信号数据对齐
        full_data['时间'] = full_data['时间'].dt.floor('S')
        full_data['timestamp'] = full_data['时间'].apply(lambda x: int(x.timestamp()))
        
        full_data = full_data[
            (full_data['timestamp'] >= start_timestamp) &
            (full_data['timestamp'] <= end_timestamp)
        ]

        return full_data.reset_index(drop=True)

    def _load_single_day_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """加载单日数据并进行初步处理"""
        try:
            encodings = ['gbk', 'utf-8', 'utf-8-sig', 'gb2312']
            data = None
            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if data is None:
                raise ValueError("无法使用任何支持的编码格式读取文件")

            data = data.drop_duplicates(keep='first')
            # 将时间字符串转换为无时区的datetime对象。
            # 后续调用 .timestamp() 时，Pandas/Python会隐式使用本地系统时区（北京时间）
            # 来计算Unix时间戳，这符合项目要求。
            data['时间'] = pd.to_datetime(data['时间'], errors='coerce')
            data = self._filter_trading_hours(data, "09:30", "15:00")

            if len(data) == 0 or self._check_limit_up_down(data):
                return None

            self._validate_data_quality(data)
            data = self._clean_data(data)
            
            return data

        except Exception as e:
            print(f"加载单日数据失败 {file_path}: {e}")
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
