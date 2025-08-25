# {{ AURA-X | Action: Add | Reason: 创建深度学习信号数据加载器，处理预测结果CSV | Approval: Cunzhi(ID:1735632000) }}
"""
深度学习信号数据加载器
负责加载和处理深度学习预测结果CSV数据
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List


class SignalDataLoader:
    """深度学习信号数据加载器"""
    
    def load_signal_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        加载深度学习信号数据
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            信号数据DataFrame
        """
        try:
            data = pd.read_csv(file_path)
            
            # 验证必要列
            required_cols = ['timestamp', 'target', 'has_pos', '0', '1']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"缺少必要列: {missing_cols}")
            
            # 添加时间列（北京时间）
            # 将Unix时间戳转换为无时区的datetime对象（本地时间）
            data['datetime'] = pd.to_datetime(data['timestamp'], origin='1970-01-01 08:00:00', unit='s')
            
            return data.sort_values('timestamp').reset_index(drop=True)
            
        except Exception as e:
            import traceback
            print(f"加载信号数据失败 {file_path}")
            print(f"异常堆栈信息:\n{traceback.format_exc()}")
            return None
    
    def generate_signals(self, data: pd.DataFrame, 
                        source: str = "predict",
                        threshold_strategy: Union[str, float] = "max") -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 原始信号数据
            source: 信号源 ("target" 或 "predict")
            threshold_strategy: 阈值策略 ("max" 或具体数值)
            
        Returns:
            包含交易信号的DataFrame
        """
        result = data.copy()
        
        if source == "target":
            # 使用真实标签作为信号
            result['signal'] = result['target']
        else:
            # 使用预测结果生成信号
            result['signal'] = self._apply_threshold_strategy(
                result, threshold_strategy
            )
        
        return result
    
    def _apply_threshold_strategy(self, data: pd.DataFrame, 
                                strategy: Union[str, float]) -> pd.Series:
        """
        应用阈值策略生成预测信号
        
        Args:
            data: 包含预测概率的数据
            strategy: 阈值策略
            
        Returns:
            预测信号序列
        """
        prob_cols = ['0', '1']  # 概率列
        
        if strategy == "max":
            # 取最大概率对应的动作
            return data[prob_cols].idxmax(axis=1).astype(int)
        
        elif isinstance(strategy, (int, float)):
            # 使用固定阈值
            threshold = float(strategy)
            
            # 如果概率大于阈值，则选择该动作，否则选择概率最大的
            signals = []
            for _, row in data.iterrows():
                if row['1'] >= threshold:
                    signals.append(1)
                elif row['0'] >= threshold:
                    signals.append(0)
                else:
                    # 都不满足阈值，选择概率最大的
                    signals.append(int(row[prob_cols].idxmax()))
            
            return pd.Series(signals, index=data.index)
        
        else:
            raise ValueError(f"不支持的阈值策略: {strategy}")
    
    def align_with_lob_data(self, signal_data: pd.DataFrame, 
                           lob_timestamps: List[int]) -> pd.DataFrame:
        """
        将信号数据与LOB数据时间戳对齐
        
        Args:
            signal_data: 信号数据
            lob_timestamps: LOB数据时间戳列表
            
        Returns:
            对齐后的信号数据
        """
        # 创建LOB时间戳DataFrame
        lob_df = pd.DataFrame({'timestamp': lob_timestamps})
        
        # 使用最近邻方法对齐
        # 对于每个LOB时间戳，找到最近的信号时间戳
        aligned_signals = []
        
        for lob_ts in lob_timestamps:
            # 找到最接近的信号时间戳
            time_diffs = np.abs(signal_data['timestamp'] - lob_ts)
            closest_idx = time_diffs.idxmin()
            
            # 检查时间差是否在合理范围内（比如60秒）
            if time_diffs.iloc[closest_idx] <= 60:
                aligned_signals.append(signal_data.iloc[closest_idx])
            else:
                # 如果时间差太大，使用空信号
                empty_signal = signal_data.iloc[0].copy()
                empty_signal['timestamp'] = lob_ts
                empty_signal['signal'] = np.nan
                aligned_signals.append(empty_signal)
        
        result = pd.DataFrame(aligned_signals).reset_index(drop=True)
        return result
    
