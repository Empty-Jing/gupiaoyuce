"""
LSTM 预测模型定义
输入: [batch, seq_len=20, features=7]
features: close, volume, macd, rsi_6, kdj_k, kdj_d, sentiment_score
输出: 方向概率(3类) / 价格区间 / 预测收益率
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    基于 LSTM 的股票多任务预测模型。
    
    输入时序窗口为 20 天，每天 7 维特征，输出包含：
    - direction_prob: UP/DOWN/FLAT 三分类概率
    - price_range: [预测最低价, 预测最高价] 相对变化
    - predicted_return: 下一日预测收益率
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # 方向预测头: UP / DOWN / FLAT → 3 类
        self.direction_head = nn.Linear(hidden_size, 3)
        # 价格区间预测头: [price_low, price_high] 相对变化
        self.price_range_head = nn.Linear(hidden_size, 2)
        # 收益率预测头: predicted_return
        self.return_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> dict:
        """
        前向传播。

        参数:
            x: 形状 [batch, seq_len=20, features=7]

        返回:
            dict 包含三个 tensor:
            - direction_prob: [batch, 3] softmax 概率，顺序 UP/DOWN/FLAT
            - price_range: [batch, 2] 预测价格区间 [low, high]
            - predicted_return: [batch, 1] 预测收益率
        """
        # lstm_out: [batch, seq_len, hidden_size]
        # h_n: [num_layers, batch, hidden_size]
        lstm_out, (h_n, _) = self.lstm(x)
        # 取最后一层的隐藏状态作为序列表示
        last_hidden = h_n[-1]  # [batch, hidden_size]

        return {
            "direction_prob": torch.softmax(self.direction_head(last_hidden), dim=-1),  # [batch, 3]
            "price_range": self.price_range_head(last_hidden),  # [batch, 2]
            "predicted_return": self.return_head(last_hidden),  # [batch, 1]
        }
