"""
模型训练入口
提供 LSTM 和 XGBoost 模型的训练函数
如果数据库无真实数据，自动使用模拟数据进行训练
"""

import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from typing import Callable, Optional
from sklearn.preprocessing import StandardScaler

from app.models.lstm_model import LSTMModel
from app.models.xgboost_model import XGBoostPredictor
from app.core.indicator_engine import IndicatorEngine

logger = logging.getLogger(__name__)


# 模型保存根目录
MODEL_DIR = Path("data/models")

# LSTM 输入特征列（顺序固定）
LSTM_FEATURES = ["close", "volume", "macd", "rsi_6", "kdj_k", "kdj_d", "sentiment_score"]
SEQ_LEN = 20  # 时序窗口长度


def _generate_mock_data(n_rows: int = 300) -> pd.DataFrame:
    """
    生成模拟 K 线 + 技术指标数据（用于无真实数据时的快速测试）。

    参数:
        n_rows: 生成行数

    返回:
        包含 open/high/low/close/volume/macd/rsi_6/kdj_k/kdj_d/sentiment_score 的 DataFrame
    """
    rng = np.random.default_rng(42)
    # 模拟收盘价随机游走
    close = 10.0 + np.cumsum(rng.normal(0, 0.2, n_rows))
    close = np.abs(close)  # 防止负数
    volume = rng.uniform(1e6, 1e7, n_rows)
    macd = rng.normal(0, 0.5, n_rows)
    rsi_6 = rng.uniform(20, 80, n_rows)
    kdj_k = rng.uniform(20, 80, n_rows)
    kdj_d = rng.uniform(20, 80, n_rows)
    sentiment_score = rng.uniform(-1, 1, n_rows)

    return pd.DataFrame({
        "close": close,
        "volume": volume,
        "macd": macd,
        "rsi_6": rsi_6,
        "kdj_k": kdj_k,
        "kdj_d": kdj_d,
        "sentiment_score": sentiment_score,
    })


def _prepare_lstm_data(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    从 DataFrame 构建 LSTM 训练所需的滑动窗口数据。

    参数:
        df: 包含 7 维特征列的 DataFrame
        seq_len: 时序窗口长度

    返回:
        X: numpy [n_samples, seq_len, 7]
        y: numpy [n_samples] 三分类标签 (UP=0, DOWN=1, FLAT=2)
        scaler: 训练数据拟合的 StandardScaler（需保存供推理时使用）
    """
    # 确保所有特征列存在，缺失列填 0
    for col in LSTM_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # 提取特征矩阵并填充 NaN
    feat = df[LSTM_FEATURES].fillna(0.0).values.astype(np.float32)

    # StandardScaler 归一化（按列）
    scaler = StandardScaler()
    feat = scaler.fit_transform(feat)

    # 生成标签: 下一日涨跌
    close_vals = df["close"].ffill().values
    diff = close_vals[1:] - close_vals[:-1]
    threshold = close_vals[:-1] * 0.005  # 0.5% 阈值判断 FLAT
    raw_label = np.where(diff > threshold, 0,  # UP=0
                         np.where(diff < -threshold, 1,  # DOWN=1
                                  2))  # FLAT=2

    # 构建滑动窗口 [n_samples, seq_len, 7]
    n = len(feat) - seq_len
    X_list, y_list = [], []
    for i in range(n):
        window_end = i + seq_len
        if window_end >= len(raw_label):
            break
        X_list.append(feat[i:window_end])
        y_list.append(raw_label[window_end])

    if len(X_list) == 0:
        raise ValueError("数据量不足以构建训练集，至少需要 seq_len+1 行数据")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), scaler


def train_lstm(symbol: str, epochs: int = 50, data_limit: int | None = None,
               progress_callback: Optional[Callable] = None) -> dict:
    """
    训练 LSTM 模型。

    流程:
    1. 尝试从数据库/DataCollector 获取历史 K 线数据
    2. 若无真实数据，使用模拟数据
    3. 用 IndicatorEngine 计算技术指标（真实数据路径）
    4. 准备滑动窗口训练数据: seq_len=20，features=7
    5. 时间序列 train/val 分割（80/20）
    6. 训练 LSTM，包含 Early Stopping (patience=10)
    7. 保存模型和 scaler 到 data/models/
    8. 返回训练和验证指标

    参数:
        symbol: 股票代码，如 '000001'
        epochs: 最大训练轮数
        data_limit: 限制数据行数，用于快速测试（None 表示不限制）

    返回:
        dict 包含 train_loss, val_loss, val_accuracy
    """
    # ── 1. 获取训练数据 ────────────────────────────────────────────────────────
    df = None
    try:
        from app.core.data_collector import DataCollector
        collector = DataCollector()
        raw_df = collector.fetch_stock_history(symbol)
        if raw_df is not None and len(raw_df) > SEQ_LEN + 2:
            engine = IndicatorEngine()
            df = engine.calculate_all(raw_df)
            if "sentiment_score" not in df.columns:
                df["sentiment_score"] = 0.0
            logger.info(f"[{symbol}] LSTM 训练数据: {len(df)} 条历史K线 (东方财富/新浪)")
    except Exception as e:
        logger.warning(f"[{symbol}] LSTM 获取真实数据失败: {e}")
        df = None

    if df is None or len(df) <= SEQ_LEN + 2:
        n_rows = data_limit if data_limit else 300
        df = _generate_mock_data(n_rows)
        logger.info(f"[{symbol}] LSTM 使用模拟数据训练: {n_rows} 条")
    elif data_limit:
        df = df.tail(data_limit).reset_index(drop=True)

    # ── 2. 准备训练数据 ─────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(5, "正在计算技术指标和特征")
    X, y, scaler = _prepare_lstm_data(df, seq_len=SEQ_LEN)

    # ── 3. 时间序列 train/val 分割（80/20，不随机打乱）────────────────────────
    split_idx = int(len(X) * 0.8)
    if split_idx < 1:
        split_idx = len(X) - 1
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    # ── 4. 初始化模型 ────────────────────────────────────────────────────────────
    model = LSTMModel(input_size=7, hidden_size=128, num_layers=2, dropout=0.2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── 5. 训练循环（含 Early Stopping，patience=10）────────────────────────────
    best_loss = float("inf")
    patience_counter = 0
    patience = 10
    last_loss = float("inf")

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    if progress_callback:
        progress_callback(10, "开始训练")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out["direction_prob"], y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        last_loss = avg_loss

        # 进度回调：训练阶段占 10%-90%
        if progress_callback:
            pct = 10 + int(80 * (epoch + 1) / epochs)
            progress_callback(pct, f"Epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")

        # Early Stopping 检查
        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if progress_callback:
                    progress_callback(90, f"Early stopping at epoch {epoch + 1}")
                break

    # ── 6. 验证集评估 ────────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(92, "正在评估验证集")
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        if len(X_val_t) > 0:
            out = model(X_val_t)
            val_loss = float(criterion(out["direction_prob"], y_val_t).item())
            preds = out["direction_prob"].argmax(dim=-1)
            val_correct = int((preds == y_val_t).sum().item())
    val_accuracy = val_correct / len(y_val) if len(y_val) > 0 else 0.0

    # ── 7. 保存模型和 scaler ─────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(96, "正在保存模型")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / f"{symbol}_lstm.pt"
    torch.save(model.state_dict(), str(save_path))

    scaler_path = MODEL_DIR / f"{symbol}_lstm_scaler.pkl"
    joblib.dump(scaler, str(scaler_path))
    logger.info(f"[{symbol}] LSTM scaler 已保存: {scaler_path}")

    if progress_callback:
        progress_callback(100, "训练完成")

    return {
        "train_loss": float(last_loss),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
    }


def _prepare_xgboost_data(df: pd.DataFrame):
    """
    构建 XGBoost 训练所需的特征矩阵和二分类标签。

    参数:
        df: 包含技术指标列的 DataFrame

    返回:
        X: numpy [n_samples, n_features]
        y: numpy [n_samples] 二分类标签 (涨=1, 跌=0)
        scaler: 训练数据拟合的 StandardScaler（需保存供推理时使用）
    """
    feature_cols = [
        "close", "volume", "macd", "macd_signal", "macd_hist",
        "rsi_6", "rsi_12", "rsi_24",
        "kdj_k", "kdj_d", "kdj_j",
        "ma_5", "ma_10", "ma_20",
        "sentiment_score",
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # 使用全量数据（不再截断为最近 252 天）
    df_use = df.copy()
    feat = df_use[feature_cols].fillna(0.0).values.astype(np.float32)

    scaler = StandardScaler()
    feat = scaler.fit_transform(feat)

    # 二分类标签: 下一日涨(1) 或跌(0)
    close_vals = df_use["close"].ffill().values
    raw_label = (close_vals[1:] > close_vals[:-1]).astype(int)

    X = feat[:-1]
    y = raw_label

    return X, y, scaler


def train_xgboost(symbol: str, progress_callback: Optional[Callable] = None) -> dict:
    """
    训练 XGBoost 模型。

    流程:
    1. 获取数据 + 技术指标（无真实数据则用模拟数据）
    2. 构建特征矩阵（技术指标 + 情感分值等）
    3. 标签: 下一日涨/跌（二分类）
    4. 时间序列 train/val 分割（80/20）
    5. 训练 XGBoost + 验证集评估
    6. 保存模型和 scaler 到 data/models/
    7. 返回训练和验证指标

    参数:
        symbol: 股票代码

    返回:
        dict 包含 train_accuracy, val_accuracy
    """
    # ── 1. 获取训练数据 ────────────────────────────────────────────────────────
    df = None
    try:
        from app.core.data_collector import DataCollector
        collector = DataCollector()
        raw_df = collector.fetch_stock_history(symbol)
        if raw_df is not None and len(raw_df) > 30:
            engine = IndicatorEngine()
            df = engine.calculate_all(raw_df)
            if "sentiment_score" not in df.columns:
                df["sentiment_score"] = 0.0
            logger.info(f"[{symbol}] XGBoost 训练数据: {len(df)} 条历史K线 (东方财富/新浪)")
    except Exception as e:
        logger.warning(f"[{symbol}] XGBoost 获取真实数据失败: {e}")
        df = None

    if df is None or len(df) <= 30:
        n_rows = 300
        base = _generate_mock_data(n_rows)
        base["macd_signal"] = base["macd"] * 0.9 + np.random.normal(0, 0.05, n_rows)
        base["macd_hist"] = base["macd"] - base["macd_signal"]
        base["rsi_12"] = np.random.uniform(20, 80, n_rows)
        base["rsi_24"] = np.random.uniform(20, 80, n_rows)
        base["kdj_j"] = 3 * base["kdj_k"] - 2 * base["kdj_d"]
        close_arr = base["close"].values
        base["ma_5"] = pd.Series(close_arr).rolling(5, min_periods=1).mean().values
        base["ma_10"] = pd.Series(close_arr).rolling(10, min_periods=1).mean().values
        base["ma_20"] = pd.Series(close_arr).rolling(20, min_periods=1).mean().values
        df = base

    # ── 2. 准备数据 ──────────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(10, "正在构建特征矩阵")
    X, y, scaler = _prepare_xgboost_data(df)

    if len(X) < 10:
        raise ValueError("有效训练样本不足 10 条")

    # ── 3. 时间序列 train/val 分割（80/20）────────────────────────────────────
    split_idx = int(len(X) * 0.8)
    if split_idx < 1:
        split_idx = len(X) - 1
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # ── 4. 训练 XGBoost ─────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(30, "正在训练 XGBoost 模型")
    predictor = XGBoostPredictor()
    predictor.train(X_train, y_train)
    if progress_callback:
        progress_callback(70, "训练完成，正在评估")

    # ── 5. 评估 ──────────────────────────────────────────────────────────────────
    train_probs = predictor.predict(X_train)
    train_preds = (train_probs >= 0.5).astype(int)
    train_accuracy = float(np.mean(train_preds == y_train))

    val_probs = predictor.predict(X_val)
    val_preds = (val_probs >= 0.5).astype(int)
    val_accuracy = float(np.mean(val_preds == y_val)) if len(y_val) > 0 else 0.0

    # ── 6. 保存模型和 scaler ─────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(85, "正在保存模型")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / f"{symbol}_xgboost.json"
    predictor.save(str(save_path))

    scaler_path = MODEL_DIR / f"{symbol}_xgb_scaler.pkl"
    joblib.dump(scaler, str(scaler_path))
    logger.info(f"[{symbol}] XGBoost scaler 已保存: {scaler_path}")

    if progress_callback:
        progress_callback(100, "训练完成")

    return {
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    }
