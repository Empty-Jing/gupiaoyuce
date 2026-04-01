"""
预测引擎
双模型融合: 0.6 * LSTM + 0.4 * XGBoost
无模型文件时返回默认中性结果
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# 模型保存目录
MODEL_DIR = Path("data/models")

# LSTM 时序特征列（顺序固定）
LSTM_FEATURES = ["close", "volume", "macd", "rsi_6", "kdj_k", "kdj_d", "sentiment_score"]
SEQ_LEN = 20

# 趋势评级阈值
STRONG_THRESHOLD = 0.8
NORMAL_THRESHOLD = 0.6


def _default_result() -> dict:
    """返回无模型时的默认中性结果。"""
    return {
        "direction": "FLAT",
        "probability": 0.5,
        "price_range": [0.0, 0.0],
        "predicted_return": 0.0,
        "trend_rating": "中性",
        "model_weights": {},
        "lstm_result": {},
        "xgboost_result": {},
    }


def _get_trend_rating(direction: str, probability: float) -> str:
    """
    根据方向和概率生成趋势评级（5 档）。

    参数:
        direction: "UP" / "DOWN" / "FLAT"
        probability: 该方向的置信概率 [0, 1]

    返回:
        "强烈看涨" / "看涨" / "中性" / "看跌" / "强烈看跌"
    """
    if direction == "FLAT":
        return "中性"
    if direction == "UP":
        if probability > STRONG_THRESHOLD:
            return "强烈看涨"
        elif probability > NORMAL_THRESHOLD:
            return "看涨"
        else:
            return "中性"
    else:  # DOWN
        if probability > STRONG_THRESHOLD:
            return "强烈看跌"
        elif probability > NORMAL_THRESHOLD:
            return "看跌"
        else:
            return "中性"


def _prepare_recent_features(symbol: str) -> np.ndarray | None:
    """
    获取最近 SEQ_LEN 天的特征数据，用于 LSTM 预测。

    参数:
        symbol: 股票代码

    返回:
        numpy [1, SEQ_LEN, 7] float32，若数据不足返回 None
    """
    try:
        from app.core.data_collector import DataCollector
        from app.core.indicator_engine import IndicatorEngine
        from sklearn.preprocessing import StandardScaler
        import joblib

        collector = DataCollector()
        raw_df = collector.fetch_stock_history(symbol)
        if raw_df is None or len(raw_df) < SEQ_LEN + 5:
            return None

        engine = IndicatorEngine()
        df = engine.calculate_all(raw_df)
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0

        for col in LSTM_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        feat = df[LSTM_FEATURES].fillna(0.0).values.astype(np.float32)
        if len(feat) < SEQ_LEN:
            return None

        # 优先加载训练时保存的 scaler，保证训练/推理一致性
        scaler_path = MODEL_DIR / f"{symbol}_lstm_scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(str(scaler_path))
            feat = scaler.transform(feat)
        else:
            logger.warning(f"[{symbol}] 未找到 LSTM scaler，回退到 fit_transform")
            scaler = StandardScaler()
            feat = scaler.fit_transform(feat)

        window = feat[-SEQ_LEN:]
        return window[np.newaxis, :, :].astype(np.float32)

    except Exception:
        return None


def _prepare_recent_xgb_features(symbol: str) -> np.ndarray | None:
    """
    获取最近 1 天的 XGBoost 特征向量。

    参数:
        symbol: 股票代码

    返回:
        numpy [1, n_features] float32，若数据不足返回 None
    """
    feature_cols = [
        "close", "volume", "macd", "macd_signal", "macd_hist",
        "rsi_6", "rsi_12", "rsi_24",
        "kdj_k", "kdj_d", "kdj_j",
        "ma_5", "ma_10", "ma_20",
        "sentiment_score",
    ]
    try:
        from app.core.data_collector import DataCollector
        from app.core.indicator_engine import IndicatorEngine
        from sklearn.preprocessing import StandardScaler
        import joblib

        collector = DataCollector()
        raw_df = collector.fetch_stock_history(symbol)
        if raw_df is None or len(raw_df) < 30:
            return None

        engine = IndicatorEngine()
        df = engine.calculate_all(raw_df)
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        feat = df[feature_cols].fillna(0.0).values.astype(np.float32)

        scaler_path = MODEL_DIR / f"{symbol}_xgb_scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(str(scaler_path))
            feat = scaler.transform(feat)
        else:
            logger.warning(f"[{symbol}] 未找到 XGBoost scaler，回退到 fit_transform")
            scaler = StandardScaler()
            feat = scaler.fit_transform(feat)

        return feat[[-1], :]

    except Exception:
        return None


class PredictionEngine:
    """
    双模型融合预测引擎。

    融合规则:
    - LSTM 权重: 0.6，输出 3 类概率 [UP, DOWN, FLAT]
    - XGBoost 权重: 0.4，输出涨的概率 p → 近似为 [1-p, 0, p] (映射到 [DOWN, FLAT, UP])
    - 最终合并后取 argmax 对应方向
    """

    def __init__(self):
        """初始化引擎（延迟加载模型，避免启动时崩溃）。"""
        # 模型缓存字典: {symbol: {"lstm": model, "xgb": predictor}}
        self._cache: dict = {}

    def _load_lstm(self, symbol: str):
        """
        从磁盘加载 LSTM 模型。

        参数:
            symbol: 股票代码

        返回:
            LSTMModel 或 None（文件不存在时）
        """
        path = MODEL_DIR / f"{symbol}_lstm.pt"
        if not path.exists():
            return None
        try:
            import torch
            from app.models.lstm_model import LSTMModel
            model = LSTMModel(input_size=7, hidden_size=128, num_layers=2, dropout=0.2)
            model.load_state_dict(torch.load(str(path), weights_only=True))
            model.eval()
            return model
        except Exception:
            return None

    def _load_xgb(self, symbol: str):
        """
        从磁盘加载 XGBoost 模型。

        参数:
            symbol: 股票代码

        返回:
            XGBoostPredictor 或 None（文件不存在时）
        """
        path = MODEL_DIR / f"{symbol}_xgboost.json"
        if not path.exists():
            return None
        try:
            from app.models.xgboost_model import XGBoostPredictor
            predictor = XGBoostPredictor()
            predictor.load(str(path))
            return predictor
        except Exception:
            return None

    def predict(self, symbol: str) -> dict:
        """
        统一预测接口，返回双模型融合结果。

        参数:
            symbol: 股票代码

        返回:
            dict 包含:
            - direction: "UP" / "DOWN" / "FLAT"
            - probability: 融合后最优方向的概率 [0, 1]
            - price_range: [预测最低价相对变化, 预测最高价相对变化]
            - predicted_return: 预测收益率
            - trend_rating: "强烈看涨" / "看涨" / "中性" / "看跌" / "强烈看跌"
        """
        # ── 加载模型（使用缓存）─────────────────────────────────────────────────
        if symbol not in self._cache:
            self._cache[symbol] = {
                "lstm": self._load_lstm(symbol),
                "xgb": self._load_xgb(symbol),
            }
        lstm_model = self._cache[symbol]["lstm"]
        xgb_predictor = self._cache[symbol]["xgb"]

        # ── 若两个模型都不存在，返回默认中性结果 ────────────────────────────────
        if lstm_model is None and xgb_predictor is None:
            return _default_result()

        # 方向标签顺序: 0=UP, 1=DOWN, 2=FLAT
        direction_names = ["UP", "DOWN", "FLAT"]
        combined_probs = np.array([1 / 3, 1 / 3, 1 / 3])  # 初始均匀分布

        price_range = [0.0, 0.0]
        predicted_return = 0.0
        lstm_weight = 0.6 if (lstm_model is not None and xgb_predictor is not None) else 1.0
        xgb_weight = 0.4 if (lstm_model is not None and xgb_predictor is not None) else 1.0

        # ── LSTM 预测 ────────────────────────────────────────────────────────────
        lstm_probs = None
        lstm_result: dict = {}
        if lstm_model is not None:
            feat_3d = _prepare_recent_features(symbol)
            if feat_3d is not None:
                try:
                    import torch
                    x_tensor = torch.from_numpy(feat_3d)
                    with torch.no_grad():
                        out = lstm_model(x_tensor)
                    # direction_prob: [1, 3] → UP/DOWN/FLAT
                    lstm_probs = out["direction_prob"].numpy()[0]  # [3]
                    # 价格区间和收益率
                    pr = out["price_range"].numpy()[0]  # [2]
                    price_range = [float(pr[0]), float(pr[1])]
                    predicted_return = float(out["predicted_return"].numpy()[0][0])
                    lstm_result = {
                        "direction_probs": {"UP": float(lstm_probs[0]), "DOWN": float(lstm_probs[1]), "FLAT": float(lstm_probs[2])},
                        "price_range": [float(pr[0]), float(pr[1])],
                        "predicted_return": float(predicted_return),
                    }
                except Exception:
                    lstm_probs = None

        # ── XGBoost 预测 ─────────────────────────────────────────────────────────
        xgb_probs = None
        xgb_result: dict = {}
        if xgb_predictor is not None:
            feat_2d = _prepare_recent_xgb_features(symbol)
            if feat_2d is not None:
                try:
                    p_up = float(xgb_predictor.predict(feat_2d)[0])
                    p_down = 1.0 - p_up
                    # 根据 p_up 与 0.5 的距离分配 FLAT 概率：越接近 50/50 → FLAT 越高
                    margin = abs(p_up - 0.5)
                    if margin < 0.1:
                        flat_prob = 0.3 * (1.0 - margin / 0.1)
                    else:
                        flat_prob = 0.0
                    remaining = 1.0 - flat_prob
                    xgb_probs = np.array([p_up * remaining, p_down * remaining, flat_prob])
                    xgb_result = {
                        "p_up": round(p_up, 4),
                        "p_down": round(p_down, 4),
                    }
                except Exception:
                    xgb_probs = None

        # ── 融合概率 ─────────────────────────────────────────────────────────────
        if lstm_probs is not None and xgb_probs is not None:
            combined_probs = lstm_weight * lstm_probs + xgb_weight * xgb_probs
        elif lstm_probs is not None:
            combined_probs = lstm_probs
        elif xgb_probs is not None:
            combined_probs = xgb_probs
        else:
            # 模型存在但数据不足，返回默认中性
            return _default_result()

        # ── 归一化并确定方向 ─────────────────────────────────────────────────────
        combined_probs = combined_probs / combined_probs.sum()
        best_idx = int(np.argmax(combined_probs))
        direction = direction_names[best_idx]
        probability = float(combined_probs[best_idx])
        trend_rating = _get_trend_rating(direction, probability)

        model_weights: dict[str, float] = {}
        if lstm_model is not None:
            model_weights["lstm"] = lstm_weight
        if xgb_predictor is not None:
            model_weights["xgboost"] = xgb_weight

        return {
            "direction": direction,
            "probability": round(probability, 4),
            "price_range": [round(price_range[0], 4), round(price_range[1], 4)],
            "predicted_return": round(predicted_return, 4),
            "trend_rating": trend_rating,
            "model_weights": model_weights,
            "lstm_result": lstm_result,
            "xgboost_result": xgb_result,
        }
