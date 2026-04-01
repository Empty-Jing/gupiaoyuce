"""
测试预测模型模块
测试 LSTM 前向传播、XGBoost 训练/预测/保存/加载及 PredictionEngine 接口。
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from app.models.lstm_model import LSTMModel
from app.models.xgboost_model import XGBoostPredictor
from app.core.prediction_engine import PredictionEngine, _default_result, _get_trend_rating


# ── LSTM 模型测试 ──────────────────────────────────────────────────────────────

class TestLSTMModel:
    """LSTMModel 单元测试集。"""

    def setup_method(self):
        self.model = LSTMModel(input_size=7, hidden_size=128, num_layers=2, dropout=0.2)
        self.model.eval()

    def test_forward_direction_prob_shape(self):
        """forward() 中 direction_prob 应为 [batch, 3] 形状。"""
        x = torch.randn(4, 20, 7)
        with torch.no_grad():
            out = self.model(x)
        assert out["direction_prob"].shape == (4, 3)

    def test_forward_price_range_shape(self):
        """forward() 中 price_range 应为 [batch, 2] 形状。"""
        x = torch.randn(4, 20, 7)
        with torch.no_grad():
            out = self.model(x)
        assert out["price_range"].shape == (4, 2)

    def test_forward_predicted_return_shape(self):
        """forward() 中 predicted_return 应为 [batch, 1] 形状。"""
        x = torch.randn(4, 20, 7)
        with torch.no_grad():
            out = self.model(x)
        assert out["predicted_return"].shape == (4, 1)

    def test_forward_direction_prob_sums_to_one(self):
        """direction_prob 经 softmax 后每行之和应约等于 1.0。"""
        x = torch.randn(8, 20, 7)
        with torch.no_grad():
            out = self.model(x)
        row_sums = out["direction_prob"].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(8), atol=1e-5)

    def test_forward_direction_prob_all_non_negative(self):
        """direction_prob 的所有值应 ≥ 0（softmax 保证）。"""
        x = torch.randn(4, 20, 7)
        with torch.no_grad():
            out = self.model(x)
        assert (out["direction_prob"] >= 0).all()

    def test_forward_batch_size_one(self):
        """batch=1 时 forward() 应正常工作。"""
        x = torch.randn(1, 20, 7)
        with torch.no_grad():
            out = self.model(x)
        assert out["direction_prob"].shape == (1, 3)
        assert out["price_range"].shape == (1, 2)
        assert out["predicted_return"].shape == (1, 1)

    def test_forward_returns_dict_with_correct_keys(self):
        """forward() 应返回包含三个预期键的字典。"""
        x = torch.randn(2, 20, 7)
        with torch.no_grad():
            out = self.model(x)
        assert "direction_prob" in out
        assert "price_range" in out
        assert "predicted_return" in out

    def test_model_has_lstm_layer(self):
        """LSTMModel 应有 lstm 属性。"""
        assert hasattr(self.model, "lstm")

    def test_model_has_direction_head(self):
        """LSTMModel 应有 direction_head 线性层。"""
        assert hasattr(self.model, "direction_head")


# ── XGBoost 预测器测试 ─────────────────────────────────────────────────────────

class TestXGBoostPredictor:
    """XGBoostPredictor 单元测试集。"""

    def _make_data(self, n: int = 200, n_features: int = 10, seed: int = 0):
        """生成二分类训练数据。"""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, n_features)).astype(np.float32)
        y = (rng.uniform(0, 1, n) > 0.5).astype(int)
        return X, y

    def test_train_then_predict(self):
        """训练后 predict() 应返回值在 [0, 1] 的概率数组。"""
        predictor = XGBoostPredictor()
        X, y = self._make_data()
        predictor.train(X, y)
        probs = predictor.predict(X[:10])
        assert probs.shape == (10,)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_before_train_raises(self):
        """未训练时调用 predict() 应抛出 ValueError。"""
        predictor = XGBoostPredictor()
        X, _ = self._make_data(n=5)
        with pytest.raises(ValueError, match="模型未训练"):
            predictor.predict(X)

    def test_save_and_load_roundtrip(self):
        """save()/load() 后预测结果应与原始模型一致。"""
        predictor = XGBoostPredictor()
        X, y = self._make_data()
        predictor.train(X, y)
        original_probs = predictor.predict(X[:5])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            predictor.save(path)
            # 加载到新实例
            predictor2 = XGBoostPredictor()
            predictor2.load(path)
            loaded_probs = predictor2.predict(X[:5])
            np.testing.assert_allclose(original_probs, loaded_probs, rtol=1e-5)
        finally:
            os.unlink(path)

    def test_train_with_custom_params(self):
        """传入自定义超参数 train() 应正常完成。"""
        predictor = XGBoostPredictor()
        X, y = self._make_data()
        predictor.train(X, y, params={"n_estimators": 10, "max_depth": 3})
        probs = predictor.predict(X[:5])
        assert len(probs) == 5

    def test_predict_output_is_numpy_array(self):
        """predict() 应返回 numpy 数组。"""
        predictor = XGBoostPredictor()
        X, y = self._make_data()
        predictor.train(X, y)
        result = predictor.predict(X[:3])
        assert isinstance(result, np.ndarray)

    def test_save_does_nothing_when_model_is_none(self):
        """未训练时调用 save() 不应抛出异常。"""
        predictor = XGBoostPredictor()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            predictor.save(path)  # 不应报错
        finally:
            os.unlink(path)


# ── PredictionEngine 测试 ──────────────────────────────────────────────────────

class TestPredictionEngine:
    """PredictionEngine 单元测试集。"""

    def test_predict_returns_default_when_no_models(self):
        """无模型文件时 predict() 应返回默认中性结果。"""
        engine = PredictionEngine()
        result = engine.predict("999999")  # 不存在的股票
        assert isinstance(result, dict)
        assert "direction" in result
        assert "probability" in result
        assert "price_range" in result
        assert "predicted_return" in result
        assert "trend_rating" in result
        assert "model_weights" in result
        assert "lstm_result" in result
        assert "xgboost_result" in result

    def test_predict_default_direction_is_flat(self):
        """无模型时默认方向应为 FLAT。"""
        engine = PredictionEngine()
        result = engine.predict("999999")
        assert result["direction"] == "FLAT"

    def test_predict_default_probability_is_half(self):
        """无模型时默认概率应为 0.5。"""
        engine = PredictionEngine()
        result = engine.predict("999999")
        assert result["probability"] == 0.5

    def test_predict_default_trend_rating_is_neutral(self):
        """无模型时趋势评级应为中性。"""
        engine = PredictionEngine()
        result = engine.predict("999999")
        assert result["trend_rating"] == "中性"

    def test_default_result_structure(self):
        """_default_result() 应包含所有必需键。"""
        r = _default_result()
        required_keys = {"direction", "probability", "price_range", "predicted_return", "trend_rating", "model_weights", "lstm_result", "xgboost_result"}
        assert set(r.keys()) == required_keys

    def test_get_trend_rating_flat(self):
        """FLAT 方向应始终返回中性。"""
        assert _get_trend_rating("FLAT", 0.9) == "中性"
        assert _get_trend_rating("FLAT", 0.1) == "中性"

    def test_get_trend_rating_strong_up(self):
        """UP + 概率 > 0.8 应返回强烈看涨。"""
        assert _get_trend_rating("UP", 0.85) == "强烈看涨"

    def test_get_trend_rating_normal_up(self):
        """UP + 概率 0.6~0.8 应返回看涨。"""
        assert _get_trend_rating("UP", 0.7) == "看涨"

    def test_get_trend_rating_strong_down(self):
        """DOWN + 概率 > 0.8 应返回强烈看跌。"""
        assert _get_trend_rating("DOWN", 0.9) == "强烈看跌"

    def test_get_trend_rating_normal_down(self):
        """DOWN + 概率 0.6~0.8 应返回看跌。"""
        assert _get_trend_rating("DOWN", 0.65) == "看跌"

    def test_get_trend_rating_weak_up_returns_neutral(self):
        """UP + 概率 < 0.6 应返回中性。"""
        assert _get_trend_rating("UP", 0.5) == "中性"


# ── 训练函数端到端测试 ────────────────────────────────────────────────────────

class TestTrainLSTMEndToEnd:
    """train_lstm 端到端测试（使用 mock 数据路径）。"""

    def test_train_lstm_returns_dict_with_metrics(self):
        from app.models.train import train_lstm
        result = train_lstm("TEST_LSTM_E2E", epochs=5, data_limit=60)
        assert isinstance(result, dict)
        assert "train_loss" in result
        assert "val_loss" in result
        assert "val_accuracy" in result
        assert result["train_loss"] >= 0
        assert 0 <= result["val_accuracy"] <= 1

    def test_train_lstm_saves_scaler(self):
        from pathlib import Path
        from app.models.train import train_lstm, MODEL_DIR
        train_lstm("TEST_SCALER_LSTM", epochs=3, data_limit=60)
        scaler_path = MODEL_DIR / "TEST_SCALER_LSTM_lstm_scaler.pkl"
        assert scaler_path.exists()
        scaler_path.unlink(missing_ok=True)
        (MODEL_DIR / "TEST_SCALER_LSTM_lstm.pt").unlink(missing_ok=True)

    def test_lstm_scaler_roundtrip_consistency(self):
        import joblib
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from app.models.train import train_lstm, MODEL_DIR
        train_lstm("TEST_SCALER_RT", epochs=3, data_limit=60)
        scaler_path = MODEL_DIR / "TEST_SCALER_RT_lstm_scaler.pkl"
        assert scaler_path.exists()
        scaler = joblib.load(str(scaler_path))
        assert isinstance(scaler, StandardScaler)
        assert scaler.n_features_in_ == 7
        sample = np.random.randn(1, 7).astype(np.float32)
        transformed = scaler.transform(sample)
        assert transformed.shape == (1, 7)
        scaler_path.unlink(missing_ok=True)
        (MODEL_DIR / "TEST_SCALER_RT_lstm.pt").unlink(missing_ok=True)


class TestTrainXGBoostEndToEnd:
    """train_xgboost 端到端测试（使用 mock 数据路径）。"""

    def test_train_xgboost_returns_dict_with_metrics(self):
        from app.models.train import train_xgboost
        result = train_xgboost("TEST_XGB_E2E")
        assert isinstance(result, dict)
        assert "train_accuracy" in result
        assert "val_accuracy" in result
        assert 0 <= result["train_accuracy"] <= 1
        assert 0 <= result["val_accuracy"] <= 1

    def test_train_xgboost_saves_scaler(self):
        from app.models.train import train_xgboost, MODEL_DIR
        train_xgboost("TEST_SCALER_XGB")
        scaler_path = MODEL_DIR / "TEST_SCALER_XGB_xgb_scaler.pkl"
        assert scaler_path.exists()
        scaler_path.unlink(missing_ok=True)
        (MODEL_DIR / "TEST_SCALER_XGB_xgboost.json").unlink(missing_ok=True)

    def test_xgb_scaler_roundtrip_consistency(self):
        import joblib
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from app.models.train import train_xgboost, MODEL_DIR
        train_xgboost("TEST_SCALER_XRT")
        scaler_path = MODEL_DIR / "TEST_SCALER_XRT_xgb_scaler.pkl"
        assert scaler_path.exists()
        scaler = joblib.load(str(scaler_path))
        assert isinstance(scaler, StandardScaler)
        assert scaler.n_features_in_ == 15
        sample = np.random.randn(1, 15).astype(np.float32)
        transformed = scaler.transform(sample)
        assert transformed.shape == (1, 15)
        scaler_path.unlink(missing_ok=True)
        (MODEL_DIR / "TEST_SCALER_XRT_xgboost.json").unlink(missing_ok=True)


class TestPrepareDataFunctions:
    """测试 _prepare_lstm_data 和 _prepare_xgboost_data 返回 scaler。"""

    def test_prepare_lstm_data_returns_three_values(self):
        from app.models.train import _prepare_lstm_data, _generate_mock_data
        df = _generate_mock_data(100)
        X, y, scaler = _prepare_lstm_data(df)
        assert X.ndim == 3
        assert y.ndim == 1
        from sklearn.preprocessing import StandardScaler
        assert isinstance(scaler, StandardScaler)

    def test_prepare_xgboost_data_returns_three_values(self):
        import pandas as pd
        from app.models.train import _prepare_xgboost_data, _generate_mock_data
        df = _generate_mock_data(100)
        df["macd_signal"] = 0.0
        df["macd_hist"] = 0.0
        df["rsi_12"] = 50.0
        df["rsi_24"] = 50.0
        df["kdj_j"] = 50.0
        df["ma_5"] = df["close"].rolling(5, min_periods=1).mean()
        df["ma_10"] = df["close"].rolling(10, min_periods=1).mean()
        df["ma_20"] = df["close"].rolling(20, min_periods=1).mean()
        X, y, scaler = _prepare_xgboost_data(df)
        assert X.ndim == 2
        assert y.ndim == 1
        assert len(X) == len(y)
        from sklearn.preprocessing import StandardScaler
        assert isinstance(scaler, StandardScaler)

    def test_prepare_xgboost_data_uses_full_data(self):
        import pandas as pd
        from app.models.train import _prepare_xgboost_data, _generate_mock_data
        df = _generate_mock_data(500)
        df["macd_signal"] = 0.0
        df["macd_hist"] = 0.0
        df["rsi_12"] = 50.0
        df["rsi_24"] = 50.0
        df["kdj_j"] = 50.0
        df["ma_5"] = df["close"].rolling(5, min_periods=1).mean()
        df["ma_10"] = df["close"].rolling(10, min_periods=1).mean()
        df["ma_20"] = df["close"].rolling(20, min_periods=1).mean()
        X, y, _ = _prepare_xgboost_data(df)
        assert len(X) == 499
