"""
XGBoost 预测器
二分类模型：预测下一日涨(1)或跌(0)
"""

import numpy as np
import xgboost as xgb


class XGBoostPredictor:
    """
    基于 XGBoost 的股票涨跌二分类预测器。
    
    输入特征为技术指标向量，输出为上涨概率 [0, 1]。
    """

    def __init__(self):
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray, params: dict | None = None):
        """
        训练 XGBoost 二分类模型（涨/跌）。

        参数:
            X: 特征矩阵，形状 [n_samples, n_features]
            y: 标签向量，0=跌 1=涨，形状 [n_samples]
            params: 可选的超参数覆盖字典
        """
        default_params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "eval_metric": "logloss",
        }
        if params:
            default_params.update(params)
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测上涨概率。

        参数:
            X: 特征矩阵，形状 [n_samples, n_features]

        返回:
            上涨概率数组，值域 [0, 1]，形状 [n_samples]
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train() 方法")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        """
        保存模型到 JSON 文件。

        参数:
            path: 保存路径，例如 data/models/000001_xgboost.json
        """
        if self.model is not None:
            self.model.save_model(path)

    def load(self, path: str):
        """
        从 JSON 文件加载模型。

        参数:
            path: 模型文件路径
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
