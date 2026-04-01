"""
基于 FinBERT 的中文金融情感分析模块。
支持模型加载失败时的优雅降级：优先 FinBERT → LLM API 备选 → 中性结果。
"""

import json
import logging
from typing import Optional

import requests

from app.config import settings

logger = logging.getLogger(__name__)

# finbert-tone-chinese 可能返回的标签映射
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    # 如果模型直接返回文字标签
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}


class SentimentAnalyzer:
    """基于 FinBERT 的中文金融情感分析器。"""

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        初始化情感分析器。

        Args:
            model_name: HuggingFace 模型名称，默认从 settings.SENTIMENT_MODEL_NAME 读取。
        """
        self._model_name = model_name or settings.SENTIMENT_MODEL_NAME
        self._pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        """加载 FinBERT 模型，失败时记录 WARNING 并设置 pipeline 为 None。"""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )

            logger.info("正在加载情感分析模型: %s", self._model_name)
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
            )
            logger.info("情感分析模型加载成功: %s", self._model_name)
        except Exception as exc:
            logger.warning(
                "情感分析模型加载失败（%s），将使用中性降级结果。原因: %s",
                self._model_name,
                exc,
            )
            self._pipeline = None

    def _map_result(self, raw: dict) -> dict:
        """
        将 pipeline 原始输出映射为标准格式。

        Args:
            raw: pipeline 返回的单条结果，含 'label' 和 'score' 字段。

        Returns:
            标准化 dict: {"label": str, "score": float, "confidence": float}
        """
        raw_label = raw.get("label", "neutral")
        confidence = float(raw.get("score", 0.0))

        # 规范化标签（转小写后查映射表）
        label = LABEL_MAP.get(raw_label) or LABEL_MAP.get(raw_label.lower(), "neutral")

        # 根据标签计算情感分数
        if label == "positive":
            score = confidence
        elif label == "negative":
            score = -confidence
        else:
            score = 0.0

        return {"label": label, "score": score, "confidence": confidence}

    def _analyze_with_llm(self, text: str) -> dict:
        """通过 LLM API 分析单条文本情感，作为 FinBERT 不可用时的备选方案。"""
        api_key = settings.LLM_API_KEY
        if not api_key:
            return {"label": "neutral", "score": 0.0, "confidence": 0.0}

        try:
            payload = {
                "model": settings.LLM_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            '你是金融情感分析师。分析给定文本的情感倾向，'
                            '返回严格 JSON: {"label":"positive/negative/neutral",'
                            '"score":-1到1的浮点数,"confidence":0到1的浮点数}'
                        ),
                    },
                    {"role": "user", "content": f"分析这条金融新闻的情感：{text[:500]}"},
                ],
                "temperature": 0,
            }

            resp = requests.post(
                f"{settings.LLM_API_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()

            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            label = str(parsed.get("label", "neutral")).lower()
            if label not in ("positive", "negative", "neutral"):
                label = "neutral"
            score = float(parsed.get("score", 0.0))
            score = max(-1.0, min(1.0, score))
            confidence = float(parsed.get("confidence", abs(score)))
            confidence = max(0.0, min(1.0, confidence))

            return {"label": label, "score": score, "confidence": confidence}
        except Exception as exc:
            logger.warning("LLM 情感分析失败: %s", exc)
            return {"label": "neutral", "score": 0.0, "confidence": 0.0}

    def analyze(self, text: str) -> dict:
        """
        分析单条文本的情感。

        降级链: FinBERT pipeline → LLM API → 中性结果。

        Args:
            text: 待分析的文本。

        Returns:
            {"label": "positive/negative/neutral", "score": float(-1~+1), "confidence": float(0~1)}
        """
        if self._pipeline is not None:
            try:
                results = self._pipeline(text)
                raw = results[0] if isinstance(results, list) else results
                return self._map_result(raw)
            except Exception as exc:
                logger.warning("FinBERT 情感分析失败，尝试 LLM 备选。原因: %s", exc)

        return self._analyze_with_llm(text)

    def batch_analyze(self, texts: list) -> list:
        """
        批量分析文本情感。

        降级链: FinBERT pipeline → 逐条 LLM API → 中性结果。

        Args:
            texts: 待分析的文本列表。

        Returns:
            每条文本对应的情感分析结果列表，格式同 analyze()。
        """
        if self._pipeline is not None:
            try:
                raw_results = self._pipeline(texts)
                return [self._map_result(r) for r in raw_results]
            except Exception as exc:
                logger.warning("批量 FinBERT 分析失败，尝试 LLM 备选。原因: %s", exc)

        return [self._analyze_with_llm(t) for t in texts]

    def daily_sentiment_mean(self, scores: list) -> float:
        """
        计算日均情感值（简单平均）。

        Args:
            scores: 情感分数列表（float），每个值范围 -1~+1。

        Returns:
            平均情感分数，空列表返回 0.0。
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
