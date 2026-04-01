"""
测试 SentimentAnalyzer 模块
使用 mock 替代 transformers pipeline，验证情感分析逻辑和优雅降级行为。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
import pytest

from app.core.sentiment_analyzer import SentimentAnalyzer, LABEL_MAP


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def make_pipeline_mock(label: str, score: float = 0.9):
    """创建返回指定标签和分数的 pipeline mock。"""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"label": label, "score": score}]
    return mock_pipeline


# ── 测试类 ────────────────────────────────────────────────────────────────────

class TestSentimentAnalyzerWithMockedPipeline:
    """使用 mock pipeline 测试 SentimentAnalyzer 正常逻辑。"""

    def test_analyze_positive_text(self):
        """正面文本应返回 positive 标签，score > 0。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = make_pipeline_mock("positive", 0.95)
        result = analyzer.analyze("公司业绩大幅超预期，股价创历史新高")
        assert result["label"] == "positive"
        assert result["score"] > 0
        assert result["confidence"] == pytest.approx(0.95)

    def test_analyze_negative_text(self):
        """负面文本应返回 negative 标签，score < 0。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = make_pipeline_mock("negative", 0.88)
        result = analyzer.analyze("公司遭遇巨额亏损，面临退市风险")
        assert result["label"] == "negative"
        assert result["score"] < 0
        assert result["confidence"] == pytest.approx(0.88)

    def test_analyze_neutral_text(self):
        """中性文本应返回 neutral 标签，score 为 0.0。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = make_pipeline_mock("neutral", 0.75)
        result = analyzer.analyze("今日市场成交量维持平稳")
        assert result["label"] == "neutral"
        assert result["score"] == 0.0

    def test_analyze_label_0_maps_to_negative(self):
        """LABEL_0 应映射为 negative。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = make_pipeline_mock("LABEL_0", 0.80)
        result = analyzer.analyze("测试文本")
        assert result["label"] == "negative"

    def test_analyze_label_1_maps_to_neutral(self):
        """LABEL_1 应映射为 neutral。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = make_pipeline_mock("LABEL_1", 0.70)
        result = analyzer.analyze("测试文本")
        assert result["label"] == "neutral"

    def test_analyze_label_2_maps_to_positive(self):
        """LABEL_2 应映射为 positive。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = make_pipeline_mock("LABEL_2", 0.90)
        result = analyzer.analyze("测试文本")
        assert result["label"] == "positive"

    def test_analyze_score_range_positive(self):
        """positive 结果的 score 应在 (0, 1]。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = make_pipeline_mock("positive", 0.85)
        result = analyzer.analyze("好消息")
        assert 0 < result["score"] <= 1.0

    def test_analyze_score_range_negative(self):
        """negative 结果的 score 应在 [-1, 0)。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = make_pipeline_mock("negative", 0.85)
        result = analyzer.analyze("坏消息")
        assert -1.0 <= result["score"] < 0

    def test_analyze_returns_dict_with_required_keys(self):
        """analyze 应返回包含 label/score/confidence 三键的字典。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = make_pipeline_mock("positive", 0.9)
        result = analyzer.analyze("测试")
        assert "label" in result
        assert "score" in result
        assert "confidence" in result

    def test_batch_analyze_returns_list(self):
        """batch_analyze 应返回与输入等长的列表。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.8},
            {"label": "neutral", "score": 0.7},
        ]
        analyzer._pipeline = mock_pipeline
        results = analyzer.batch_analyze(["文本1", "文本2", "文本3"])
        assert len(results) == 3
        for r in results:
            assert "label" in r
            assert "score" in r

    def test_batch_analyze_labels_mapped_correctly(self):
        """batch_analyze 应正确映射每条文本的标签。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.8},
        ]
        analyzer._pipeline = mock_pipeline
        results = analyzer.batch_analyze(["好消息", "坏消息"])
        assert results[0]["label"] == "positive"
        assert results[1]["label"] == "negative"


class TestSentimentAnalyzerGracefulDegradation:
    """测试 SentimentAnalyzer 模型不可用时的优雅降级。"""

    def test_analyze_returns_neutral_when_pipeline_none(self):
        """pipeline 为 None 时 analyze 应返回中性结果。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = None
        result = analyzer.analyze("任意文本")
        assert result["label"] == "neutral"
        assert result["score"] == 0.0
        assert result["confidence"] == 0.0

    def test_batch_analyze_returns_all_neutral_when_pipeline_none(self):
        """pipeline 为 None 时 batch_analyze 应返回全中性列表。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = None
        results = analyzer.batch_analyze(["文本A", "文本B", "文本C"])
        assert len(results) == 3
        for r in results:
            assert r["label"] == "neutral"
            assert r["score"] == 0.0

    def test_analyze_returns_neutral_when_pipeline_raises(self):
        """pipeline 执行时抛出异常，analyze 应捕获并返回中性。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        mock_pipeline = MagicMock(side_effect=RuntimeError("模型推理失败"))
        analyzer._pipeline = mock_pipeline
        result = analyzer.analyze("任意文本")
        assert result["label"] == "neutral"
        assert result["score"] == 0.0

    def test_batch_analyze_returns_neutral_when_pipeline_raises(self):
        """pipeline 批量推理失败时 batch_analyze 应返回全中性。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        mock_pipeline = MagicMock(side_effect=RuntimeError("批量推理失败"))
        analyzer._pipeline = mock_pipeline
        results = analyzer.batch_analyze(["A", "B"])
        assert len(results) == 2
        assert all(r["label"] == "neutral" for r in results)

    def test_load_model_success(self):
        """模型加载成功时 _pipeline 应被设置为非 None 的 pipeline 对象。"""
        import sys

        mock_pipeline_obj = MagicMock()
        mock_pipeline_fn = MagicMock(return_value=mock_pipeline_obj)
        mock_model_cls = MagicMock()
        mock_tokenizer_cls = MagicMock()

        fake_transformers = MagicMock()
        fake_transformers.pipeline = mock_pipeline_fn
        fake_transformers.AutoModelForSequenceClassification = mock_model_cls
        fake_transformers.AutoTokenizer = mock_tokenizer_cls

        original = sys.modules.get("transformers")
        sys.modules["transformers"] = fake_transformers
        try:
            analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
            analyzer._model_name = "test-model"
            analyzer._pipeline = None
            analyzer._load_model()
        finally:
            if original is not None:
                sys.modules["transformers"] = original
            else:
                del sys.modules["transformers"]

        assert analyzer._pipeline is mock_pipeline_obj

    def test_load_model_failure_sets_pipeline_none(self):
        """模型加载失败时 _pipeline 应为 None（不抛出异常）。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._model_name = "非存在的模型路径/xyz"
        analyzer._pipeline = None
        # 无需 mock，直接加载不存在的模型应触发异常并降级
        analyzer._load_model()
        assert analyzer._pipeline is None


class TestSentimentAnalyzerDailySentiment:
    """测试 daily_sentiment_mean 计算逻辑。"""

    def test_daily_sentiment_mean_empty_list(self):
        """空列表应返回 0.0。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = None
        assert analyzer.daily_sentiment_mean([]) == 0.0

    def test_daily_sentiment_mean_single_value(self):
        """单值列表应返回该值本身。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = None
        assert analyzer.daily_sentiment_mean([0.5]) == pytest.approx(0.5)

    def test_daily_sentiment_mean_multiple_values(self):
        """多值列表应返回平均值。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = None
        result = analyzer.daily_sentiment_mean([0.2, 0.4, 0.6])
        assert result == pytest.approx(0.4)

    def test_daily_sentiment_mean_negative_values(self):
        """包含负值的列表应正确计算平均值。"""
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._pipeline = None
        result = analyzer.daily_sentiment_mean([-0.5, 0.5])
        assert result == pytest.approx(0.0)


# ── LLM 备选分析测试 ─────────────────────────────────────────────────────────

class TestSentimentAnalyzerLLMFallback:
    """测试 LLM API 备选情感分析逻辑。"""

    def _make_analyzer_no_pipeline(self):
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer._model_name = "test-model"
        analyzer._pipeline = None
        return analyzer

    @patch("app.core.sentiment_analyzer.settings")
    @patch("app.core.sentiment_analyzer.requests.post")
    def test_llm_fallback_positive(self, mock_post, mock_settings):
        mock_settings.LLM_API_KEY = "test-key"
        mock_settings.LLM_API_BASE_URL = "https://api.test.com/v1"
        mock_settings.LLM_MODEL_NAME = "gpt-4o-mini"
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": '{"label":"positive","score":0.8,"confidence":0.9}'}}]
            }),
        )
        analyzer = self._make_analyzer_no_pipeline()
        result = analyzer._analyze_with_llm("公司业绩大涨")
        assert result["label"] == "positive"
        assert result["score"] == pytest.approx(0.8)
        assert result["confidence"] == pytest.approx(0.9)

    @patch("app.core.sentiment_analyzer.settings")
    @patch("app.core.sentiment_analyzer.requests.post")
    def test_llm_fallback_negative(self, mock_post, mock_settings):
        mock_settings.LLM_API_KEY = "test-key"
        mock_settings.LLM_API_BASE_URL = "https://api.test.com/v1"
        mock_settings.LLM_MODEL_NAME = "gpt-4o-mini"
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": '{"label":"negative","score":-0.7,"confidence":0.85}'}}]
            }),
        )
        analyzer = self._make_analyzer_no_pipeline()
        result = analyzer._analyze_with_llm("巨额亏损退市")
        assert result["label"] == "negative"
        assert result["score"] == pytest.approx(-0.7)

    @patch("app.core.sentiment_analyzer.settings")
    def test_llm_fallback_no_api_key_returns_neutral(self, mock_settings):
        mock_settings.LLM_API_KEY = ""
        analyzer = self._make_analyzer_no_pipeline()
        result = analyzer._analyze_with_llm("任意文本")
        assert result["label"] == "neutral"
        assert result["score"] == 0.0

    @patch("app.core.sentiment_analyzer.settings")
    @patch("app.core.sentiment_analyzer.requests.post")
    def test_llm_fallback_request_error_returns_neutral(self, mock_post, mock_settings):
        mock_settings.LLM_API_KEY = "test-key"
        mock_settings.LLM_API_BASE_URL = "https://api.test.com/v1"
        mock_settings.LLM_MODEL_NAME = "gpt-4o-mini"
        mock_post.side_effect = Exception("网络错误")
        analyzer = self._make_analyzer_no_pipeline()
        result = analyzer._analyze_with_llm("任意文本")
        assert result["label"] == "neutral"
        assert result["score"] == 0.0

    @patch("app.core.sentiment_analyzer.settings")
    @patch("app.core.sentiment_analyzer.requests.post")
    def test_llm_fallback_invalid_json_returns_neutral(self, mock_post, mock_settings):
        mock_settings.LLM_API_KEY = "test-key"
        mock_settings.LLM_API_BASE_URL = "https://api.test.com/v1"
        mock_settings.LLM_MODEL_NAME = "gpt-4o-mini"
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "这不是有效的JSON"}}]
            }),
        )
        analyzer = self._make_analyzer_no_pipeline()
        result = analyzer._analyze_with_llm("任意文本")
        assert result["label"] == "neutral"

    @patch("app.core.sentiment_analyzer.settings")
    @patch("app.core.sentiment_analyzer.requests.post")
    def test_llm_clamps_score_range(self, mock_post, mock_settings):
        mock_settings.LLM_API_KEY = "test-key"
        mock_settings.LLM_API_BASE_URL = "https://api.test.com/v1"
        mock_settings.LLM_MODEL_NAME = "gpt-4o-mini"
        mock_post.return_value = MagicMock(
            status_code=200,
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "choices": [{"message": {"content": '{"label":"positive","score":5.0,"confidence":2.0}'}}]
            }),
        )
        analyzer = self._make_analyzer_no_pipeline()
        result = analyzer._analyze_with_llm("超大涨")
        assert result["score"] == 1.0
        assert result["confidence"] == 1.0

    def test_analyze_uses_llm_when_pipeline_none(self):
        analyzer = self._make_analyzer_no_pipeline()
        with patch.object(analyzer, "_analyze_with_llm", return_value={"label": "positive", "score": 0.5, "confidence": 0.8}) as mock_llm:
            result = analyzer.analyze("测试文本")
            mock_llm.assert_called_once_with("测试文本")
            assert result["label"] == "positive"

    def test_batch_analyze_uses_llm_when_pipeline_none(self):
        analyzer = self._make_analyzer_no_pipeline()
        with patch.object(analyzer, "_analyze_with_llm", return_value={"label": "neutral", "score": 0.0, "confidence": 0.0}) as mock_llm:
            results = analyzer.batch_analyze(["文本A", "文本B"])
            assert mock_llm.call_count == 2
            assert len(results) == 2
