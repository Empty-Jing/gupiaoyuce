import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from app.core.data_collector import (
    DataCollector,
    HIST_SINA_TO_CN_MAP,
    REALTIME_COLUMNS_MAP,
    REALTIME_SINA_COLUMNS_MAP,
    _SINA_SYMBOL_PREFIX,
    _fetch_sina_realtime,
)
import app.core.data_collector as _dc_module


def make_hist_df():
    return pd.DataFrame({
        "日期": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "开盘": [10.0, 10.5, 11.0],
        "收盘": [10.5, 11.0, 11.5],
        "最高": [11.0, 11.5, 12.0],
        "最低": [9.5, 10.0, 10.5],
        "成交量": [100000, 120000, 150000],
        "成交额": [1050000.0, 1320000.0, 1725000.0],
        "振幅": [5.0, 4.8, 4.5],
        "涨跌幅": [5.0, 4.76, 4.55],
        "涨跌额": [0.5, 0.5, 0.5],
        "换手率": [1.5, 1.8, 2.0],
    })


def make_realtime_df():
    return pd.DataFrame({
        "代码": ["000001", "600036", "300750"],
        "名称": ["平安银行", "招商银行", "宁德时代"],
        "最新价": [12.50, 35.80, 200.0],
        "涨跌幅": [2.5, -1.2, 3.8],
        "成交量": [50000, 30000, 20000],
        "成交额": [625000.0, 1074000.0, 4000000.0],
        "最高": [12.80, 36.0, 205.0],
        "最低": [12.20, 35.5, 198.0],
        "今开": [12.30, 36.0, 199.0],
        "昨收": [12.20, 36.24, 192.71],
    })


def make_sina_hist_df():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "open": [10.0, 10.5, 11.0],
        "high": [11.0, 11.5, 12.0],
        "low": [9.5, 10.0, 10.5],
        "close": [10.5, 11.0, 11.5],
        "volume": [100000, 120000, 150000],
        "amount": [1050000.0, 1320000.0, 1725000.0],
        "outstanding_share": [1000000, 1000000, 1000000],
        "turnover": [1.5, 1.8, 2.0],
    })


SINA_RESPONSE_TEXT = (
    'var hq_str_sz000001="平安银行,12.300,12.200,12.500,12.800,12.200,12.49,12.50,'
    '50000,625000.00,100,12.49,200,12.48,300,12.47,400,12.46,500,12.45,'
    '100,12.50,200,12.51,300,12.52,400,12.53,500,12.54,'
    '2024-01-02,15:00:00,00";\n'
    'var hq_str_sh600036="招商银行,36.000,36.240,35.800,36.000,35.500,35.79,35.80,'
    '30000,1074000.00,100,35.79,200,35.78,300,35.77,400,35.76,500,35.75,'
    '100,35.80,200,35.81,300,35.82,400,35.83,500,35.84,'
    '2024-01-02,15:00:00,00";\n'
)


class TestDataCollector:

    def setup_method(self):
        _dc_module._em_last_failure = 0.0
        self.dc = DataCollector()

    @patch("app.core.data_collector.ak.stock_zh_a_hist")
    def test_fetch_stock_history_returns_cn_columns(self, mock_hist):
        mock_hist.return_value = make_hist_df()
        df = self.dc.fetch_stock_history("000001")
        assert isinstance(df, pd.DataFrame)
        assert "开盘" in df.columns
        assert "收盘" in df.columns
        assert "最高" in df.columns
        assert "最低" in df.columns
        assert "成交量" in df.columns

    @patch("app.core.data_collector.ak.stock_zh_a_hist")
    def test_fetch_stock_history_row_count(self, mock_hist):
        mock_hist.return_value = make_hist_df()
        df = self.dc.fetch_stock_history("000001")
        assert len(df) == 3

    @patch("app.core.data_collector.ak.stock_zh_a_hist")
    def test_fetch_stock_history_passes_params(self, mock_hist):
        mock_hist.return_value = make_hist_df()
        self.dc.fetch_stock_history("600036", period="weekly", adjust="hfq",
                                     start_date="20240101", end_date="20241231")
        call_kwargs = mock_hist.call_args.kwargs
        assert call_kwargs["symbol"] == "600036"
        assert call_kwargs["period"] == "weekly"
        assert call_kwargs["adjust"] == "hfq"
        assert call_kwargs["start_date"] == "20240101"
        assert call_kwargs["end_date"] == "20241231"

    @patch("app.core.data_collector.ak.stock_zh_a_hist")
    def test_fetch_stock_history_no_optional_params(self, mock_hist):
        mock_hist.return_value = make_hist_df()
        self.dc.fetch_stock_history("000001")
        call_kwargs = mock_hist.call_args.kwargs
        assert "start_date" not in call_kwargs
        assert "end_date" not in call_kwargs

    @patch("app.core.data_collector.ak.stock_zh_a_hist")
    def test_fetch_stock_history_drops_symbol_code_col(self, mock_hist):
        df_with_extra = make_hist_df()
        df_with_extra["股票代码"] = "000001"
        mock_hist.return_value = df_with_extra
        df = self.dc.fetch_stock_history("000001")
        assert "股票代码" not in df.columns

    @patch("app.core.data_collector.ak.stock_zh_a_daily")
    @patch("app.core.data_collector.ak.stock_zh_a_hist")
    def test_fetch_stock_history_raises_after_max_retries(self, mock_hist, mock_daily):
        mock_hist.side_effect = Exception("持续网络错误")
        mock_daily.side_effect = Exception("新浪也失败")
        with patch("app.core.data_collector.time.sleep"):
            with pytest.raises(Exception):
                self.dc.fetch_stock_history("000001")

    @patch("app.core.data_collector.ak.stock_zh_a_daily")
    @patch("app.core.data_collector.ak.stock_zh_a_hist")
    def test_fetch_stock_history_fallback_to_sina_cn_columns(self, mock_hist, mock_daily):
        mock_hist.side_effect = RuntimeError("push2 unreachable")
        mock_daily.return_value = make_sina_hist_df()
        with patch("app.core.data_collector.time.sleep"):
            df = self.dc.fetch_stock_history("000001")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "收盘" in df.columns
        assert "开盘" in df.columns
        assert "成交额" in df.columns
        assert "换手率" in df.columns
        assert "outstanding_share" not in df.columns
        mock_daily.assert_called_once()
        call_kwargs = mock_daily.call_args.kwargs
        assert call_kwargs["symbol"] == "sz000001"

    @patch("app.core.data_collector.ak.stock_zh_a_daily")
    @patch("app.core.data_collector.ak.stock_zh_a_hist")
    def test_fetch_stock_history_fallback_sh_prefix(self, mock_hist, mock_daily):
        mock_hist.side_effect = RuntimeError("push2 unreachable")
        mock_daily.return_value = make_sina_hist_df()
        with patch("app.core.data_collector.time.sleep"):
            self.dc.fetch_stock_history("600519")
        call_kwargs = mock_daily.call_args.kwargs
        assert call_kwargs["symbol"] == "sh600519"

    @patch("app.core.data_collector.ak.stock_zh_a_spot_em")
    def test_fetch_realtime_quotes_filters_symbols(self, mock_spot):
        mock_spot.return_value = make_realtime_df()
        df = self.dc.fetch_realtime_quotes(["000001", "600036"])
        assert len(df) == 2
        assert set(df["symbol"].tolist()) == {"000001", "600036"}

    @patch("app.core.data_collector.ak.stock_zh_a_spot_em")
    def test_fetch_realtime_quotes_columns(self, mock_spot):
        mock_spot.return_value = make_realtime_df()
        df = self.dc.fetch_realtime_quotes(["000001"])
        assert "price" in df.columns
        assert "change_pct" in df.columns

    @patch("app.core.data_collector.ak.stock_zh_a_spot_em")
    def test_fetch_realtime_quotes_empty_when_no_match(self, mock_spot):
        mock_spot.return_value = make_realtime_df()
        df = self.dc.fetch_realtime_quotes(["999999"])
        assert df.empty

    @patch("app.core.data_collector._fetch_sina_realtime")
    @patch("app.core.data_collector.ak.stock_zh_a_spot_em")
    def test_fetch_realtime_quotes_fallback_to_sina(self, mock_spot_em, mock_sina_rt):
        mock_spot_em.side_effect = RuntimeError("push2 unreachable")
        mock_sina_rt.return_value = make_realtime_df()
        with patch("app.core.data_collector.time.sleep"):
            df = self.dc.fetch_realtime_quotes(["000001", "600036"])
        assert len(df) == 2
        assert "price" in df.columns
        assert "change_pct" in df.columns
        mock_sina_rt.assert_called_once()

    @patch("app.core.data_collector.ak.stock_news_em")
    def test_fetch_news_returns_dataframe(self, mock_news):
        mock_news.return_value = pd.DataFrame({
            "新闻标题": ["平安银行业绩亮眼", "银行板块上涨"],
            "新闻内容": ["正文1", "正文2"],
            "发布时间": ["2024-01-01 09:00:00", "2024-01-01 10:00:00"],
        })
        df = self.dc.fetch_news("000001")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    @patch("app.core.data_collector.ak.stock_zh_a_spot_em")
    def test_fetch_stock_list_returns_dataframe(self, mock_spot):
        mock_spot.return_value = make_realtime_df()
        df = self.dc.fetch_stock_list()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    @patch("app.core.data_collector.ak.stock_info_a_code_name")
    @patch("app.core.data_collector.ak.stock_zh_a_spot_em")
    def test_fetch_stock_list_falls_back_to_code_name(self, mock_spot, mock_code_name):
        mock_spot.side_effect = RuntimeError("spot failed")
        mock_code_name.return_value = pd.DataFrame({
            "code": ["000001", "600000"],
            "name": ["平安银行", "浦发银行"],
        })

        with patch("app.core.data_collector.ak.stock_zh_a_spot", side_effect=RuntimeError("sina also failed")):
            with patch("app.core.data_collector.time.sleep"):
                df = self.dc.fetch_stock_list()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["code", "name"]
        assert len(df) == 2
        mock_code_name.assert_called_once()

    @patch("app.core.data_collector.ak.stock_zh_a_spot")
    @patch("app.core.data_collector.ak.stock_zh_a_spot_em")
    def test_fetch_stock_list_falls_back_to_sina(self, mock_spot_em, mock_spot_sina):
        mock_spot_em.side_effect = RuntimeError("em failed")
        mock_spot_sina.return_value = make_realtime_df()
        with patch("app.core.data_collector.time.sleep"):
            df = self.dc.fetch_stock_list()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        mock_spot_sina.assert_called_once()

    @patch("app.core.data_collector.ak.stock_zh_a_hist")
    @patch("app.core.data_collector.asyncio.run")
    def test_save_history_to_db_calls_upsert(self, mock_asyncio_run, mock_hist):
        mock_hist.return_value = make_hist_df()
        mock_asyncio_run.return_value = 3
        count = self.dc.save_history_to_db("000001")
        assert count == 3
        mock_asyncio_run.assert_called_once()

    @patch("app.core.data_collector.ak.stock_zh_a_hist")
    @patch("app.core.data_collector.asyncio.run")
    def test_save_history_to_db_empty_returns_zero(self, mock_asyncio_run, mock_hist):
        mock_hist.return_value = pd.DataFrame()
        count = self.dc.save_history_to_db("000001")
        assert count == 0
        mock_asyncio_run.assert_not_called()

    @patch("app.core.data_collector.ak.stock_info_a_code_name")
    def test_fetch_stock_code_name_list_returns_code_name(self, mock_code_name):
        mock_code_name.return_value = pd.DataFrame({
            "code": ["000001", "600036", "300750"],
            "name": ["平安银行", "招商银行", "宁德时代"],
        })
        df = self.dc.fetch_stock_code_name_list()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["code", "name"]
        assert len(df) == 3

    @patch("app.core.data_collector.ak.stock_info_a_code_name")
    def test_fetch_stock_code_name_list_raises_on_failure(self, mock_code_name):
        mock_code_name.side_effect = RuntimeError("network error")
        with pytest.raises(RuntimeError):
            self.dc.fetch_stock_code_name_list()


class TestRetryProxyBypass:

    @patch("app.core.data_collector.time.sleep")
    def test_retry_bypasses_proxy_env(self, _mock_sleep):
        env_during_call = {}

        def capture_env():
            env_during_call["NO_PROXY"] = os.environ.get("NO_PROXY")
            env_during_call["http_proxy"] = os.environ.get("http_proxy")
            return 42

        with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:7897"}, clear=False):
            from app.core.data_collector import _retry
            result = _retry(capture_env, max_retries=1)

        assert result == 42
        assert env_during_call["NO_PROXY"] == "*"
        assert env_during_call["http_proxy"] is None

    @patch("app.core.data_collector.time.sleep")
    def test_retry_restores_proxy_env_after_success(self, _mock_sleep):
        original_no_proxy = "localhost"

        with patch.dict(
            os.environ,
            {"http_proxy": "http://127.0.0.1:7897", "NO_PROXY": original_no_proxy},
            clear=False,
        ):
            from app.core.data_collector import _retry
            _retry(lambda: "ok", max_retries=1)
            assert os.environ.get("NO_PROXY") == original_no_proxy

    @patch("app.core.data_collector.time.sleep")
    def test_retry_restores_proxy_env_after_failure(self, _mock_sleep):
        original_no_proxy = "localhost"

        with patch.dict(
            os.environ,
            {"http_proxy": "http://127.0.0.1:7897", "NO_PROXY": original_no_proxy},
            clear=False,
        ):
            from app.core.data_collector import _retry
            with pytest.raises(ValueError):
                _retry(lambda: (_ for _ in ()).throw(ValueError("boom")), max_retries=1)
            assert os.environ.get("NO_PROXY") == original_no_proxy

    @patch("app.core.data_collector.time.sleep")
    def test_retry_no_proxy_fallback(self, _mock_sleep):
        from app.core.data_collector import _retry
        call_count = 0

        def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("timeout")
            return "ok"

        with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:7897"}, clear=False):
            with pytest.raises(ConnectionError):
                _retry(fail_once, max_retries=1)
        assert call_count == 1


class TestFetchSinaRealtime:

    @patch("app.core.data_collector.requests.get")
    def test_parses_sina_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = SINA_RESPONSE_TEXT
        mock_resp.encoding = "gbk"
        mock_get.return_value = mock_resp

        df = _fetch_sina_realtime(["000001", "600036"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(df["代码"].tolist()) == {"000001", "600036"}
        assert df[df["代码"] == "000001"]["最新价"].iloc[0] == 12.5
        assert df[df["代码"] == "600036"]["名称"].iloc[0] == "招商银行"

    @patch("app.core.data_collector.requests.get")
    def test_calculates_change_pct(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = SINA_RESPONSE_TEXT
        mock_resp.encoding = "gbk"
        mock_get.return_value = mock_resp

        df = _fetch_sina_realtime(["000001"])
        row = df[df["代码"] == "000001"].iloc[0]
        expected_pct = round((12.5 - 12.2) / 12.2 * 100, 2)
        assert row["涨跌幅"] == expected_pct

    @patch("app.core.data_collector.requests.get")
    def test_returns_empty_for_invalid_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = '<html>error</html>'
        mock_resp.encoding = "gbk"
        mock_get.return_value = mock_resp

        df = _fetch_sina_realtime(["000001"])
        assert df.empty

    @patch("app.core.data_collector.requests.get")
    def test_bypasses_proxy(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = SINA_RESPONSE_TEXT
        mock_resp.encoding = "gbk"
        mock_get.return_value = mock_resp

        original_proxy = "http://127.0.0.1:7897"
        with patch.dict(os.environ, {"http_proxy": original_proxy}, clear=False):
            _fetch_sina_realtime(["000001"])
            assert os.environ.get("http_proxy") == original_proxy

    @patch("app.core.data_collector.requests.get")
    def test_symbol_prefix_mapping(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = SINA_RESPONSE_TEXT
        mock_resp.encoding = "gbk"
        mock_get.return_value = mock_resp

        _fetch_sina_realtime(["000001", "600036"])
        call_url = mock_get.call_args[0][0]
        assert "sz000001" in call_url
        assert "sh600036" in call_url
