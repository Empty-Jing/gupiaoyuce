import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from app.core.train_manager import (
    _TRAIN_JOBS,
    _JOBS_LOCK,
    submit_job,
    get_job_status,
    list_jobs,
    _make_progress_callback,
)


@pytest.fixture(autouse=True)
def _clear_jobs():
    with _JOBS_LOCK:
        _TRAIN_JOBS.clear()
    yield
    with _JOBS_LOCK:
        _TRAIN_JOBS.clear()


class TestProgressCallback:
    def test_callback_updates_progress(self):
        with _JOBS_LOCK:
            _TRAIN_JOBS["test123"] = {"progress": 0, "message": ""}

        cb = _make_progress_callback("test123")
        cb(50, "half done")

        with _JOBS_LOCK:
            assert _TRAIN_JOBS["test123"]["progress"] == 50
            assert _TRAIN_JOBS["test123"]["message"] == "half done"

    def test_callback_clamps_progress(self):
        with _JOBS_LOCK:
            _TRAIN_JOBS["test123"] = {"progress": 0, "message": ""}

        cb = _make_progress_callback("test123")
        cb(150, "over")
        with _JOBS_LOCK:
            assert _TRAIN_JOBS["test123"]["progress"] == 100

        cb(-10, "under")
        with _JOBS_LOCK:
            assert _TRAIN_JOBS["test123"]["progress"] == 0

    def test_callback_noop_for_missing_job(self):
        cb = _make_progress_callback("nonexistent")
        cb(50, "should not crash")


class TestSubmitJob:
    @patch("app.core.train_manager._run_training")
    def test_submit_returns_job_id(self, mock_run):
        job_id = submit_job("000001", "lstm")
        assert isinstance(job_id, str)
        assert len(job_id) == 12

    @patch("app.core.train_manager._run_training")
    def test_submit_creates_pending_job(self, mock_run):
        job_id = submit_job("000001", "lstm")
        status = get_job_status(job_id)
        assert status is not None
        assert status["status"] == "pending"
        assert status["symbol"] == "000001"
        assert status["model_type"] == "lstm"
        assert status["progress"] == 0

    @patch("app.core.train_manager._run_training")
    def test_submit_multiple_jobs(self, mock_run):
        id1 = submit_job("000001", "lstm")
        id2 = submit_job("000001", "xgboost")
        id3 = submit_job("600036", "lstm")
        assert id1 != id2 != id3
        assert len(list_jobs()) == 3


class TestGetJobStatus:
    def test_nonexistent_job_returns_none(self):
        assert get_job_status("nonexistent") is None

    @patch("app.core.train_manager._run_training")
    def test_returns_copy(self, mock_run):
        job_id = submit_job("000001", "lstm")
        status1 = get_job_status(job_id)
        status2 = get_job_status(job_id)
        assert status1 is not status2
        assert status1 == status2


class TestListJobs:
    @patch("app.core.train_manager._run_training")
    def test_list_empty(self, mock_run):
        assert list_jobs() == []

    @patch("app.core.train_manager._run_training")
    def test_list_filter_by_symbol(self, mock_run):
        submit_job("000001", "lstm")
        submit_job("600036", "lstm")
        submit_job("000001", "xgboost")

        result = list_jobs(symbol="000001")
        assert len(result) == 2
        assert all(j["symbol"] == "000001" for j in result)

    @patch("app.core.train_manager._run_training")
    def test_list_limit(self, mock_run):
        for _ in range(5):
            submit_job("000001", "lstm")
        assert len(list_jobs(limit=3)) == 3


class TestRunTrainingIntegration:
    @patch("app.models.train.train_lstm")
    def test_successful_lstm_training(self, mock_train):
        mock_train.return_value = {"train_loss": 0.5, "val_loss": 0.6, "val_accuracy": 0.7}

        from app.core.train_manager import _run_training

        with _JOBS_LOCK:
            _TRAIN_JOBS["integ_test"] = {
                "job_id": "integ_test",
                "symbol": "000001",
                "model_type": "lstm",
                "status": "pending",
                "progress": 0,
                "message": "",
                "result": None,
                "error": None,
                "started_at": None,
                "finished_at": None,
                "created_at": "2026-01-01T00:00:00",
            }

        _run_training("integ_test", "000001", "lstm")

        job = get_job_status("integ_test")
        assert job["status"] == "success"
        assert job["progress"] == 100
        assert job["result"]["val_accuracy"] == 0.7
        assert job["finished_at"] is not None
        mock_train.assert_called_once()

    @patch("app.models.train.train_xgboost")
    def test_successful_xgboost_training(self, mock_train):
        mock_train.return_value = {"train_accuracy": 0.8, "val_accuracy": 0.75}

        from app.core.train_manager import _run_training

        with _JOBS_LOCK:
            _TRAIN_JOBS["integ_xgb"] = {
                "job_id": "integ_xgb",
                "symbol": "600036",
                "model_type": "xgboost",
                "status": "pending",
                "progress": 0,
                "message": "",
                "result": None,
                "error": None,
                "started_at": None,
                "finished_at": None,
                "created_at": "2026-01-01T00:00:00",
            }

        _run_training("integ_xgb", "600036", "xgboost")

        job = get_job_status("integ_xgb")
        assert job["status"] == "success"
        assert job["result"]["val_accuracy"] == 0.75

    @patch("app.models.train.train_lstm", side_effect=RuntimeError("OOM"))
    def test_failed_training(self, mock_train):
        from app.core.train_manager import _run_training

        with _JOBS_LOCK:
            _TRAIN_JOBS["fail_test"] = {
                "job_id": "fail_test",
                "symbol": "000001",
                "model_type": "lstm",
                "status": "pending",
                "progress": 0,
                "message": "",
                "result": None,
                "error": None,
                "started_at": None,
                "finished_at": None,
                "created_at": "2026-01-01T00:00:00",
            }

        _run_training("fail_test", "000001", "lstm")

        job = get_job_status("fail_test")
        assert job["status"] == "failed"
        assert "OOM" in job["error"]
        assert job["finished_at"] is not None

    def test_invalid_model_type(self):
        from app.core.train_manager import _run_training

        with _JOBS_LOCK:
            _TRAIN_JOBS["bad_type"] = {
                "job_id": "bad_type",
                "symbol": "000001",
                "model_type": "transformer",
                "status": "pending",
                "progress": 0,
                "message": "",
                "result": None,
                "error": None,
                "started_at": None,
                "finished_at": None,
                "created_at": "2026-01-01T00:00:00",
            }

        _run_training("bad_type", "000001", "transformer")

        job = get_job_status("bad_type")
        assert job["status"] == "failed"
        assert "不支持" in job["error"]


class TestProgressCallbackInTraining:
    def test_lstm_receives_callback(self):
        progress_log = []

        def mock_callback(progress, message=""):
            progress_log.append((progress, message))

        from app.models.train import train_lstm
        train_lstm("TEST_CB", epochs=3, data_limit=50, progress_callback=mock_callback)

        assert len(progress_log) > 0
        assert progress_log[0][0] == 5
        assert progress_log[-1][0] == 100
        progresses = [p for p, _ in progress_log]
        assert progresses == sorted(progresses)

    def test_xgboost_receives_callback(self):
        progress_log = []

        def mock_callback(progress, message=""):
            progress_log.append((progress, message))

        from app.models.train import train_xgboost
        train_xgboost("TEST_CB", progress_callback=mock_callback)

        assert len(progress_log) > 0
        assert progress_log[-1][0] == 100
        progresses = [p for p, _ in progress_log]
        assert progresses == sorted(progresses)

    def test_lstm_works_without_callback(self):
        from app.models.train import train_lstm
        result = train_lstm("TEST_NOCB", epochs=2, data_limit=50)
        assert "val_accuracy" in result

    def test_xgboost_works_without_callback(self):
        from app.models.train import train_xgboost
        result = train_xgboost("TEST_NOCB")
        assert "val_accuracy" in result
