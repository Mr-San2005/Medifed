"""Tests for the MediFed dashboard and local diagnosis service."""

import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import uvicorn

from app.services import available_variants, load_results
from app.server import app


class TestAppServices(unittest.TestCase):

    def test_result_csvs_are_aggregated(self):
        """TC-APP-001 known result CSVs populate hospital metrics and rounds."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data_counts.csv").write_text(
                "hospital,class,found,kept,train,val\n1,NORMAL,20,20,16,4\n",
                encoding="utf-8",
            )
            (root / "comparison_methods.csv").write_text(
                "hospital,local_accuracy,fedavg_mean,fedavg_std,fedbn_mean,fedbn_std\n1,0.8,0.7,0.1,0.9,0.0\n",
                encoding="utf-8",
            )
            (root / "summary_rounds.csv").write_text(
                "mode,hospital,round,mean_val_acc,std_val_acc,mean_val_loss\nbaseline,1,1,0.7,0.0,0.4\n",
                encoding="utf-8",
            )
            result = load_results(root)
            self.assertEqual(result["hospitals"][0]["counts"][0]["val"], 4)
            self.assertEqual(result["hospitals"][0]["fedbn_accuracy"], 0.9)
            self.assertEqual(result["rounds"][0]["accuracy"], 0.7)

    def test_missing_files_are_marked_empty(self):
        """TC-APP-002 missing result files produce explicit empty-state flags."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = load_results(temp_dir)
            self.assertTrue(result["empty_states"]["data_counts"])
            self.assertTrue(result["empty_states"]["method_comparison"])
            self.assertTrue(result["empty_states"]["round_curves"])

    def test_only_existing_checkpoints_are_listed(self):
        """TC-APP-003 available variants include only checkpoint-backed models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "local_hospital2.pt").touch()
            self.assertEqual(available_variants(root), ["local_hospital2"])


class TestAppHttp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Start one local HTTP server for endpoint checks."""
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cls.port = listener.getsockname()[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=cls.port, log_level="error")
        )
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.monotonic() + 10
        while not cls.server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        if not cls.server.started:
            raise RuntimeError("Test HTTP server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)

    def _get(self, path):
        with urlopen(self.base_url + path, timeout=5) as response:
            return response.status, response.headers, response.read()

    def test_health_endpoint(self):
        """TC-APP-004 health endpoint returns an OK response."""
        status, _, body = self._get("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_pages_return_html(self):
        """TC-APP-005 dashboard and diagnosis routes serve HTML pages."""
        for path in ("/", "/diagnose"):
            status, headers, body = self._get(path)
            self.assertEqual(status, 200)
            self.assertIn("text/html", headers["content-type"])
            self.assertTrue(body.startswith(b"<!doctype html>"))

    def test_prediction_returns_ranked_classes_and_heatmap(self):
        """TC-APP-006 prediction endpoint returns ranked classes and a heatmap payload."""
        fake_result = {
            "ranked": [{"class": "NORMAL", "probability": 0.8}],
            "top_class": "NORMAL",
            "milliseconds": 1.0,
            "original_png": b"original",
            "heatmap_png": b"heatmap",
        }
        with patch("app.server.run_prediction", return_value=fake_result):
            request = Request(
                self.base_url + "/api/predict?variant=fedbn_hospital1",
                data=b"image-bytes",
                headers={"Content-Type": "image/png"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                result = json.loads(response.read())
                self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(result["ranked"][0]["class"], "NORMAL")
        self.assertTrue(result["heatmap_png"])

    def test_corrupt_image_returns_bad_request(self):
        """TC-APP-007 invalid image bytes return HTTP 400."""
        request = Request(
            self.base_url + "/api/predict?variant=fedbn_hospital1",
            data=b"not an image",
            headers={"Content-Type": "image/png"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=5)
        self.assertEqual(error.exception.code, 400)

    def test_oversized_image_returns_payload_too_large(self):
        """TC-APP-008 request bodies above 10 MB return HTTP 413."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(
            "POST",
            "/api/predict?variant=fedbn_hospital1",
            body=b"",
            headers={"Content-Type": "image/png", "Content-Length": str(10 * 1024 * 1024 + 1)},
        )
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        connection.close()

    def test_unknown_variant_returns_not_found(self):
        """TC-APP-009 unknown model variants return HTTP 404."""
        request = Request(
            self.base_url + "/api/predict?variant=missing-model",
            data=b"image-bytes",
            headers={"Content-Type": "image/png"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=5)
        self.assertEqual(error.exception.code, 404)

    def test_missing_checkpoint_returns_service_unavailable(self):
        """TC-APP-010 known variants without checkpoint files return HTTP 503."""
        with patch("app.server.available_variants", return_value=[]):
            request = Request(
                self.base_url + "/api/predict?variant=fedbn_hospital1",
                data=b"image-bytes",
                headers={"Content-Type": "image/png"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=5)
        self.assertEqual(error.exception.code, 503)


if __name__ == "__main__":
    unittest.main()