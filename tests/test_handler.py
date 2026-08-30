import os
import sqlite3
import stat
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock


sys.modules.setdefault("runpod", types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda *_: None, start=lambda *_: None)))

import handler


class HandlerContractTests(unittest.TestCase):
    def test_session_identifier_rejects_path_traversal(self):
        """Session IDs cannot escape worker-owned paths."""
        self.assertEqual("aitk-job-123", handler.require_identifier({"session_id": "aitk-job-123"}, "session_id"))
        with self.assertRaisesRegex(ValueError, "unsupported characters"):
            handler.require_identifier({"session_id": "../../outside"}, "session_id")

    def test_installed_revision_must_match_contract(self):
        """The runtime revision must match the web contract exactly."""
        with mock.patch.dict(os.environ, {"AI_TOOLKIT_REVISION": handler.EXPECTED_REVISION}):
            handler.validate_installed_revision()
        with mock.patch.dict(os.environ, {"AI_TOOLKIT_REVISION": "wrong"}):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                handler.validate_installed_revision()

    def test_worker_owned_paths_cannot_escape(self):
        """Generated process paths remain inside dataset and output roots."""
        valid = {
            "training_folder": str(handler.OUTPUT_ROOT),
            "datasets": [{"folder_path": str(handler.DATASET_ROOT / "subject"), "mask_path": str(handler.DATASET_ROOT / "masks")}],
        }
        handler.validate_process_paths(valid)
        invalid = {
            "training_folder": str(handler.OUTPUT_ROOT),
            "datasets": [{"folder_path": "/tmp/outside"}],
        }
        with self.assertRaisesRegex(ValueError, "must stay inside"):
            handler.validate_process_paths(invalid)

    def test_archive_extraction_rejects_traversal_and_symlinks(self):
        """Unsafe ZIP entries are rejected before extraction."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "dataset"
            destination.mkdir()
            safe_archive = root / "safe.zip"
            with zipfile.ZipFile(safe_archive, "w") as archive:
                archive.writestr("subject/image.png", b"image")
            handler.extract_archive(safe_archive, destination)
            self.assertEqual(b"image", (destination / "subject/image.png").read_bytes())

            traversal_archive = root / "traversal.zip"
            with zipfile.ZipFile(traversal_archive, "w") as archive:
                archive.writestr("../outside.txt", b"unsafe")
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                handler.extract_archive(traversal_archive, destination)

            symlink_archive = root / "symlink.zip"
            link = zipfile.ZipInfo("link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(symlink_archive, "w") as archive:
                archive.writestr(link, "target")
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                handler.extract_archive(symlink_archive, destination)

    def test_latest_loss_reads_official_ui_logger_schema(self):
        """Loss telemetry follows AI Toolkit's current SQLite schema."""
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "loss_log.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE metric_keys (key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER);
                CREATE TABLE metrics (step INTEGER NOT NULL, key TEXT NOT NULL, value_real REAL, value_text TEXT, PRIMARY KEY (step, key));
                INSERT INTO metric_keys VALUES ('loss', 1, 20);
                INSERT INTO metrics VALUES (10, 'loss', 0.42, NULL);
                INSERT INTO metrics VALUES (20, 'loss', 0.18, NULL);
                """
            )
            connection.commit()
            connection.close()
            self.assertEqual((20, 0.18), handler.latest_loss(database))

    def test_completed_run_uploads_a_fresh_final_sample(self):
        """A final sample is uploaded even when it was written moments ago."""
        with tempfile.TemporaryDirectory() as temporary:
            sample_root = Path(temporary)
            sample = sample_root / "sample_step_25.png"
            sample.write_bytes(b"sample")
            uploaded = {}

            def fake_upload(_client, _session_id, path, relative_name):
                """Return a deterministic manifest entry without external storage."""
                return {
                    "url": f"https://storage.example/{relative_name}",
                    "fileName": path.name,
                    "contentType": "image/png",
                    "fileSize": path.stat().st_size,
                    "path": relative_name,
                }

            with mock.patch.object(handler, "upload_file", side_effect=fake_upload):
                self.assertEqual([], handler.discover_samples(object(), "session", sample_root, uploaded))
                samples = handler.discover_samples(object(), "session", sample_root, uploaded, minimum_age_seconds=0)
            self.assertEqual(25, samples[0]["step"])
            self.assertEqual("https://storage.example/samples/sample_step_25.png", samples[0]["url"])

    def test_dataset_origins_and_redirects_are_allowlisted_before_requests(self):
        """Unapproved initial and redirect origins never receive a request."""
        environment = {"AITK_DATASET_ALLOWED_ORIGINS": "https://storage.example"}
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(handler.requests, "get") as request:
            with self.assertRaisesRegex(ValueError, "not allowlisted"):
                handler.open_dataset_response("https://metadata.example/dataset.zip")
            request.assert_not_called()

        redirect = mock.Mock(status_code=302, headers={"location": "https://metadata.example/private"})
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(handler.requests, "get", return_value=redirect) as request:
            with self.assertRaisesRegex(ValueError, "not allowlisted"):
                handler.open_dataset_response("https://storage.example/dataset.zip")
            request.assert_called_once_with("https://storage.example/dataset.zip", stream=True, timeout=(20, 300), allow_redirects=False)
            redirect.close.assert_called_once()

    def test_storage_endpoints_and_returned_urls_require_https(self):
        """Worker uploads and exposed artifact links cannot use plaintext HTTP."""
        environment = {
            "AITK_S3_BUCKET": "bucket",
            "AITK_S3_ACCESS_KEY": "access",
            "AITK_S3_SECRET_KEY": "secret",
            "AITK_S3_ENDPOINT": "http://storage.example",
        }
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(handler.boto3, "client") as client:
            with self.assertRaisesRegex(ValueError, "must be an absolute HTTPS URL"):
                handler.storage_client()
            client.assert_not_called()

        with mock.patch.dict(os.environ, {"AITK_S3_PUBLIC_BASE_URL": "http://cdn.example"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be an absolute HTTPS URL"):
                handler.object_url(mock.Mock(), "bucket", "artifact.safetensors")

        presigner = mock.Mock()
        presigner.generate_presigned_url.return_value = "http://storage.example/signed"
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be an absolute HTTPS URL"):
                handler.object_url(presigner, "bucket", "artifact.safetensors")


if __name__ == "__main__":
    unittest.main()
