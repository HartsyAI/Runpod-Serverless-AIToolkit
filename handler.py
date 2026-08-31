"""RunPod Serverless contract for Hartsy's version-pinned AI Toolkit backend."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import threading
import time
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any

import boto3
import requests
import runpod
import yaml


CONTRACT_VERSION = 2
EXPECTED_REVISION = "be995185f598c83abb990a088e9f634c4d36eb46"
TOOLKIT_ROOT = Path(os.getenv("AI_TOOLKIT_ROOT", "/app/ai-toolkit")).resolve()
WORK_ROOT = Path(os.getenv("AITK_WORK_ROOT", "/workspace")).resolve()
VOLUME_ROOT = Path(os.getenv("AITK_VOLUME_ROOT", "/runpod-volume")).resolve()
DATASET_ROOT = Path("/dataset").resolve()
OUTPUT_ROOT = (WORK_ROOT / "output").resolve()
MAX_ARCHIVE_BYTES = max(1, int(os.getenv("AITK_MAX_ARCHIVE_BYTES", str(4 * 1024**3))))
MAX_EXTRACTED_BYTES = max(1, int(os.getenv("AITK_MAX_EXTRACTED_BYTES", str(8 * 1024**3))))
MAX_DATASET_ARCHIVES = max(1, int(os.getenv("AITK_MAX_DATASET_ARCHIVES", "8")))
SAMPLE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mp3", ".wav", ".flac", ".ogg"}
ARTIFACT_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".json", ".yaml", ".yml"}
STEP_PATTERN = re.compile(r"(?:^|[_-])(?:step)?[_-]?(\d+)(?:[_-]|\.)", re.IGNORECASE)
SAFE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
MAX_DATASET_REDIRECTS = 5


def require_text(payload: dict[str, Any], key: str) -> str:
    """Return a required, non-empty string from a RunPod input payload."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"input.{key} is required")
    return value.strip()


def require_identifier(payload: dict[str, Any], key: str) -> str:
    """Return a path-safe identifier from a RunPod input payload."""
    value = require_text(payload, key)
    if not SAFE_IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"input.{key} contains unsupported characters")
    return value


def validate_installed_revision() -> None:
    """Ensure the container and web application use the same AI Toolkit commit."""
    actual_revision = os.getenv("AI_TOOLKIT_REVISION", "").strip()
    if not actual_revision and (TOOLKIT_ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(TOOLKIT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        actual_revision = result.stdout.strip()
    if actual_revision != EXPECTED_REVISION:
        raise RuntimeError(
            f"Installed AI Toolkit revision {actual_revision or 'unknown'} does not match {EXPECTED_REVISION}"
        )


def validate_process_paths(process: dict[str, Any]) -> None:
    """Keep generated training and dataset paths inside worker-owned roots."""
    training_folder = Path(str(process.get("training_folder", ""))).resolve()
    if training_folder != OUTPUT_ROOT:
        raise ValueError(f"config.process[0].training_folder must be {OUTPUT_ROOT}")
    datasets = process.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("config.process[0].datasets must contain at least one dataset")
    for dataset_index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"config.process[0].datasets[{dataset_index}] must be an object")
        for key, value in dataset.items():
            if value is None or not (key.endswith("_path") or re.search(r"_path_\d+$", key)):
                continue
            path = Path(str(value)).resolve()
            if path != DATASET_ROOT and DATASET_ROOT not in path.parents:
                raise ValueError(f"config.process[0].datasets[{dataset_index}].{key} must stay inside {DATASET_ROOT}")


def reset_directory(path: Path) -> None:
    """Create an empty request-scoped directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def require_https_url(url: str, label: str) -> str:
    """Validate an absolute HTTPS URL without embedded credentials."""
    parsed = urllib.parse.urlsplit(url)
    try:
        parsed.port
    except ValueError as error:
        raise ValueError(f"{label} is not a valid URL") from error
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{label} must be an absolute HTTPS URL without credentials")
    return url


def url_origin(url: str, label: str) -> str:
    """Return the normalized HTTPS origin for an already validated URL."""
    parsed = urllib.parse.urlsplit(require_https_url(url, label))
    host = parsed.hostname.lower()
    authority = f"[{host}]" if ":" in host else host
    if parsed.port and parsed.port != 443:
        authority = f"{authority}:{parsed.port}"
    return f"https://{authority}"


def dataset_allowed_origins() -> set[str]:
    """Load the exact object-storage origins permitted for dataset downloads."""
    configured = os.getenv("AITK_DATASET_ALLOWED_ORIGINS", "")
    if not configured.strip():
        raise RuntimeError("AITK_DATASET_ALLOWED_ORIGINS must contain at least one HTTPS origin")
    origins = set()
    for value in configured.split(","):
        candidate = value.strip()
        parsed = urllib.parse.urlsplit(require_https_url(candidate, "Dataset allowlist origin"))
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Dataset allowlist entries must be origins without paths, queries, or fragments")
        origins.add(url_origin(candidate, "Dataset allowlist origin"))
    return origins


def open_dataset_response(url: str):
    """Open an allowlisted dataset URL while validating every redirect target."""
    allowed_origins = dataset_allowed_origins()
    current_url = url
    for redirect_count in range(MAX_DATASET_REDIRECTS + 1):
        if url_origin(current_url, "Dataset archive URL") not in allowed_origins:
            raise ValueError("Dataset archive URL origin is not allowlisted")
        response = requests.get(current_url, stream=True, timeout=(20, 300), allow_redirects=False)
        if response.status_code in REDIRECT_STATUSES:
            location = response.headers.get("location")
            response.close()
            if not location:
                raise ValueError("Dataset archive redirect is missing a location")
            if redirect_count == MAX_DATASET_REDIRECTS:
                raise ValueError("Dataset archive exceeded the redirect limit")
            current_url = urllib.parse.urljoin(current_url, location)
            continue
        try:
            response.raise_for_status()
        except Exception:
            response.close()
            raise
        return response
    raise ValueError("Dataset archive exceeded the redirect limit")


def download_archive(url: str, destination: Path, max_bytes: int = MAX_ARCHIVE_BYTES) -> int:
    """Download an allowlisted dataset archive with a compressed-size limit."""
    with open_dataset_response(url) as response:
        declared_size = int(response.headers.get("content-length", "0") or 0)
        if declared_size > max_bytes:
            raise ValueError("Dataset archive is too large")
        downloaded = 0
        with destination.open("wb") as archive:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError("Dataset archive is too large")
                archive.write(chunk)
    return downloaded


def extract_archive(archive_path: Path, destination: Path, max_bytes: int = MAX_EXTRACTED_BYTES) -> int:
    """Extract a ZIP while rejecting traversal, links, duplicates, and zip bombs."""
    with zipfile.ZipFile(archive_path) as archive:
        extracted_size = 0
        for member in archive.infolist():
            member_mode = member.external_attr >> 16
            if stat.S_ISLNK(member_mode):
                raise ValueError("Dataset archives cannot contain symbolic links")
            extracted_size += member.file_size
            if extracted_size > max_bytes:
                raise ValueError("Extracted dataset is too large")
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError("Dataset archive contains an unsafe path")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"Dataset archives contain duplicate path: {member.filename}")
            with archive.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    return extracted_size


def storage_client():
    """Create an S3 client whose optional custom endpoint is HTTPS-only."""
    required = ["AITK_S3_BUCKET", "AITK_S3_ACCESS_KEY", "AITK_S3_SECRET_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Worker storage is missing: {', '.join(missing)}")
    endpoint = os.getenv("AITK_S3_ENDPOINT", "").strip()
    if endpoint:
        endpoint = require_https_url(endpoint, "AITK_S3_ENDPOINT")
    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        region_name=os.getenv("AITK_S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ["AITK_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["AITK_S3_SECRET_KEY"],
    )


def object_url(client, bucket: str, key: str) -> str:
    """Return an HTTPS public or presigned URL for an uploaded object."""
    public_base = os.getenv("AITK_S3_PUBLIC_BASE_URL", "").rstrip("/")
    if public_base:
        require_https_url(public_base, "AITK_S3_PUBLIC_BASE_URL")
        return require_https_url(f"{public_base}/{urllib.parse.quote(key, safe='/')}", "Storage object URL")
    generated = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=min(int(os.getenv("AITK_URL_TTL_SECONDS", "604800")), 604800),
    )
    return require_https_url(generated, "Presigned storage object URL")


def upload_file(client, session_id: str, path: Path, relative_name: str) -> dict[str, Any]:
    """Publish one worker output onto the attached network volume and return its provider-neutral manifest entry.

    The endpoint's network volume is already mounted at VOLUME_ROOT, and its object keys are the same paths as
    RunPod's S3-compatible API uses for that bucket — so publishing is a local copy, not a network upload. `client`
    is kept only to sign the returned URL via object_url().
    """
    bucket = os.environ["AITK_S3_BUCKET"]
    safe_name = "/".join(part for part in Path(relative_name).parts if part not in {"", ".", ".."})
    key = f"training/{session_id}/{safe_name}"
    if not VOLUME_ROOT.is_mount():
        raise RuntimeError(f"{VOLUME_ROOT} is not a mounted network volume; cannot publish worker output")
    destination = (VOLUME_ROOT / key).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "url": object_url(client, bucket, key),
        "fileName": path.name,
        "contentType": content_type,
        "fileSize": path.stat().st_size,
        "path": relative_name,
    }


def latest_loss(loss_db: Path) -> tuple[int | None, float | None]:
    """Read the latest loss value from AI Toolkit's official UI logger database."""
    if not loss_db.exists():
        return None, None
    try:
        connection = sqlite3.connect(f"file:{loss_db}?mode=ro", uri=True, timeout=2)
        try:
            key_row = connection.execute(
                "SELECT key FROM metric_keys ORDER BY CASE WHEN key = 'loss' THEN 0 WHEN lower(key) LIKE '%loss%' THEN 1 ELSE 2 END, key LIMIT 1"
            ).fetchone()
            if not key_row:
                return None, None
            row = connection.execute(
                "SELECT step, COALESCE(value_real, CAST(value_text AS REAL)) FROM metrics WHERE key = ? ORDER BY step DESC LIMIT 1",
                (key_row[0],),
            ).fetchone()
            return (int(row[0]), float(row[1])) if row and row[1] is not None else (None, None)
        finally:
            connection.close()
    except sqlite3.Error:
        return None, None


def discover_samples(
    client,
    session_id: str,
    sample_root: Path,
    uploaded: dict[Path, dict[str, Any]],
    minimum_age_seconds: float = 2,
) -> list[dict[str, Any]]:
    """Upload newly completed image, video, or audio training samples."""
    if sample_root.exists():
        for path in sorted(sample_root.iterdir()):
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in SAMPLE_EXTENSIONS or path.name.startswith("."):
                continue
            if time.time() - path.stat().st_mtime < minimum_age_seconds:
                continue
            if path not in uploaded:
                uploaded_file = upload_file(client, session_id, path, f"samples/{path.name}")
                match = STEP_PATTERN.search(path.name)
                uploaded[path] = {
                    "url": uploaded_file["url"],
                    "step": int(match.group(1)) if match else 0,
                    "epoch": None,
                    "caption": None,
                    "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)),
                }
    return list(uploaded.values())


def drain_output(stream, tail: list[str]) -> None:
    """Drain child-process output while retaining a bounded activity tail."""
    for line in iter(stream.readline, ""):
        clean = line.rstrip()
        if clean:
            tail.append(clean)
            del tail[:-100]
    stream.close()


def artifact_kind(path: Path) -> str:
    """Classify a generated file for Hartsy's artifact manifest."""
    content_type = mimetypes.guess_type(path.name)[0] or ""
    if path.suffix.lower() in {".yaml", ".yml", ".json"}:
        return "config"
    if content_type.startswith("image/"):
        return "preview_image"
    if content_type.startswith("video/"):
        return "preview_video"
    if content_type.startswith("audio/"):
        return "audio"
    return "weights"


def upload_artifacts(client, session_id: str, output_dir: Path) -> list[dict[str, Any]]:
    """Upload supported final artifacts and select the primary weights file."""
    candidates = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and (path.resolve() == output_dir or output_dir in path.resolve().parents)
        and ".tmp" not in path.parts
        and ".thumbs" not in path.parts
        and "samples" not in path.relative_to(output_dir).parts
        and path.name != "loss_log.db"
        and path.suffix.lower() in ARTIFACT_EXTENSIONS
    ]
    weight_files = [path for path in candidates if artifact_kind(path) == "weights"]
    preferred = next((path for path in weight_files if path.suffix.lower() == ".safetensors" and "optimizer" not in path.name.lower()), None)
    preferred = preferred or (weight_files[0] if weight_files else None)
    artifacts = []
    for path in sorted(candidates):
        relative = path.relative_to(output_dir).as_posix()
        uploaded = upload_file(client, session_id, path, f"artifacts/{relative}")
        uploaded["kind"] = artifact_kind(path)
        uploaded["isPrimary"] = path == preferred
        artifacts.append(uploaded)
    return artifacts


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one versioned AI Toolkit training request for RunPod Serverless."""
    payload = job.get("input") or {}
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"Unsupported contract_version; expected {CONTRACT_VERSION}")
    session_id = require_identifier(payload, "session_id")
    config_yaml = require_text(payload, "config_yaml")
    if len(config_yaml.encode("utf-8")) > 2 * 1024 * 1024:
        raise ValueError("AI Toolkit config is too large")
    if payload.get("ai_toolkit_revision") != EXPECTED_REVISION:
        raise ValueError("Hartsy and the AI Toolkit worker revisions do not match")

    validate_installed_revision()
    config = yaml.safe_load(config_yaml)
    if not isinstance(config, dict) or not isinstance(config.get("config"), dict):
        raise ValueError("config_yaml must contain a config object")
    processes = config["config"].get("process")
    if not isinstance(processes, list) or len(processes) != 1 or not isinstance(processes[0], dict):
        raise ValueError("config_yaml must contain exactly one process object")
    process = processes[0]
    validate_process_paths(process)
    total_steps = int(process.get("train", {}).get("steps", 0))
    run_name = str(config["config"]["name"])
    output_dir = (OUTPUT_ROOT / run_name).resolve()
    if output_dir != OUTPUT_ROOT and OUTPUT_ROOT not in output_dir.parents:
        raise ValueError("Unsafe AI Toolkit output path")

    reset_directory(DATASET_ROOT)
    reset_directory(OUTPUT_ROOT)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    archives = payload.get("dataset_urls") or []
    if not isinstance(archives, list) or not archives:
        raise ValueError("input.dataset_urls must contain at least one archive")
    if len(archives) > MAX_DATASET_ARCHIVES:
        raise ValueError(f"input.dataset_urls cannot contain more than {MAX_DATASET_ARCHIVES} archives")
    remaining_archive_bytes = MAX_ARCHIVE_BYTES
    remaining_extracted_bytes = MAX_EXTRACTED_BYTES
    for index, url in enumerate(archives):
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"input.dataset_urls[{index}] must be a non-empty URL")
        archive_path = WORK_ROOT / f"dataset-{index}.zip"
        try:
            downloaded = download_archive(url.strip(), archive_path, remaining_archive_bytes)
            remaining_archive_bytes -= downloaded
            extracted = extract_archive(archive_path, DATASET_ROOT, remaining_extracted_bytes)
            remaining_extracted_bytes -= extracted
        finally:
            archive_path.unlink(missing_ok=True)

    config_path = WORK_ROOT / f"{session_id}.yaml"
    config_path.write_text(config_yaml, encoding="utf-8")
    client = storage_client()
    uploaded_samples: dict[Path, dict[str, Any]] = {}
    output_tail: list[str] = []
    started_at = time.monotonic()
    command = [os.getenv("AI_TOOLKIT_PYTHON", "python"), str(TOOLKIT_ROOT / "run.py"), str(config_path)]
    process_handle = subprocess.Popen(command, cwd=TOOLKIT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process_handle.stdout is not None
    reader = threading.Thread(target=drain_output, args=(process_handle.stdout, output_tail), daemon=True)
    reader.start()
    last_step = 0
    last_loss = None
    last_update = 0.0

    try:
        while process_handle.poll() is None:
            now = time.monotonic()
            step, loss = latest_loss(output_dir / "loss_log.db")
            if step is not None:
                last_step = step
            if loss is not None:
                last_loss = loss
            samples = discover_samples(client, session_id, output_dir / "samples", uploaded_samples)
            if now - last_update >= 5:
                progress = min(99, max(1, round(last_step / total_steps * 100))) if total_steps > 0 else 1
                elapsed = now - started_at
                eta = round(elapsed / last_step * max(total_steps - last_step, 0)) if last_step > 0 and total_steps > 0 else None
                runpod.serverless.progress_update(job, {
                    "progress": progress,
                    "message": f"AI Toolkit is training · step {last_step:,} of {total_steps:,}" if last_step else "AI Toolkit is loading the model and dataset",
                    "step": last_step or None,
                    "totalSteps": total_steps or None,
                    "epoch": None,
                    "totalEpochs": None,
                    "loss": last_loss,
                    "etaSeconds": eta,
                    "samples": samples,
                    "logsTail": output_tail[-20:],
                })
                last_update = now
            time.sleep(2)

        reader.join(timeout=5)
        if process_handle.returncode != 0:
            detail = output_tail[-1] if output_tail else "AI Toolkit exited without an error message"
            raise RuntimeError(f"AI Toolkit failed with exit code {process_handle.returncode}: {detail}")

        step, loss = latest_loss(output_dir / "loss_log.db")
        if step is not None:
            last_step = step
        if loss is not None:
            last_loss = loss
        samples = discover_samples(client, session_id, output_dir / "samples", uploaded_samples, minimum_age_seconds=0)
        artifacts = upload_artifacts(client, session_id, output_dir)
        if not any(artifact["kind"] == "weights" for artifact in artifacts):
            raise RuntimeError("AI Toolkit completed without producing a model artifact")
        return {
            "sessionId": session_id,
            "aiToolkitRevision": EXPECTED_REVISION,
            "finalLoss": last_loss,
            "totalSteps": last_step or total_steps or None,
            "artifacts": artifacts,
            "samples": [sample["url"] for sample in samples],
            "sampleDetails": samples,
            "logsTail": output_tail[-20:],
        }
    except Exception as error:
        runpod.serverless.progress_update(job, {
            "progress": min(99, max(0, round(last_step / total_steps * 100))) if total_steps else 0,
            "message": "AI Toolkit training failed",
            "step": last_step or None,
            "totalSteps": total_steps or None,
            "loss": last_loss,
            "error": str(error),
            "samples": list(uploaded_samples.values()),
        })
        raise
    finally:
        if process_handle.poll() is None:
            process_handle.terminate()
            try:
                process_handle.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process_handle.kill()
                process_handle.wait()
        reader.join(timeout=5)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
