# Hartsy AI Toolkit worker

This repository owns the worker image used by HartsyWeb's AI Toolkit backend. It turns the official AI Toolkit command-line runner into Hartsy's versioned RunPod Serverless contract. Both the container and each request must match `AIToolkitCatalog.UpstreamRevision`; mismatches are rejected before a billable training process starts.

Build and push from this directory:

```bash
docker build --pull --build-arg AI_TOOLKIT_REVISION=be995185f598c83abb990a088e9f634c4d36eb46 -t your-registry/hartsy-ai-toolkit:be995185 .
docker push your-registry/hartsy-ai-toolkit:be995185
```

Create a queue-based RunPod Serverless endpoint from that immutable image. Use one concurrent request per worker because `/dataset` and `/workspace/output` are intentionally reset for every request. Allocate enough container disk for the model cache plus dataset/output and choose a GPU configuration appropriate for the largest model families you enable. Hartsy sends a 24-hour execution policy and seven-day status TTL by default; the endpoint must permit those limits.

Required worker environment:

- `AITK_S3_BUCKET`, `AITK_S3_ACCESS_KEY`, `AITK_S3_SECRET_KEY`
- `AITK_S3_ENDPOINT` for S3-compatible storage, when not using AWS
- `AITK_S3_REGION` (default `us-east-1`)
- `AITK_S3_PUBLIC_BASE_URL` for public objects, or omit it to return seven-day signed URLs
- `AITK_DATASET_ALLOWED_ORIGINS`, a comma-separated list of exact HTTPS origins Hartsy may use for prepared archives, for example `https://storage.hartsy.ai,https://example.r2.cloudflarestorage.com`

Optional limits: `AITK_MAX_ARCHIVE_BYTES`, `AITK_MAX_EXTRACTED_BYTES`, `AITK_MAX_DATASET_ARCHIVES`, `AITK_URL_TTL_SECONDS`, `AITK_WORK_ROOT`, and `AI_TOOLKIT_PYTHON`.

Hartsy needs `RUNPOD_API_KEY` and `RUNPOD_AI_TOOLKIT_ENDPOINT_ID`. It can override `AI_TOOLKIT_EXECUTION_TIMEOUT_MS` and `AI_TOOLKIT_JOB_TTL_MS` within RunPod's seven-day maximum.

The worker validates its installed AI Toolkit revision and all worker-owned paths, safely downloads and extracts Hartsy's prepared dataset, launches unmodified `run.py`, reads the official `loss_log.db`, uploads new samples as they appear, emits structured RunPod progress updates, and returns an artifact manifest for durable Hartsy ingestion.

## Versioned request contract

HartsyWeb submits contract version 2 using RunPod's `/run` endpoint:

```json
{
  "input": {
    "contract_version": 2,
    "internal_job_id": "12345",
    "session_id": "aitk-12345-1788050000",
    "config_yaml": "config:\n  name: example\n  process: [...]",
    "dataset_urls": ["https://storage.example/signed-dataset.zip"],
    "ai_toolkit_revision": "be995185f598c83abb990a088e9f634c4d36eb46"
  },
  "policy": {
    "executionTimeout": 86400000,
    "ttl": 604800000
  }
}
```

Progress updates are published through `runpod.serverless.progress_update()` and include `progress`, `step`, `totalSteps`, `loss`, `etaSeconds`, `samples`, and `logsTail`. HartsyWeb polls the RunPod status endpoint, persists loss/sample history, and relays it to the training UI.

A completed job returns:

```json
{
  "sessionId": "aitk-12345-1788050000",
  "aiToolkitRevision": "be995185f598c83abb990a088e9f634c4d36eb46",
  "finalLoss": 0.1842,
  "totalSteps": 3000,
  "artifacts": [
    {
      "url": "https://storage.example/training/example.safetensors",
      "fileName": "example.safetensors",
      "contentType": "application/octet-stream",
      "fileSize": 123456789,
      "path": "artifacts/example.safetensors",
      "kind": "weights",
      "isPrimary": true
    }
  ],
  "samples": ["https://storage.example/training/sample_1000.png"],
  "sampleDetails": []
}
```

## Safety and limits

- Dataset downloads and every redirect must use an origin in `AITK_DATASET_ALLOWED_ORIGINS`; the complete request is capped at 8 archives, 4 GiB compressed, and 8 GiB extracted by default.
- Custom S3 endpoints, public object bases, and every returned artifact URL must use HTTPS.
- ZIP path traversal, symbolic links, duplicate output paths, unsafe session IDs, and dataset/config paths outside worker-owned roots are rejected.
- The container base image, AI Toolkit commit, and worker dependencies are pinned. The handler also verifies the installed revision at runtime.
- The endpoint must use one concurrent request per worker. Each request resets `/dataset` and `/workspace/output`.
- Samples and final artifacts are uploaded to S3-compatible storage before their URLs are returned to HartsyWeb.

## Test locally

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m py_compile handler.py tests/test_handler.py
docker build --check .
```

These tests cover revision matching, path ownership, secure ZIP extraction, the official AI Toolkit UI logger database schema, and final-sample discovery.

## Migration from the legacy worker

Contract version 2 replaces the legacy RabbitMQ/network-volume response schema. Do not deploy this worker until the paired HartsyWeb AI Toolkit changes are deployed. After the image is pushed, update the existing RunPod endpoint template to the immutable image tag and keep the endpoint ID in `RUNPOD_AI_TOOLKIT_ENDPOINT_ID`.
