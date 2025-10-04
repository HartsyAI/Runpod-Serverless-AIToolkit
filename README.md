<div align="center">

# Hartsy RunPod Serverless AI-Toolkit Handler

**Minimalist training orchestrator with persistent storage and RabbitMQ events**

</div>

> [!NOTE]
> This repository exposes the exact handler Hartsy runs in production. It shows how we bootstrap AI Toolkit jobs on RunPod serverless, persist every artifact on the mounted network volume, and broadcast lifecycle events through RabbitMQ. **Fork freely**, but assume you will swap credentials, exchange names, and storage conventions for non-Hartsy installs.

---

## Why We Built It This Way

- **Persistent network volume (`NETWORK_VOLUME_PATH`)** keeps datasets, configs, and outputs available after the worker exits, simplifying post-processing without uploading during the run.
- **RabbitMQ fan-out (`publish_event()`)** lets the site react to `training.started`, `training.progress`, `training.completed`, and `training.failed` events asynchronously, so webhooks, notifications, and asset pipelines can scale independently.
- **Single-purpose handler**: `handler.py` focuses on dataset prep, AI Toolkit invocation, and message publishing—nothing else. Simplicity reduces failure surface area on ephemeral RunPod workers.

> [!TIP]
> Because storage happens on RunPod's shared volume, the same session folder can be reused by downstream jobs (e.g., rendering UI previews, zipping outputs, archiving).


## Architecture Snapshot

| Category | Details |
| --- | --- |
| Runtime | RunPod Serverless worker (Python 3.10) |
| Entrypoint | `handler.py:handler()` |
| Training backend | `/app/ai-toolkit/run.py` (AI Toolkit CLI) |
| Storage | Mounted network volume (default `/runpod-volume`) |
| Messaging | RabbitMQ topic exchange (`hartsy.training` by default) |
| Dataset format | Single ZIP accessible through `dataset_urls[0]` |


## How the Handler Flows

- **Ingest**: `handler(event)` validates the payload and launches `run_training()`.
- **Prepare**: `download_and_extract_dataset()` unpacks the ZIP into `session_dir/dataset` on the network volume.
- **Train**: AI Toolkit is executed via `subprocess.Popen`, with live log parsing by `parse_progress_from_log()`.
- **Persist**: Outputs stay inside `session_dir/output`, referenced relative to the network volume root.
- **Signal**: `publish_event()` emits updates to RabbitMQ, letting Hartsy services handle UI updates, asset uploads, or retries.

> [!WARNING]
> If RabbitMQ credentials are absent (`RABBITMQ_URL` unset) the handler keeps running but logs a warning and skips message delivery. Ensure your deployment fails fast or injects sane defaults.


## System Flow at a Glance

```mermaid
flowchart TD
    A[Hartsy Site] -->|Invoke RunPod endpoint| B[Serverless Handler]
    B --> C[Download dataset ZIP to network volume]
    C --> D[Update YAML & launch AI Toolkit]
    D --> E[Train & read logs]
    E --> F[Publish RabbitMQ events]
    D --> G[Write outputs to session folder]
    G --> H[Persistent storage for downstream jobs]

```


## Code Highlights (`handler.py`)

- **`publish_event()`** establishes a RabbitMQ connection per call, declares the topic exchange, and sends JSON payloads flagged as persistent messages.
- **`parse_progress_from_log()`** inspects AI Toolkit stdout for `step`, `epoch`, and `loss` patterns to compute percentage and ETA before emitting `training.progress` events.
- **`run_training()`** is the primary orchestrator: builds session directories on the volume, mutates YAML paths, spawns the AI Toolkit subprocess, captures return codes, and aggregates output file paths for the final response.
- **`handler()`** simply wraps the async routine behind `asyncio.run()` to satisfy RunPod's serverless invocation contract.


## Directory Layout on the Network Volume

```
/runpod-volume/
└── training_session_<epoch>/
    ├── config.yaml          # YAML rewritten with local paths
    ├── dataset/             # Extracted dataset ZIP contents
    └── output/              # AI Toolkit artifacts (checkpoints, logs, samples)
```

All relative paths returned in the API response (e.g., `output_files`) are anchored against `NETWORK_VOLUME_PATH`, so downstream services can mount the same volume and read assets in place.


## Quick Start for RunPod Serverless

### 1. Create RunPod Serverless Endpoint
- **Base Image**: Use `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` or similar with AI Toolkit pre-installed
- **Start Command**: Copy the bash command from Option 3 below
- **Network Volume**: Attach persistent storage and set mount path to `/runpod-volume`

### 2. Required Environment Variables
Set these in your RunPod endpoint configuration:

| Variable | Required | Example Value | Purpose |
| --- | --- | --- | --- |
| `NETWORK_VOLUME_PATH` | Yes | `/runpod-volume` | Mount path for persistent storage |
| `RABBITMQ_URL` | Optional | `amqp://user:pass@host:5672/` | RabbitMQ connection for event publishing |
| `RABBITMQ_EXCHANGE` | Optional | `hartsy.training` | Topic exchange for training events |

### 3. Dependencies
The handler requires: `runpod`, `aiohttp`, `pika`, `PyYAML` (installed automatically by start command)

> [!IMPORTANT]  
> RunPod serverless tasks must attach a network volume for persistent storage. Without it, training outputs are lost when the worker exits.


## Deployment Options

### Option 1: Start Command Installation (Recommended for RunPod Serverless)
For RunPod serverless endpoints, use this start command in your template settings:

```bash
bash -c "echo 'Installing dependencies...' && pip install runpod aiohttp pika PyYAML && echo 'Dependencies installed' && mkdir -p /workspace && cd /workspace && echo 'Downloading handler...' && wget -O handler.py https://raw.githubusercontent.com/HartsyAI/Runpod-Serverless-AIToolkit/refs/heads/main/handler.py && echo 'Handler downloaded successfully' && ls -la handler.py && echo 'Starting handler...' && python handler.py"
```

**Why these dependencies:**
- `runpod` - RunPod serverless framework
- `aiohttp` - Async HTTP client for dataset downloads  
- `pika` - RabbitMQ client for training event publishing
- `PyYAML` - YAML configuration parsing

> [!IMPORTANT]
> **Dependencies must be installed in the RunPod environment**, not in the handler code itself. The handler doesn't install packages.
> 
> This command downloads the latest handler from GitHub each time, ensuring you always run the current version without manual uploads.

### Option 2: Using requirements.txt
1. **Install dependencies**: Use the `requirements.txt` file in RunPod's template setup
2. **Mount volume**: Attach a persistent network volume at `/runpod-volume`
3. **Environment variables**: Set `RABBITMQ_URL`, `RABBITMQ_EXCHANGE`, `NETWORK_VOLUME_PATH`
4. **Deploy**: Upload `handler.py` and set entry point to `handler`

### Option 3: Custom Docker Image
1. **Build image**: `docker build -t your-registry/ai-training-handler .`
2. **Push image**: `docker push your-registry/ai-training-handler`
3. **Create endpoint**: Use your custom image in RunPod template
4. **Configure**: Set environment variables and mount volume

```jsonc
{
  "input": {
    "internal_job_id": "12345",               // Hartsy DB identifier
    "config": "...YAML string...",            // Raw AI Toolkit configuration
    "dataset_urls": ["https://hartsy.com/.../dataset.zip"],
    "callback_base_url": "https://hartsy.com",
    "callback_token": "<temporary-auth-token>"
  }
}
```

- **`dataset_urls`**: Only the first entry is used; must be a public or signed URL accessible from RunPod.
- **`internal_job_id`**: Primary key from Hartsy's DB, echoed in every RabbitMQ message.
- **`config`**: Full AI Toolkit YAML. Paths are auto-rewritten to the session directories.


## Output Contract (Handler → RunPod client)

Successful run example:

```jsonc
{
  "success": true,
  "message": "Training completed successfully",
  "session_id": "training_session_1727895321",
  "model_name": "hartsy-dreambooth",
  "job_id": "12345",
  "images_downloaded": 58,
  "files_uploaded": 7
}
```

Failure responses bubble up validation or runtime issues:

```jsonc
{
  "success": false,
  "error": "Training process failed with return code 1",
  "session_id": "training_session_1727895321",
  "job_id": "12345"
}
```


## Messaging Surface (Handler → RabbitMQ)

- **`training.started`**: Fired once the session directory is created and config parsed.
- **`training.progress`**: Sent during dataset extraction and periodically while training (every ~30 seconds when parseable progress is found).
- **`training.completed`**: Includes relative paths to artifacts on the network volume.
- **`training.failed`**: Fired on non-zero exit codes or exceptions; message body includes error text.

All messages share the structure:

```jsonc
{
  "event_type": "training.progress",
  "job_id": "12345",
  "timestamp": "2025-10-03T22:05:12.123456",
  "data": {
    "status": "training",
    "progress": 65,
    "current_step": "Epoch 3/5",
    "eta_minutes": 14
  }
}
```

> [!TIP]
> Downstream consumers can bind queues to `hartsy.training` using wildcard routing keys (`training.*`) to receive the full lifecycle stream.


## Swapping RabbitMQ for API Calls (or Other Transports)

- **Where to change**: Replace `publish_event()` with your transport of choice (HTTP, gRPC, SNS). The rest of the handler calls that helper.
- **Keep payloads consistent**: Maintaining the same `data` schema keeps Hartsy site integrations compatible.
- **Rate limiting**: If switching to synchronous HTTP APIs, consider batching progress updates or increasing `progress_update_interval` to avoid flooding endpoints.
- **Error handling**: `publish_event()` currently logs failures and continues. For APIs, you may want to implement retries or mark the whole run as failed when messaging breaks.


## Running Locally (Debug Harness)

- **Mirror**: Use the same container image as the RunPod worker so `/app/ai-toolkit/run.py` exists.
- **Mount**: Create a local folder and set `NETWORK_VOLUME_PATH` to that path.
- **RabbitMQ**: Point `RABBITMQ_URL` at a dev broker (e.g., Docker `rabbitmq:3-management`).
- **Invoke**: Run `python handler.py` and deliver a JSON payload via `run_training()` or mimic RunPod by calling `handler(event)`.


## Triggering the Handler via RunPod API

> [!EXAMPLE]
> Issue a RunPod **sync run** with the Hartsy payload. Replace `ENDPOINT_ID` and tokens with your own values.

```bash
curl https://api.runpod.ai/v2/ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{
        "input": {
          "internal_job_id": "12345",
          "config": "'"'"your_yaml_config_here"'"'",
          "dataset_urls": ["https://hartsy.com/api/training/downloads/dataset.zip"],
          "callback_base_url": "https://hartsy.com",
          "callback_token": "<temporary-auth-token>"
        }
      }'
```

JavaScript (Fetch) variant:

```javascript
await fetch("https://api.runpod.ai/v2/ENDPOINT_ID/run", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${process.env.RUNPOD_API_KEY}`
  },
  body: JSON.stringify({
    input: {
      internal_job_id: "12345",
      config: yamlString,
      dataset_urls: ["https://hartsy.com/.../dataset.zip"],
      callback_base_url: "https://hartsy.com",
      callback_token: temporaryToken
    }
  })
});
```


## Further Reading

- RunPod Serverless docs: <https://docs.runpod.io/serverless>
- RabbitMQ tutorials: <https://www.rabbitmq.com/getstarted.html>
- Pika client docs: <https://pika.readthedocs.io/>
- AI Toolkit documentation (Hartsy internal)
- Aiohttp user guide: <https://docs.aiohttp.org/>

---

_Maintained by Hartsy LLC • Version 3.0 of the serverless training handler_

