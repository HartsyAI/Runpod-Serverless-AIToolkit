# Runpod Serverless Handler for AI Toolkit

## Overview

This is a **custom-built handler** for a RunPod serverless endpoint that orchestrates AI model training with automated file management. It integrates with Supabase for storage and uses the AI Toolkit for training execution.

**Key Features:**

- Downloads training datasets from Supabase storage
- Executes AI training using subprocess calls to AI Toolkit
- Real-time file monitoring and uploading during training
- Comprehensive error handling and logging
- Streams training logs to RunPod console


## Architecture Components

### 1. **Environment Setup \& Configuration**

```python
def init_supabase_client() -> Client:
```

- **Purpose**: Initializes Supabase client for storage operations
- **Dependencies**: Requires `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` environment variables
- **Location**: AI Toolkit expected at `/app/ai-toolkit/`


### 2. **Dataset Management**

```python
def download_dataset(self, bucket_name: str, dataset_folder: str, local_path: Path):
```

- **Purpose**: Downloads training datasets from Supabase storage bucket
- **Process**: Lists files in specified folder, downloads each file locally
- **Storage**: Files saved to `/workspace/training/{session_id}/dataset/`


### 3. **Real-time File Upload System**

```python
class FileUploadHandler(FileSystemEventHandler):
```

- **Technology**: Uses Python `watchdog` library for file system monitoring
- **Functionality**:
    - Monitors output directory recursively
    - Debounces file events (3-second delay between uploads of same file)
    - Excludes config.yaml files from upload
    - Maintains upload state to prevent duplicates


### 4. **Training Process Management**

```python
def run_training(self, config_content: str, dataset_config: Dict, upload_config: Dict):
```

- **Process Flow**:

1. Creates unique session directory
2. Downloads dataset files
3. Updates training configuration with local paths
4. Starts file watcher for real-time uploads
5. Executes training via subprocess
6. Streams all output to console
7. Handles cleanup and final file uploads


### 5. **API Handler Interface**

```python
def handler(event):
```

- **Input Validation**: Checks for required fields (`config`, `dataset_config`, `upload_config`)
- **Error Handling**: Returns structured JSON responses
- **Integration**: Uses RunPod SDK for serverless functionality


## Usage

### Environment Variables Required

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_key_here
```


### Expected Input Format

```json
{
  "config": "yaml_config_string",
  "dataset_config": {
    "bucket_name": "training-datasets",
    "folder_path": "my-dataset-folder"
  },
  "upload_config": {
    "bucket_name": "training-outputs", 
    "folder_path": "training-session-123456"
  }
}
```


### Response Format

```json
{
  "success": true,
  "message": "Training completed successfully",
  "session_id": "training_session_1234567890",
  "model_name": "my_model",
  "upload_folder": "training-session-123456",
  "output_path": "/workspace/training/training_session_1234567890/output"
}
```


## File Organization

### Local Structure (During Training)

```
/workspace/training/{session_id}/
├── config.yaml          # Training configuration
├── dataset/             # Downloaded dataset files
│   ├── image1.jpg
│   └── image2.jpg
└── output/              # Training outputs (monitored)
    ├── samples/         # Generated sample images
    ├── checkpoints/     # Model checkpoints
    └── logs/           # Training logs
```


### Supabase Storage Structure

```
{upload_folder}/
├── {model_name}/
│   ├── samples/
│   │   ├── sample_001.png
│   │   └── sample_002.png
│   ├── checkpoints/
│   │   └── model.safetensors
│   └── logs/
│       └── training.log
```


## Technical Details

### Dependencies

- `runpod` - RunPod Python SDK for serverless functionality
- `supabase` - Supabase Python client for storage operations
- `watchdog` - File system monitoring for real-time uploads
- `yaml` - YAML configuration parsing
- `pathlib` - Modern path handling


### Custom Implementation Notes

- **Fully custom code** - Not based on RunPod templates
- **Real-time uploads** - Uses file system events, not polling
- **Subprocess management** - Direct integration with AI Toolkit


## RunPod Integration

This handler leverages:

- **RunPod Serverless SDK** ([docs](https://docs.runpod.io/serverless)) - For endpoint management
- **Custom worker architecture** - Built from scratch for specific training workflow
- **No GitHub integration used** - All code and data managed through API calls


## Related Documentation

- [RunPod Serverless Documentation](https://docs.runpod.io/serverless)
- [RunPod Python SDK](https://github.com/runpod/runpod-python)
- [Supabase Storage API](https://supabase.com/docs/guides/storage)
- [Python Watchdog Library](https://pythonhosted.org/watchdog/)

***

*This is a production-ready, custom-built solution for automated AI training with integrated storage management on RunPod's serverless infrastructure.*
<span style="display:none">
