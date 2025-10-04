"""
RunPod Serverless AI Training Handler For AI Toolkit - By Hartsy

Simplified handler that focuses on training execution and event publishing.
Files are stored on RunPod network volume and events sent to RabbitMQ for processing.

Features:
- Trains models using AI Toolkit
- Saves outputs to persistent network volume
- Publishes training events to RabbitMQ
- No file uploads during training (handled separately)
- Simple, reliable and easy to scale

Author: Kalebbroo - Hartsy
License: MIT
Version: 3.0
"""

import json
import os
import subprocess
import logging
import yaml
import time
import sys
import asyncio
import aiohttp
import zipfile
import io
import pika
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import runpod
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration
NETWORK_VOLUME_PATH = Path(os.environ.get('NETWORK_VOLUME_PATH', '/runpod-volume'))
RABBITMQ_URL = os.environ.get('RABBITMQ_URL', '')
RABBITMQ_EXCHANGE = os.environ.get('RABBITMQ_EXCHANGE', 'hartsy.training')

def publish_event(event_type: str, job_id: str, data: Dict[str, Any]) -> bool:
    """Publishes training event to RabbitMQ for processing by Hartsy.
    
    Args:
        event_type: Type of event (training.started, training.progress, etc.)
        job_id: Internal job ID from Hartsy database
        data: Event payload
        
    Returns:
        bool: True if published successfully
    """
    if not RABBITMQ_URL:
        logger.warning("RabbitMQ URL not configured, skipping event publish")
        return False
    
    try:
        connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        channel = connection.channel()
        channel.exchange_declare(
            exchange=RABBITMQ_EXCHANGE,
            exchange_type='topic',
            durable=True
        )
        message = {
            'event_type': event_type,
            'job_id': job_id,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        channel.basic_publish(
            exchange=RABBITMQ_EXCHANGE,
            routing_key=event_type,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent
                content_type='application/json'
            )
        )
        connection.close()
        logger.info(f"Published event: {event_type} for job {job_id}")
        return True
    except Exception as ex:
        logger.error(f"Failed to publish event {event_type}: {ex}")
        return False

def extract_model_name_from_config(config_content: str) -> str:
    """Extracts model name from YAML configuration."""
    try:
        config_data = yaml.safe_load(config_content)
        model_name = (
            config_data.get('config', {}).get('name') or
            config_data.get('name') or
            config_data.get('model_name') or
            'default_model'
        )
        logger.info(f"Extracted model name: {model_name}")
        return model_name
    except Exception as ex:
        logger.warning(f"Could not extract model name: {ex}")
        return 'default_model'

def parse_progress_from_log(log_line: str, start_time: float) -> Dict[str, Any]:
    """Parses training progress from AI Toolkit log output.
    
    Args:
        log_line: Single line from training output
        start_time: Training start timestamp
        
    Returns:
        Dict with progress info (progress, current_step, eta_minutes, etc.)
    """
    progress_info = {
        "progress": 0,
        "current_step": "Training",
        "eta_minutes": None,
        "loss": None,
        "current_step_number": None,
        "total_steps": None,
        "current_epoch": None,
        "total_epochs": None
    }
    try:
        # Step progress: "step 100/1000" by matching "step X/Y"
        step_match = re.search(r'step[:\s]+(\d+)[/\s]+(\d+)', log_line, re.IGNORECASE)
        if step_match:
            current = int(step_match.group(1))
            total = int(step_match.group(2))
            progress_info["current_step_number"] = current
            progress_info["total_steps"] = total
            progress_info["progress"] = int((current / total) * 100)
            progress_info["current_step"] = f"Step {current}/{total}"
            # Calculate ETA based on elapsed time
            elapsed_minutes = (time.time() - start_time) / 60
            if progress_info["progress"] > 0:
                estimated_total = (elapsed_minutes / progress_info["progress"]) * 100
                eta = max(1, int(estimated_total - elapsed_minutes))
                progress_info["eta_minutes"] = eta
            return progress_info
        # Epoch progress: "epoch 2/5" by matching "epoch X/Y"
        epoch_match = re.search(r'epoch[:\s]+(\d+)[/\s]+(\d+)', log_line, re.IGNORECASE)
        if epoch_match:
            current = int(epoch_match.group(1))
            total = int(epoch_match.group(2))
            progress_info["current_epoch"] = current
            progress_info["total_epochs"] = total
            progress_info["progress"] = int((current / total) * 100)
            progress_info["current_step"] = f"Epoch {current}/{total}"
            progress_info["eta_minutes"] = (total - current) * 12  # Rough estimate
            return progress_info
        # Loss values "loss=0.1234". A loss value indicates how training is progressing.
        loss_match = re.search(r'loss[:\s=]+([0-9.]+)', log_line, re.IGNORECASE)
        if loss_match:
            progress_info["loss"] = float(loss_match.group(1))
            progress_info["current_step"] = "Training in progress"
            return progress_info
    except Exception as ex:
        logger.debug(f"Error parsing progress: {ex}")
    return progress_info

async def download_and_extract_dataset(dataset_url: str, local_path: Path) -> int:
    """Downloads dataset ZIP and extracts images.
    Args:
        dataset_url: Public URL to dataset ZIP file
        local_path: Local directory to extract to
    Returns:
        int: Number of images extracted
    """
    logger.info(f"Downloading dataset from: {dataset_url}")
    local_path.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        async with session.get(dataset_url, timeout=aiohttp.ClientTimeout(total=300)) as response:
            if response.status != 200:
                raise ValueError(f"Failed to download dataset: HTTP {response.status}")
            zip_data = await response.read()
            logger.info(f"Downloaded ZIP: {len(zip_data) / 1024 / 1024:.1f}MB")
    # Extract ZIP
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
        zip_ref.extractall(local_path)
        logger.info(f"Extracted ZIP to {local_path}")
    # Count images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = [f for f in local_path.iterdir() if f.suffix.lower() in image_extensions]
    logger.info(f"Extracted {len(image_files)} images")
    return len(image_files)

async def run_training(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main training execution function.
    Args:
        event: RunPod event with training configuration
    Returns:
        Dict with training results
    """
    input_data = event.get("input", {})
    # Validate inputs
    required_fields = ["internal_job_id", "config", "dataset_urls"]
    for field in required_fields:
        if field not in input_data:
            return {"success": False, "error": f"Missing required field: {field}"}
    job_id = str(input_data["internal_job_id"])
    config_content = input_data["config"]
    dataset_urls = input_data["dataset_urls"]
    if not isinstance(dataset_urls, list) or len(dataset_urls) == 0:
        return {"success": False, "error": "At least one dataset URL is required"}
    try:
        model_name = extract_model_name_from_config(config_content)
        session_id = f"training_session_{int(time.time())}"
        start_time = time.time()
        # Setup paths on network volume
        session_dir = NETWORK_VOLUME_PATH / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = session_dir / "dataset"
        output_dir = session_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)      
        logger.info(f"Starting training session {session_id} for job {job_id}")
        # Publish training started event
        publish_event('training.started', job_id, {
            'session_id': session_id,
            'model_name': model_name,
            'status': 'initializing'
        })
        # Download dataset
        images_downloaded = await download_and_extract_dataset(dataset_urls[0], dataset_path)
        publish_event('training.progress', job_id, {
            'status': 'preparing',
            'progress': 15,
            'current_step': f"Extracted {images_downloaded} images, preparing training",
            'images_downloaded': images_downloaded
        })
        # Setup configuration
        config_file = session_dir / "config.yaml"
        config_data = yaml.safe_load(config_content)
        # Update paths in config
        if 'config' in config_data and 'process' in config_data['config']:
            for process in config_data['config']['process']:
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = str(dataset_path)
                if 'training_folder' in process:
                    process['training_folder'] = str(output_dir)
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        # Start training
        ai_toolkit_dir = Path("/app/ai-toolkit")
        run_script = ai_toolkit_dir / "run.py"
        if not run_script.exists():
            raise ValueError(f"AI Toolkit not found at {ai_toolkit_dir}")
        cmd = ["python", str(run_script), str(config_file)]
        logger.info(f"Starting training: {' '.join(cmd)}")
        publish_event('training.progress', job_id, {
            'status': 'training',
            'progress': 20,
            'current_step': 'Starting AI Toolkit training'
        })
        # Execute training with progress monitoring
        original_cwd = os.getcwd()
        os.chdir(str(ai_toolkit_dir))
        last_progress_update = time.time()
        progress_update_interval = 30  # Send update every 30 seconds
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            logger.info("Training process started")
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.rstrip()
                    print(line, flush=True)
                    # Parse and send progress updates
                    current_time = time.time()
                    if current_time - last_progress_update >= progress_update_interval:
                        progress_info = parse_progress_from_log(line, start_time)
                        if progress_info["progress"] > 0:
                            publish_event('training.progress', job_id, {
                                'status': 'training',
                                'progress': min(95, 20 + int(progress_info["progress"] * 0.75)),
                                'current_step': progress_info["current_step"],
                                'current_step_number': progress_info["current_step_number"],
                                'total_steps': progress_info["total_steps"],
                                'current_epoch': progress_info["current_epoch"],
                                'total_epochs': progress_info["total_epochs"],
                                'eta_minutes': progress_info["eta_minutes"],
                                'loss': progress_info["loss"]
                            })
                            last_progress_update = current_time
            remaining_output = process.stdout.read()
            if remaining_output:
                print(remaining_output, flush=True)
            return_code = process.poll()
            if return_code == 0:
                logger.info("Training completed successfully")
                # List output files
                output_files = list(output_dir.rglob('*'))
                model_files = [str(f.relative_to(NETWORK_VOLUME_PATH)) 
                             for f in output_files if f.is_file()]
                # Publish completion event
                publish_event('training.completed', job_id, {
                    'status': 'completed',
                    'progress': 100,
                    'session_id': session_id,
                    'model_name': model_name,
                    'network_volume_path': str(session_dir.relative_to(NETWORK_VOLUME_PATH)),
                    'output_files': model_files,
                    'images_processed': images_downloaded
                })
                return {
                    "success": True,
                    "message": "Training completed successfully",
                    "session_id": session_id,
                    "model_name": model_name,
                    "job_id": job_id,
                    "network_volume_path": str(session_dir.relative_to(NETWORK_VOLUME_PATH)),
                    "output_files": model_files
                }
            else:
                logger.error(f"Training failed with return code: {return_code}")
                publish_event('training.failed', job_id, {
                    'status': 'failed',
                    'error': f"Training process failed with return code {return_code}",
                    'session_id': session_id
                })
                return {
                    "success": False,
                    "error": f"Training process failed with return code {return_code}",
                    "session_id": session_id,
                    "job_id": job_id
                }
        finally:
            os.chdir(original_cwd)
    except Exception as ex:
        logger.error(f"Training error: {str(ex)}")
        publish_event('training.failed', job_id, {
            'status': 'failed',
            'error': str(ex)
        })
        return {
            "success": False,
            "error": str(ex),
            "job_id": job_id
        }

async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod handler entry point."""
    try:
        logger.info("=== RunPod AI Training Handler Started ===")
        return await run_training(event)
    except Exception as ex:
        logger.error(f"Handler error: {str(ex)}", exc_info=True)
        return {"success": False, "error": str(ex)}

if __name__ == "__main__":
    logger.info("Starting RunPod Serverless AI-Toolkit Handler v3.0")
    runpod.serverless.start({"handler": handler})
