"""
RunPod Serverless AI Training Handler For AI Toolkit - By Hartsy

Production-ready handler with async RabbitMQ messaging, persistent connections,
and comprehensive progress tracking. Files stored on network volume.

Features:
- Async RabbitMQ messaging with aio-pika
- Persistent connection pooling
- Detailed progress updates with step/epoch/loss tracking
- Network volume storage for trained models
- Robust error handling and retries
- Publisher confirms for guaranteed delivery

Author: Kalebbroo - Hartsy
License: MIT
Version: 3.1
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
import aio_pika
from pathlib import Path
from typing import Dict, Any, Optional
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

class RabbitMQConnectionManager:
    """Manages persistent RabbitMQ connection with automatic reconnection."""
    
    _connection: Optional[aio_pika.Connection] = None
    _channel: Optional[aio_pika.Channel] = None
    _exchange: Optional[aio_pika.Exchange] = None
    _lock = asyncio.Lock()
    _is_initialized = False

    @classmethod
    async def initialize(cls):
        """Initialize connection manager with RabbitMQ."""
        if not RABBITMQ_URL:
            logger.warning("RabbitMQ URL not configured")
            return False
        
        async with cls._lock:
            if not cls._is_initialized:
                try:
                    cls._connection = await aio_pika.connect_robust(
                        RABBITMQ_URL,
                        heartbeat=60,
                        connection_attempts=3,
                        retry_delay=5
                    )
                    cls._is_initialized = True
                    logger.info("RabbitMQ connection manager initialized")
                    return True
                except Exception as ex:
                    logger.error(f"Failed to initialize RabbitMQ: {ex}")
                    return False
        return True

    @classmethod
    async def get_channel(cls) -> Optional[aio_pika.Channel]:
        """Get or create RabbitMQ channel with exchange declared."""
        if not cls._is_initialized:
            await cls.initialize()
        
        if not cls._connection or cls._connection.is_closed:
            await cls.initialize()
        
        async with cls._lock:
            if cls._channel is None or cls._channel.is_closed:
                cls._channel = await cls._connection.channel()
                
                # Declare exchange
                cls._exchange = await cls._channel.declare_exchange(
                    RABBITMQ_EXCHANGE,
                    aio_pika.ExchangeType.TOPIC,
                    durable=True
                )
                
                # Enable publisher confirms
                await cls._channel.set_qos(prefetch_count=1)
                
                logger.info("RabbitMQ channel and exchange ready")
        
        return cls._channel

    @classmethod
    async def get_exchange(cls) -> Optional[aio_pika.Exchange]:
        """Get exchange, ensuring channel is ready."""
        await cls.get_channel()
        return cls._exchange

    @classmethod
    async def cleanup(cls):
        """Clean up connections on shutdown."""
        async with cls._lock:
            if cls._channel and not cls._channel.is_closed:
                await cls._channel.close()
                logger.debug("Channel closed")
            
            if cls._connection and not cls._connection.is_closed:
                await cls._connection.close()
                logger.info("RabbitMQ connection closed")
            
            cls._channel = None
            cls._exchange = None
            cls._connection = None
            cls._is_initialized = False

async def publish_event(event_type: str, job_id: str, data: Dict[str, Any], max_retries: int = 3) -> bool:
    """Publishes training event to RabbitMQ with delivery confirmation.
    
    Args:
        event_type: Event type (training.started, training.progress, etc.)
        job_id: Internal job ID from Hartsy database
        data: Event payload
        max_retries: Maximum retry attempts
        
    Returns:
        bool: True if published successfully
    """
    if not RABBITMQ_URL:
        logger.warning("RabbitMQ not configured, skipping event publish")
        return False
    
    message_body = {
        'event_type': event_type,
        'job_id': job_id,
        'timestamp': datetime.utcnow().isoformat(),
        'data': data
    }
    
    for attempt in range(max_retries):
        try:
            exchange = await RabbitMQConnectionManager.get_exchange()
            if not exchange:
                raise Exception("Failed to get RabbitMQ exchange")
            
            # Create message with persistence
            message = aio_pika.Message(
                body=json.dumps(message_body).encode('utf-8'),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type='application/json',
                message_id=f"{job_id}_{int(time.time())}",
                timestamp=datetime.utcnow()
            )
            
            # Publish with confirmation
            await exchange.publish(
                message,
                routing_key=event_type,
                timeout=15.0  # 15 second timeout for confirmation
            )
            
            logger.info(f"Published event: {event_type} for job {job_id}")
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Publish timeout (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                logger.error(f"Failed to publish after {max_retries} timeout attempts")
                return False
            await asyncio.sleep(min(2 ** attempt, 10))
            
        except aio_pika.exceptions.AMQPError as ex:
            logger.error(f"AMQP error (attempt {attempt + 1}/{max_retries}): {ex}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to publish after {max_retries} AMQP errors")
                return False
            await asyncio.sleep(min(2 ** attempt, 10))
            
        except Exception as ex:
            logger.error(f"Unexpected publish error: {ex}", exc_info=True)
            if attempt == max_retries - 1:
                return False
            await asyncio.sleep(1)
    
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

def parse_progress_from_log(log_line: str, start_time: float, total_steps_estimate: int = 1000) -> Dict[str, Any]:
    """Parses comprehensive training progress from AI Toolkit log output.
    
    Enhanced to extract:
    - Step progress (current/total)
    - Epoch progress (current/total)
    - Loss values
    - Learning rate
    - GPU memory usage
    - ETA calculations
    
    Args:
        log_line: Single line from training output
        start_time: Training start timestamp
        total_steps_estimate: Estimated total steps (for early progress)
        
    Returns:
        Dict with detailed progress information
    """
    progress_info = {
        "progress": 0,
        "current_step": "Training",
        "eta_minutes": None,
        "loss": None,
        "learning_rate": None,
        "current_step_number": None,
        "total_steps": None,
        "current_epoch": None,
        "total_epochs": None,
        "gpu_memory_used_gb": None
    }
    
    try:
        # Step progress with percentage: "100/1000 [10%]" or "step 100/1000"
        step_match = re.search(r'(\d+)/(\d+)\s*\[?\s*(\d+)%?\]?|step[:\s]+(\d+)[/\s]+(\d+)', log_line, re.IGNORECASE)
        if step_match:
            if step_match.group(1):  # Format: "100/1000 [10%]"
                current = int(step_match.group(1))
                total = int(step_match.group(2))
            else:  # Format: "step 100/1000"
                current = int(step_match.group(4))
                total = int(step_match.group(5))
            
            progress_info["current_step_number"] = current
            progress_info["total_steps"] = total
            progress_info["progress"] = int((current / total) * 100)
            
            # Build detailed step description
            step_desc = f"Step {current}/{total}"
            
            # Calculate ETA based on elapsed time
            elapsed_minutes = (time.time() - start_time) / 60
            if progress_info["progress"] > 5:  # Only calculate ETA after 5% to be more accurate
                estimated_total = (elapsed_minutes / progress_info["progress"]) * 100
                eta = max(1, int(estimated_total - elapsed_minutes))
                progress_info["eta_minutes"] = eta
                step_desc += f" (~{eta}min remaining)"
            
            progress_info["current_step"] = step_desc
            return progress_info
        
        # Epoch progress: "Epoch 2/5" or "epoch: 2/5"
        epoch_match = re.search(r'epoch[:\s]+(\d+)[/\s]+(\d+)', log_line, re.IGNORECASE)
        if epoch_match:
            current = int(epoch_match.group(1))
            total = int(epoch_match.group(2))
            progress_info["current_epoch"] = current
            progress_info["total_epochs"] = total
            progress_info["progress"] = int((current / total) * 100)
            progress_info["current_step"] = f"Epoch {current}/{total}"
            
            # Rough ETA: 10-15 minutes per epoch for LoRA
            remaining_epochs = total - current
            progress_info["eta_minutes"] = remaining_epochs * 12
            return progress_info
        
        # Loss values: "loss: 0.1234" or "loss=0.1234"
        loss_match = re.search(r'loss[:\s=]+([0-9.]+)', log_line, re.IGNORECASE)
        if loss_match:
            progress_info["loss"] = float(loss_match.group(1))
            progress_info["current_step"] = f"Training (loss: {progress_info['loss']:.4f})"
        
        # Learning rate: "lr: 0.0001" or "learning_rate=0.0001"
        lr_match = re.search(r'(?:lr|learning[_\s]?rate)[:\s=]+([0-9.e-]+)', log_line, re.IGNORECASE)
        if lr_match:
            progress_info["learning_rate"] = float(lr_match.group(1))
        
        # GPU memory: "12.5GB / 24.0GB" or "GPU: 12.5/24.0"
        gpu_match = re.search(r'(\d+\.?\d*)\s*(?:GB|gb)?[/\s]+(\d+\.?\d*)\s*(?:GB|gb)?', log_line)
        if gpu_match and 'gpu' in log_line.lower():
            progress_info["gpu_memory_used_gb"] = float(gpu_match.group(1))
            
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
    """Main training execution function with comprehensive progress tracking.
    
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
        # Initialize RabbitMQ connection
        await RabbitMQConnectionManager.initialize()
        
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
        logger.info(f"Model: {model_name}")
        logger.info(f"Network volume: {NETWORK_VOLUME_PATH}")
        
        # Publish training started event
        await publish_event('training.started', job_id, {
            'session_id': session_id,
            'model_name': model_name,
            'status': 'initializing',
            'network_volume_path': str(session_dir.relative_to(NETWORK_VOLUME_PATH))
        })
        
        # Download dataset
        images_downloaded = await download_and_extract_dataset(dataset_urls[0], dataset_path)
        
        await publish_event('training.progress', job_id, {
            'status': 'preparing',
            'progress': 15,
            'current_step': f"Prepared {images_downloaded} images for training",
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
        
        await publish_event('training.progress', job_id, {
            'status': 'training',
            'progress': 20,
            'current_step': 'AI Toolkit training started'
        })
        
        # Execute training with detailed progress monitoring
        original_cwd = os.getcwd()
        os.chdir(str(ai_toolkit_dir))
        
        last_progress_update = time.time()
        progress_update_interval = 30  # Send update every 30 seconds
        last_progress_info = None
        
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
                    
                    # Parse progress from each line
                    current_time = time.time()
                    progress_info = parse_progress_from_log(line, start_time)
                    
                    # Send update if interval elapsed and we have meaningful progress
                    if current_time - last_progress_update >= progress_update_interval:
                        if progress_info["progress"] > 0 or progress_info["loss"] is not None:
                            # Calculate actual progress (20% base + 75% of training)
                            actual_progress = min(95, 20 + int(progress_info["progress"] * 0.75))
                            
                            event_data = {
                                'status': 'training',
                                'progress': actual_progress,
                                'current_step': progress_info["current_step"],
                                'current_step_number': progress_info["current_step_number"],
                                'total_steps': progress_info["total_steps"],
                                'current_epoch': progress_info["current_epoch"],
                                'total_epochs': progress_info["total_epochs"],
                                'eta_minutes': progress_info["eta_minutes"],
                                'loss': progress_info["loss"],
                                'learning_rate': progress_info["learning_rate"],
                                'gpu_memory_used_gb': progress_info["gpu_memory_used_gb"]
                            }
                            
                            await publish_event('training.progress', job_id, event_data)
                            last_progress_update = current_time
                            last_progress_info = progress_info
            
            # Get remaining output
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
                
                # Calculate total size
                total_size_bytes = sum(f.stat().st_size for f in output_files if f.is_file())
                
                logger.info(f"Generated {len(model_files)} output files ({total_size_bytes / 1024 / 1024:.1f}MB)")
                
                # Publish completion event
                await publish_event('training.completed', job_id, {
                    'status': 'completed',
                    'progress': 100,
                    'session_id': session_id,
                    'model_name': model_name,
                    'network_volume_path': str(session_dir.relative_to(NETWORK_VOLUME_PATH)),
                    'output_files': model_files,
                    'images_processed': images_downloaded,
                    'total_size_bytes': total_size_bytes,
                    'training_duration_minutes': int((time.time() - start_time) / 60),
                    'final_loss': last_progress_info["loss"] if last_progress_info else None
                })
                
                return {
                    "success": True,
                    "message": "Training completed successfully",
                    "session_id": session_id,
                    "model_name": model_name,
                    "job_id": job_id,
                    "network_volume_path": str(session_dir.relative_to(NETWORK_VOLUME_PATH)),
                    "output_files": model_files,
                    "total_size_mb": int(total_size_bytes / 1024 / 1024)
                }
            else:
                logger.error(f"Training failed with return code: {return_code}")
                
                await publish_event('training.failed', job_id, {
                    'status': 'failed',
                    'error': f"Training process exited with code {return_code}",
                    'session_id': session_id,
                    'return_code': return_code
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
        logger.error(f"Training error: {str(ex)}", exc_info=True)
        
        await publish_event('training.failed', job_id, {
            'status': 'failed',
            'error': str(ex),
            'error_type': type(ex).__name__
        })
        
        return {
            "success": False,
            "error": str(ex),
            "job_id": job_id
        }
    
    finally:
        # Cleanup RabbitMQ connection
        await RabbitMQConnectionManager.cleanup()

async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Async RunPod handler entry point."""
    try:
        logger.info("=== RunPod AI Training Handler Started ===")
        logger.info(f"Network volume: {NETWORK_VOLUME_PATH}")
        logger.info(f"RabbitMQ configured: {bool(RABBITMQ_URL)}")
        
        result = await run_training(event)
        
        logger.info("=== RunPod AI Training Handler Completed ===")
        return result
        
    except Exception as ex:
        logger.error(f"Handler error: {str(ex)}", exc_info=True)
        return {"success": False, "error": str(ex)}

if __name__ == "__main__":
    logger.info("Starting RunPod Serverless AI-Toolkit Handler v3.1")
    logger.info("Using aio-pika for async RabbitMQ messaging")
    runpod.serverless.start({"handler": handler})
