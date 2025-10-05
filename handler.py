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
- Sample image detection and broadcasting

Author: Kalebbroo - Hartsy
License: MIT
Version: 3.2
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
RABBITMQ_EXCHANGE = os.environ.get('RABBITMQ_EXCHANGE', 'training.events')
BACKEND_ID = os.environ.get('BACKEND_ID', 'ai-toolkit')  # Static backend identifier

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
                # Declare exchange (publishers only need the exchange)
                cls._exchange = await cls._channel.declare_exchange(
                    RABBITMQ_EXCHANGE,
                    aio_pika.ExchangeType.TOPIC,
                    durable=True
                )
                logger.info(f"RabbitMQ channel and exchange '{RABBITMQ_EXCHANGE}' ready")
                # Enable publisher confirms for guaranteed delivery
                await cls._channel.set_qos(prefetch_count=1)
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
        data: Event payload - will be merged with base message fields
        max_retries: Maximum retry attempts
        
    Returns:
        bool: True if published successfully
    """
    if not RABBITMQ_URL:
        logger.warning("RabbitMQ not configured, skipping event publish")
        return False
    
    # Build message body with base fields + data payload
    message_body = {
        'job_id': job_id,
        'backend_id': BACKEND_ID,
        'event_type': event_type,
        'timestamp': datetime.utcnow().isoformat(),
        **data  # Merge data fields at top level
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
    - Loss values
    - Learning rate
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
        "estimated_minutes_remaining": None,
        "loss": None,
        "learning_rate": None,
        "current_step_number": None,
        "total_steps": None
    }
    
    try:
        # Step progress: "614/1000 [22:47<13:31, 2.10s/it, lr: 1.0e-06 loss: 3.866e-01]"
        step_match = re.search(r'(\d+)/(\d+)\s*\[', log_line)
        if step_match:
            current = int(step_match.group(1))
            total = int(step_match.group(2))
            
            progress_info["current_step_number"] = current
            progress_info["total_steps"] = total
            progress_info["progress"] = int((current / total) * 100)
            
            # Extract loss if present
            loss_match = re.search(r'loss:\s*([0-9.e-]+)', log_line)
            if loss_match:
                progress_info["loss"] = float(loss_match.group(1))
            
            # Extract learning rate if present
            lr_match = re.search(r'lr:\s*([0-9.e-]+)', log_line)
            if lr_match:
                progress_info["learning_rate"] = float(lr_match.group(1))
            
            # Build detailed step description
            step_desc = f"Step {current}/{total}"
            if progress_info["loss"] is not None:
                step_desc += f" (loss: {progress_info['loss']:.4f})"
            
            # Calculate ETA from time remaining in log: "[22:47<13:31, ...]"
            time_match = re.search(r'<(\d+):(\d+)', log_line)
            if time_match:
                minutes = int(time_match.group(1))
                seconds = int(time_match.group(2))
                progress_info["estimated_minutes_remaining"] = minutes + (1 if seconds > 30 else 0)
                step_desc += f" (~{progress_info['estimated_minutes_remaining']}min remaining)"
            
            progress_info["current_step"] = step_desc
            return progress_info
            
    except Exception as ex:
        logger.debug(f"Error parsing progress: {ex}")
    
    return progress_info

def detect_sample_image(log_line: str, output_dir: Path) -> Optional[str]:
    """Detects when AI Toolkit generates a sample image.
    
    Args:
        log_line: Log line to check
        output_dir: Output directory where samples are saved
        
    Returns:
        Path to generated image if detected, None otherwise
    """
    # AI Toolkit saves samples to output/samples/ directory
    if "Generating Images:" in log_line and "100%" in log_line:
        # Sample generation completed - find the most recent image
        samples_dir = output_dir / "samples"
        if samples_dir.exists():
            image_files = sorted(
                samples_dir.glob("*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if image_files:
                return str(image_files[0].relative_to(NETWORK_VOLUME_PATH))
    return None

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
        session_id = f"session_{int(time.time())}"  # For logging/identification
        start_time = time.time()
        
        # Setup paths on network volume using job_id for easy API retrieval
        # Structure: /runpod-volume/jobs/{job_id}/
        #   ├── dataset/           (training images)
        #   ├── output/
        #   │   ├── samples/       (preview images generated during training)
        #   │   └── *.safetensors  (final model files)
        #   └── config.yaml
        # This allows simple API calls: GET /api/training/{job_id}/samples
        job_dir = NETWORK_VOLUME_PATH / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = job_dir / "dataset"
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training job {job_id} (session {session_id})")
        logger.info(f"Model: {model_name}")
        logger.info(f"Job directory: {job_dir}")
        
        # Publish training started event
        await publish_event('training.started', job_id, {
            'session_id': session_id,
            'model_name': model_name,
            'status': 'initializing',
            'network_volume_path': f"jobs/{job_id}"
        })
        
        # Download dataset
        images_downloaded = await download_and_extract_dataset(dataset_urls[0], dataset_path)
        
        await publish_event('training.progress', job_id, {
            'status': 'preparing',
            'progress': 15,
            'current_step': f"Prepared {images_downloaded} images for training",
            'metadata': {
                'images_downloaded': images_downloaded
            }
        })
        
        # Setup configuration
        config_file = job_dir / "config.yaml"
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
        sample_count = 0
        
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
                    
                    # Check for sample image generation
                    sample_image_path = detect_sample_image(line, output_dir)
                    if sample_image_path:
                        sample_count += 1
                        current_step = last_progress_info["current_step_number"] if last_progress_info else 0
                        
                        await publish_event('training.testimage', job_id, {
                            'image_url': sample_image_path,
                            'step_number': current_step,
                            'caption': f"Training sample at step {current_step}",
                            'metadata': {
                                'sample_index': sample_count,
                                'network_path': sample_image_path
                            }
                        })
                        logger.info(f"Published test image event: {sample_image_path}")
                    
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
                                'estimated_minutes_remaining': progress_info["estimated_minutes_remaining"],
                                'loss': progress_info["loss"],
                                'metadata': {
                                    'learning_rate': progress_info["learning_rate"]
                                }
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
                    'session_id': session_id,
                    'model_name': model_name,
                    'network_volume_path': f"jobs/{job_id}",
                    'output_files': model_files,
                    'images_processed': images_downloaded,
                    'message': 'Training completed successfully',
                    'metadata': {
                        'total_size_bytes': total_size_bytes,
                        'training_duration_minutes': int((time.time() - start_time) / 60),
                        'final_loss': last_progress_info["loss"] if last_progress_info else None,
                        'total_samples_generated': sample_count
                    }
                })
                return {
                    "success": True,
                    "message": "Training completed successfully",
                    "session_id": session_id,
                    "model_name": model_name,
                    "job_id": job_id,
                    "network_volume_path": f"jobs/{job_id}",
                    "output_files": model_files,
                    "total_size_mb": int(total_size_bytes / 1024 / 1024)
                }
            else:
                logger.error(f"Training failed with return code: {return_code}")
                await publish_event('training.failed', job_id, {
                    'status': 'failed',
                    'error': f"Training process exited with code {return_code}",
                    'message': 'Training failed',
                    'metadata': {
                        'session_id': session_id,
                        'return_code': return_code
                    }
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
            'message': 'Training failed with exception',
            'metadata': {
                'error_type': type(ex).__name__
            }
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
        logger.info(f"Backend ID: {BACKEND_ID}")
        logger.info(f"Network volume: {NETWORK_VOLUME_PATH}")
        logger.info(f"RabbitMQ configured: {bool(RABBITMQ_URL)}")
        result = await run_training(event)
        logger.info("=== RunPod AI Training Handler Completed ===")
        return result
    except Exception as ex:
        logger.error(f"Handler error: {str(ex)}", exc_info=True)
        return {"success": False, "error": str(ex)}

if __name__ == "__main__":
    logger.info(f"Starting RunPod Serverless AI-Toolkit Handler v3.2")
    logger.info(f"Backend ID: {BACKEND_ID}")
    logger.info("Using aio-pika for async RabbitMQ messaging")
    runpod.serverless.start({"handler": handler})