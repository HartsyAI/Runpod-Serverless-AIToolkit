"""
RunPod Serverless AI Training Handler for AI Toolkit - Hartsy Website Integration

This handler uses the Hartsy website as a storage gateway with async file uploads,
chunked upload support for large files, and callback functionality.

Features:
- Downloads dataset ZIP from website URL and extracts locally
- Async file uploads during training (2-40GB support)
- Chunked uploads with resume capability
- Real-time progress callbacks to website
- Status polling capability for website
- 5 retry attempts over 10 minutes for failed uploads

Author: Kalebbroo, Hartsy LLC
Version: 2.2
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
import aiofiles
import zipfile
import io
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import runpod
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from urllib.parse import urlparse
import math
import re
from datetime import datetime, timedelta

# Configure logging for production debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration constants
CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks for large file uploads
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2  # seconds, exponential backoff
UPLOAD_TIMEOUT = 300  # 5 minutes per chunk
CALLBACK_TIMEOUT = 30  # 30 seconds for callbacks
FILE_WATCH_DELAY = 3  # seconds to wait for file stabilization

def extract_model_name_from_config(config_content: str) -> str:
    """Extracts model name from YAML configuration for progress tracking and organization.
    
    This function parses the training configuration to determine the model name,
    which is used for organizing outputs and progress reporting. It checks multiple
    possible locations in the config hierarchy to ensure compatibility with
    different AI Toolkit configurations.
    
    Args:
        config_content: YAML configuration string from the training request
        
    Returns:
        str: Extracted model name or 'default_model' if not found
    """
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
        logger.warning(f"Could not extract model name from config: {ex}")
        return 'default_model'

def calculate_progress_and_eta(log_line: str, start_time: float) -> Dict[str, Any]:
    """Calculates training progress percentage and ETA from AI Toolkit log output.
    
    This function parses training logs to extract progress information and calculate
    estimated time remaining. It looks for common progress patterns in AI Toolkit
    output including step counts, epoch progress, and loss values.
    
    Args:
        log_line: Single line from training output
        start_time: Training start timestamp for ETA calculation
        
    Returns:
        Dict containing progress percentage, current step info, and ETA in minutes
    """
    progress_info = {"progress": 0, "current_step": "Training", "eta_minutes": None}
    
    try:
        # Look for step progress: "step 100/1000" or "Step: 100/1000"
        step_match = re.search(r'step[:\s]+(\d+)[/\s]+(\d+)', log_line, re.IGNORECASE)
        if step_match:
            current = int(step_match.group(1))
            total = int(step_match.group(2))
            progress = int((current / total) * 100)
            
            # Calculate ETA based on elapsed time and progress
            elapsed_minutes = (time.time() - start_time) / 60
            if progress > 0:
                estimated_total_minutes = (elapsed_minutes / progress) * 100
                eta_minutes = max(1, int(estimated_total_minutes - elapsed_minutes))
                progress_info["eta_minutes"] = eta_minutes
            
            progress_info["progress"] = progress
            progress_info["current_step"] = f"Step {current}/{total}"
            return progress_info
        
        # Look for epoch progress: "epoch 2/5" or "Epoch: 2/5"
        epoch_match = re.search(r'epoch[:\s]+(\d+)[/\s]+(\d+)', log_line, re.IGNORECASE)
        if epoch_match:
            current = int(epoch_match.group(1))
            total = int(epoch_match.group(2))
            progress = int((current / total) * 100)
            
            # Rough ETA estimate: 10-15 minutes per epoch for LoRA training
            remaining_epochs = total - current
            eta_minutes = remaining_epochs * 12  # Average 12 minutes per epoch
            
            progress_info["progress"] = progress
            progress_info["current_step"] = f"Epoch {current}/{total}"
            progress_info["eta_minutes"] = eta_minutes
            return progress_info
        
        # Look for loss values to indicate training is active
        if re.search(r'loss[:\s=]+[0-9.]+', log_line, re.IGNORECASE):
            progress_info["current_step"] = "Training in progress"
            return progress_info
            
    except Exception as ex:
        logger.debug(f"Error parsing progress from log line: {ex}")
    
    return progress_info

async def send_hartsy_callback(session: aiohttp.ClientSession, callback_url: str, 
                              callback_token: str, data: Dict[str, Any]) -> bool:
    """Sends callback to Hartsy website with retry logic and error handling.
    
    This function handles all communication back to the website, including progress
    updates, file upload notifications, and completion callbacks. It includes
    comprehensive retry logic with exponential backoff for reliability.
    
    Args:
        session: HTTP session for making requests
        callback_url: Website callback endpoint URL
        callback_token: Authentication token for the request
        data: Callback data to send
        
    Returns:
        bool: True if callback was sent successfully
    """
    headers = {
        'Authorization': f'Bearer {callback_token}',
        'Content-Type': 'application/json',
        'User-Agent': 'RunPod-TrainingHandler/2.2'
    }
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=CALLBACK_TIMEOUT)
            async with session.post(callback_url, json=data, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    logger.debug(f"Callback sent successfully on attempt {attempt}")
                    return True
                else:
                    error_text = await response.text()
                    logger.warning(f"Callback attempt {attempt} failed: HTTP {response.status} - {error_text}")
        
        except Exception as ex:
            logger.warning(f"Callback attempt {attempt} failed with exception: {ex}")
        
        if attempt < MAX_RETRIES:
            delay = min(RETRY_DELAY_BASE ** attempt, 60)  # Cap at 60 seconds
            logger.info(f"Retrying callback in {delay} seconds...")
            await asyncio.sleep(delay)
    
    logger.error(f"All callback attempts failed after {MAX_RETRIES} retries")
    return False

async def upload_file_chunked(session: aiohttp.ClientSession, file_path: Path,
                            upload_url: str, callback_token: str) -> bool:
    """Uploads large files using chunked upload with resume capability.
    
    This function handles uploading large model files (2-40GB) by breaking them into
    chunks and uploading with resume capability. It's designed for the large checkpoint
    and final model files generated during training.
    
    Args:
        session: HTTP session for making requests
        file_path: Local file to upload
        upload_url: Upload endpoint URL
        callback_token: Authentication token
        
    Returns:
        bool: True if upload completed successfully
    """
    if not file_path.exists():
        logger.error(f"File not found for upload: {file_path}")
        return False
    
    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    logger.info(f"Starting chunked upload: {file_path.name} ({file_size_mb:.1f}MB)")
    
    headers = {
        'Authorization': f'Bearer {callback_token}',
        'User-Agent': 'RunPod-TrainingHandler/2.2'
    }
    
    # For files under 50MB, use regular upload
    if file_size_mb < 50:
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                file_data = await f.read()
            
            form_data = aiohttp.FormData()
            form_data.add_field('file', file_data, filename=file_path.name)
            form_data.add_field('path', str(file_path.relative_to(file_path.parent.parent)))
            
            timeout = aiohttp.ClientTimeout(total=UPLOAD_TIMEOUT)
            async with session.post(upload_url, data=form_data, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    logger.info(f"Small file upload successful: {file_path.name}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Small file upload failed: HTTP {response.status} - {error_text}")
                    return False
        
        except Exception as ex:
            logger.error(f"Small file upload error: {ex}")
            return False
    
    # For large files, use chunked upload
    try:
        total_chunks = math.ceil(file_size / CHUNK_SIZE)
        
        async with aiofiles.open(file_path, 'rb') as f:
            for chunk_index in range(total_chunks):
                chunk_start = chunk_index * CHUNK_SIZE
                chunk_data = await f.read(CHUNK_SIZE)
                
                if not chunk_data:
                    break
                
                # Retry logic for each chunk
                chunk_uploaded = False
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        form_data = aiohttp.FormData()
                        form_data.add_field('chunk', chunk_data, filename=f"{file_path.name}.chunk{chunk_index}")
                        form_data.add_field('chunkIndex', str(chunk_index))
                        form_data.add_field('totalChunks', str(total_chunks))
                        form_data.add_field('fileName', file_path.name)
                        form_data.add_field('fileSize', str(file_size))
                        form_data.add_field('path', str(file_path.relative_to(file_path.parent.parent)))
                        
                        timeout = aiohttp.ClientTimeout(total=UPLOAD_TIMEOUT)
                        async with session.post(f"{upload_url}/chunk", data=form_data, headers=headers, timeout=timeout) as response:
                            if response.status == 200:
                                logger.debug(f"Chunk {chunk_index + 1}/{total_chunks} uploaded successfully")
                                chunk_uploaded = True
                                break
                            else:
                                error_text = await response.text()
                                logger.warning(f"Chunk {chunk_index + 1} attempt {attempt} failed: HTTP {response.status} - {error_text}")
                    
                    except Exception as ex:
                        logger.warning(f"Chunk {chunk_index + 1} attempt {attempt} error: {ex}")
                    
                    if attempt < MAX_RETRIES:
                        delay = min(RETRY_DELAY_BASE ** attempt, 30)
                        await asyncio.sleep(delay)
                
                if not chunk_uploaded:
                    logger.error(f"Failed to upload chunk {chunk_index + 1} after {MAX_RETRIES} attempts")
                    return False
        
        logger.info(f"Chunked upload completed: {file_path.name}")
        return True
    
    except Exception as ex:
        logger.error(f"Chunked upload error: {ex}")
        return False

class AsyncFileUploadHandler(FileSystemEventHandler):
    """File upload handler with async for large files.
    
    This class monitors the training output directory and uploads files asynchronously
    as they are created. It includes deduplication, file stabilization checking,
    and async upload capability to handle large model files without blocking training.
    
    The handler runs async upload tasks in the background while training continues,
    ensuring that large checkpoint files don't interrupt the training process.
    """
    
    def __init__(self, callback_url: str, callback_token: str, output_dir: Path, job_id: str):
        """Initializes the async file upload handler.
        
        Args:
            callback_url: Base URL for Hartsy website callbacks
            callback_token: Authentication token for callbacks
            output_dir: Directory to monitor for new files
            job_id: Training job ID for tracking and organization
        """
        super().__init__()
        self.callback_url = callback_url
        self.callback_token = callback_token
        self.output_dir = output_dir
        self.job_id = job_id
        
        # Tracking and state management
        self.uploaded_files: set = set()
        self.last_upload_times: Dict[str, float] = {}
        self.upload_executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent uploads
        
        # HTTP session for uploads (will be initialized when needed)
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Async file upload handler initialized for job {job_id}")
    
    async def initialize_session(self):
        """Initializes the HTTP session for uploads."""
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=UPLOAD_TIMEOUT)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    
    async def cleanup_session(self):
        """Cleans up the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
        self.upload_executor.shutdown(wait=True)
    
    def on_created(self, event):
        """Handles file creation events by scheduling async upload."""
        if not event.is_directory:
            # Schedule async upload task
            asyncio.create_task(self._handle_file_async(Path(event.src_path)))
    
    def on_modified(self, event):
        """Handles file modification events by scheduling async upload."""
        if not event.is_directory:
            # Schedule async upload task
            asyncio.create_task(self._handle_file_async(Path(event.src_path)))
    
    async def _handle_file_async(self, file_path: Path):
        """Handles file events asynchronously with validation and upload.
        
        This method implements the core logic for processing file system events,
        including file validation, deduplication, stabilization waiting, and
        initiating the upload process.
        
        Args:
            file_path: Path to the file that was created or modified
        """
        try:
            # Basic validation
            if not file_path.exists() or file_path.name.startswith('.'):
                return
            
            # Skip config files and temporary files
            if file_path.name.lower() in ['config.yaml', 'config.yml', '.gitkeep', 'readme.txt']:
                return
            
            file_key = str(file_path)
            current_time = time.time()
            
            # Check if already uploaded
            if file_key in self.uploaded_files:
                return
            
            # Rate limiting - don't process the same file too frequently
            last_time = self.last_upload_times.get(file_key, 0)
            if current_time - last_time < FILE_WATCH_DELAY:
                return
            
            self.last_upload_times[file_key] = current_time
            
            # Wait for file to stabilize (avoid uploading incomplete files)
            await asyncio.sleep(FILE_WATCH_DELAY)
            
            # Verify file still exists and has content
            if not file_path.exists():
                return
            
            file_size = file_path.stat().st_size
            if file_size == 0:
                logger.warning(f"Skipping empty file: {file_path.name}")
                return
            
            # Initialize session if needed
            await self.initialize_session()
            
            # Upload the file
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"Uploading file: {file_path.name} ({file_size_mb:.1f}MB)")
            
            upload_url = f"{self.callback_url}/api/training/upload-file"
            success = await upload_file_chunked(self.session, file_path, upload_url, self.callback_token)
            
            if success:
                self.uploaded_files.add(file_key)
                logger.info(f"Upload successful: {file_path.name}")
                
                # Send file upload notification to website
                await send_hartsy_callback(self.session, f"{self.callback_url}/api/training/file-uploaded", self.callback_token, {
                    "job_id": self.job_id,
                    "file_name": file_path.name,
                    "file_size_mb": file_size_mb,
                    "file_type": "checkpoint" if "checkpoint" in file_path.name.lower() else "output"
                })
            else:
                logger.error(f"Upload failed: {file_path.name}")
        
        except Exception as ex:
            logger.error(f"Error handling file {file_path}: {ex}")

class TrainingHandler:
    """Enhanced training handler that integrates with Hartsy.
    
    This class orchestrates the complete training workflow using the AI Toolkit while
    communicating with the Hartsy website for storage operations and comprehensive progress tracking.
    
    Key responsibilities:
    - Download and extract dataset ZIP from website URL
    - Configure and execute AI Toolkit training
    - Monitor training progress and calculate ETA
    - Upload results asynchronously during training
    - Send progress callbacks to website
    """
    
    def __init__(self):
        """Initializes the training handler with environment detection and validation.
        
        This constructor sets up the training environment, validates AI Toolkit
        installation, and prepares workspace directories. It automatically detects
        whether running on RunPod Serverless or Pods and configures paths accordingly.
        """
        # Environment detection - prefer /workspace for RunPod compatibility
        if Path("/workspace").exists():
            self.workspace = Path("/workspace")
            logger.info("Using /workspace (RunPod environment)")
        elif Path("/runpod-volume").exists():
            self.workspace = Path("/runpod-volume")
            logger.info("Using /runpod-volume (RunPod Serverless)")
        else:
            self.workspace = Path("/tmp")
            logger.warning("Using /tmp (fallback environment)")
        
        self.training_dir = self.workspace / "training"
        self.training_dir.mkdir(exist_ok=True)
        
        # AI Toolkit validation
        self.ai_toolkit_dir = Path("/app/ai-toolkit")
        self.run_script = self.ai_toolkit_dir / "run.py"
        
        if not self.ai_toolkit_dir.exists() or not self.run_script.exists():
            raise ValueError(f"AI Toolkit not found at expected location: {self.ai_toolkit_dir}")
        
        logger.info(f"Training handler initialized - workspace: {self.workspace}")
    
    async def download_and_extract_dataset(self, dataset_urls: List[str], local_path: Path) -> int:
        """Downloads dataset ZIP from Hartsy website and extracts images locally.
        
        This method downloads the dataset archive and extracts it for AI Toolkit training.
        It validates the extracted content and ensures proper image formats.
        
        Args:
            dataset_urls: List containing the dataset ZIP URL (uses first URL)
            local_path: Local directory to extract images to
            
        Returns:
            int: Number of successfully extracted images
            
        Raises:
            ValueError: If download fails or no valid images found
        """
        if not dataset_urls:
            raise ValueError("No dataset URLs provided")
        
        zip_url = dataset_urls[0]  # Use first URL (should be the dataset ZIP)
        logger.info(f"Downloading dataset ZIP from Hartsy: {zip_url}")
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Create HTTP session for download
        connector = aiohttp.TCPConnector(limit=5, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout for large ZIPs
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Download the ZIP file
            async with session.get(zip_url, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download dataset ZIP: HTTP {response.status}")
                
                zip_data = await response.read()
                logger.info(f"Downloaded ZIP: {len(zip_data) / 1024 / 1024:.1f}MB")
        
        # Extract ZIP (zipfile module is synchronous)
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
                zip_ref.extractall(local_path)
                logger.info(f"Extracted ZIP to {local_path}")
        except zipfile.BadZipFile as ex:
            raise ValueError(f"Invalid ZIP file: {ex}")
        
        # Count and validate extracted images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_files = [f for f in local_path.iterdir() if f.suffix.lower() in image_extensions]
        
        # Also check for caption files
        caption_files = [f for f in local_path.iterdir() if f.suffix.lower() == '.txt']
        
        if len(image_files) == 0:
            raise ValueError("No valid images found in dataset ZIP")
        
        logger.info(f"Extracted {len(image_files)} images and {len(caption_files)} caption files")
        return len(image_files)
    
    async def setup_async_file_watcher(self, output_dir: Path, callback_url: str,
                                     callback_token: str, job_id: str) -> AsyncFileUploadHandler:
        """Sets up asynchronous file monitoring and uploading for training outputs.
        
        This method creates and configures the file system watcher that monitors
        the training output directory and uploads files as they are created.
        The async nature ensures large file uploads don't block training.
        
        Args:
            output_dir: Directory to monitor for new files
            callback_url: Base URL for website callbacks
            callback_token: Authentication token for callbacks
            job_id: Training job ID for tracking
            
        Returns:
            AsyncFileUploadHandler: Configured file upload handler
        """
        handler = AsyncFileUploadHandler(callback_url, callback_token, output_dir, job_id)
        await handler.initialize_session()
        
        observer = Observer()
        observer.schedule(handler, str(output_dir), recursive=True)
        observer.start()
        
        # Store observer reference on handler for cleanup
        handler.observer = observer
        
        logger.info("Async file monitoring started")
        return handler
    
    async def run_training_async(self, config_content: str, dataset_urls: List[str],
                               callback_url: str, callback_token: str, job_id: str) -> Dict[str, Any]:
        """Executes the complete training workflow with async file uploads and progress tracking.
        
        This is the main training orchestration method that coordinates all aspects
        of the training process including dataset preparation, configuration setup,
        training execution, progress monitoring, and result uploading.
        
        Args:
            config_content: YAML training configuration
            dataset_urls: List containing dataset ZIP URL
            callback_url: Base URL for website callbacks
            callback_token: Authentication token
            job_id: Training job ID for tracking
            
        Returns:
            Dict containing training results and metadata
        """
        file_handler: Optional[AsyncFileUploadHandler] = None
        session_id = f"training_session_{int(time.time())}"
        start_time = time.time()
        
        try:
            model_name = extract_model_name_from_config(config_content)
            session_dir = self.training_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            logger.info(f"Starting training session {session_id} for job {job_id}")
            logger.info(f"Model: {model_name}")
            
            # Send initial progress update
            async with aiohttp.ClientSession() as session:
                await send_hartsy_callback(session, f"{callback_url}/api/training/progress-update", callback_token, {
                    "job_id": job_id,
                    "status": "initializing",
                    "progress": 5,
                    "current_step": "Downloading dataset",
                    "model_name": model_name
                })
            
            # Download and extract dataset from Hartsy ZIP URL
            dataset_path = session_dir / "dataset"
            successful_downloads = await self.download_and_extract_dataset(dataset_urls, dataset_path)
            
            # Update progress
            async with aiohttp.ClientSession() as session:
                await send_hartsy_callback(session, f"{callback_url}/api/training/progress-update", callback_token, {
                    "job_id": job_id,
                    "status": "preparing",
                    "progress": 15,
                    "current_step": f"Extracted {successful_downloads} images, preparing training",
                    "images_downloaded": successful_downloads
                })
            
            # Setup training configuration
            config_file = session_dir / "config.yaml"
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            config_data = yaml.safe_load(config_content)
            
            # Update config with local dataset path
            if 'config' in config_data and 'process' in config_data['config']:
                for process in config_data['config']['process']:
                    if 'datasets' in process:
                        for i, dataset in enumerate(process['datasets']):
                            process['datasets'][i]['folder_path'] = str(dataset_path)
            
            # Setup output directory
            output_dir = session_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Update config with local output path
            for process in config_data['config']['process']:
                if 'training_folder' in process:
                    process['training_folder'] = str(output_dir)
            
            # Write updated config
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            # Setup async file monitoring
            file_handler = await self.setup_async_file_watcher(output_dir, callback_url, callback_token, job_id)
            
            # Update progress
            async with aiohttp.ClientSession() as session:
                await send_hartsy_callback(session, f"{callback_url}/api/training/progress-update", callback_token, {
                    "job_id": job_id,
                    "status": "training",
                    "progress": 20,
                    "current_step": "Starting AI Toolkit training",
                })
            
            # Execute training
            cmd = ["python", str(self.run_script), str(config_file)]
            logger.info(f"Starting training: {' '.join(cmd)}")
            
            original_cwd = os.getcwd()
            os.chdir(str(self.ai_toolkit_dir))
            
            last_progress_update = time.time()
            progress_update_interval = 30  # seconds
            
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
                
                # Monitor training output with progress tracking
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    
                    if output:
                        line = output.rstrip()
                        print(line, flush=True)
                        
                        # Extract progress and send periodic updates
                        current_time = time.time()
                        if current_time - last_progress_update >= progress_update_interval:
                            progress_info = calculate_progress_and_eta(line, start_time)
                            
                            if progress_info["progress"] > 0:
                                async with aiohttp.ClientSession() as session:
                                    await send_hartsy_callback(session, f"{callback_url}/api/training/progress-update", callback_token, {
                                        "job_id": job_id,
                                        "status": "training",
                                        "progress": min(95, 20 + int(progress_info["progress"] * 0.75)),  # Scale to 20-95%
                                        "current_step": progress_info["current_step"],
                                        "eta_minutes": progress_info["eta_minutes"]
                                    })
                            
                            last_progress_update = current_time
                
                # Get final output
                remaining_output = process.stdout.read()
                if remaining_output:
                    print(remaining_output, flush=True)
                
                return_code = process.poll()
                
                if return_code == 0:
                    logger.info("Training completed successfully")
                    
                    # Wait for final file uploads
                    logger.info("Waiting for final file uploads")
                    await asyncio.sleep(10)
                    
                    # Send completion callback
                    async with aiohttp.ClientSession() as session:
                        await send_hartsy_callback(session, f"{callback_url}/api/training/job-complete", callback_token, {
                            "job_id": job_id,
                            "status": "completed",
                            "progress": 100,
                            "current_step": "Training completed successfully",
                            "model_name": model_name,
                            "session_id": session_id,
                            "total_files_uploaded": len(file_handler.uploaded_files) if file_handler else 0
                        })
                    
                    return {
                        "success": True,
                        "message": "Training completed successfully",
                        "session_id": session_id,
                        "model_name": model_name,
                        "job_id": job_id,
                        "images_downloaded": successful_downloads,
                        "files_uploaded": len(file_handler.uploaded_files) if file_handler else 0
                    }
                else:
                    logger.error(f"Training failed with return code: {return_code}")
                    
                    # Send failure callback
                    async with aiohttp.ClientSession() as session:
                        await send_hartsy_callback(session, f"{callback_url}/api/training/job-complete", callback_token, {
                            "job_id": job_id,
                            "status": "failed",
                            "progress": 0,
                            "error": f"Training process failed with return code {return_code}",
                            "session_id": session_id
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
            logger.error(f"Error in training process: {str(ex)}")
            
            # Send error callback
            try:
                async with aiohttp.ClientSession() as session:
                    await send_hartsy_callback(session, f"{callback_url}/api/training/job-complete", callback_token, {
                        "job_id": job_id,
                        "status": "failed",
                        "progress": 0,
                        "error": str(ex),
                        "session_id": session_id
                    })
            except:
                pass  # Don't let callback errors mask the original error
            
            return {
                "success": False,
                "error": str(ex),
                "session_id": session_id,
                "job_id": job_id
            }
        
        finally:
            # Cleanup file handler
            if file_handler:
                try:
                    if hasattr(file_handler, 'observer'):
                        file_handler.observer.stop()
                        file_handler.observer.join()
                    await file_handler.cleanup_session()
                except Exception as ex:
                    logger.warning(f"Error cleaning up file handler: {ex}")

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler function enhanced for Hartsy website integration.
    
    This function serves as the entry point for all training requests. It validates
    the input, configures the training environment, and orchestrates the complete
    training workflow using the async training handler.
    
    Expected input format from Hartsy website:
    {
        "internal_job_id": "123",  // Database ID from TrainingTable
        "config": "yaml_config_string",
        "dataset_urls": ["https://storage.supabase.com/.../dataset.zip"],
        "callback_base_url": "https://hartsy.com",
        "callback_token": "temp-job-token-for-auth"
    }
    
    Args:
        event: RunPod event containing input data and configuration
        
    Returns:
        Dict containing training results, status, and metadata
    """
    try:
        logger.info("=== RunPod AI Training Handler Started ===")
        
        input_data = event.get("input", {})
        if not input_data:
            raise ValueError("No input data provided")
        
        # Validate required fields
        required_fields = ["internal_job_id", "config", "dataset_urls", "callback_base_url", "callback_token"]
        for field in required_fields:
            if field not in input_data:
                error_msg = f"Missing required field: {field}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
        
        # Validate dataset URLs
        dataset_urls = input_data.get("dataset_urls", [])
        if not isinstance(dataset_urls, list) or len(dataset_urls) == 0:
            error_msg = "At least one dataset URL is required"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Validate internal job ID (should be a string representation of the database ID)
        internal_job_id = str(input_data["internal_job_id"])
        if not internal_job_id or internal_job_id == "None":
            error_msg = "Valid internal job ID is required"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Validate configuration
        try:
            yaml.safe_load(input_data["config"])
        except yaml.YAMLError as ex:
            error_msg = f"Invalid training configuration YAML: {ex}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        logger.info(f"Processing training job {internal_job_id} with dataset ZIP: {dataset_urls[0]}")
        
        # Create training handler and run training
        training_handler = TrainingHandler()
        result = asyncio.run(training_handler.run_training_async(
            input_data["config"],
            dataset_urls,
            input_data["callback_base_url"],
            input_data["callback_token"],
            internal_job_id
        ))
        
        logger.info("=== RunPod AI Training Handler Completed ===")
        return result
    
    except ValueError as ve:
        error_msg = f"Input validation error: {ve}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    except Exception as ex:
        error_msg = f"Handler error: {str(ex)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg}

if __name__ == "__main__":
    logger.info("Starting RunPod Serverless AI-Toolkit Handler v2.2")
    runpod.serverless.start({"handler": handler})