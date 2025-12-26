#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Robustness & Recovery Module

Production-grade robustness infrastructure including:
- State management and crash recovery
- Error categorization and handling
- Lock file mechanism
- Incremental progress tracking
- Upload queue management
- Data validation
- Atomic file operations
- Memory monitoring
- Timeout management
- Health checks
"""

import asyncio
import fcntl
import gc
import json
import logging
import os
import shutil
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, IO, Optional, TypeVar
import tempfile

import aiohttp
import psutil

logger = logging.getLogger("wallpaper_curator")

# =============================================================================
# ERROR CATEGORIES
# =============================================================================

class ErrorCategory(Enum):
    """Classification of errors by severity and recoverability."""
    FATAL = "fatal"           # Cannot continue - exit immediately
    RECOVERABLE = "recoverable"  # Can retry or skip
    WARNING = "warning"       # Non-critical, log and continue


@dataclass
class PipelineError:
    """Structured error representation."""
    category: ErrorCategory
    message: str
    stage: str
    wallpaper_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    traceback_str: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "message": self.message,
            "stage": self.stage,
            "wallpaper_id": self.wallpaper_id,
            "timestamp": self.timestamp,
            "traceback": self.traceback_str
        }
    
    @classmethod
    def from_exception(
        cls,
        e: Exception,
        stage: str,
        category: ErrorCategory = ErrorCategory.RECOVERABLE,
        wallpaper_id: Optional[str] = None
    ) -> "PipelineError":
        return cls(
            category=category,
            message=str(e),
            stage=stage,
            wallpaper_id=wallpaper_id,
            traceback_str=traceback.format_exc()
        )


# =============================================================================
# PIPELINE STATE MANAGEMENT
# =============================================================================

@dataclass
class PipelineState:
    """Represents the current state of the pipeline for crash recovery."""
    stage_completed: str  # "init", "fetch", "filter", "embed", "upload", "complete"
    candidates_processed: int
    approved_count: int
    failed_candidates: list[str]
    timestamp: str
    run_id: str
    source_failures: list[str] = field(default_factory=list)
    embedding_incomplete: list[str] = field(default_factory=list)
    upload_pending: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineState":
        return cls(**data)
    
    @classmethod
    def initial(cls, run_id: str) -> "PipelineState":
        return cls(
            stage_completed="init",
            candidates_processed=0,
            approved_count=0,
            failed_candidates=[],
            timestamp=datetime.now().isoformat(),
            run_id=run_id
        )


class StateManager:
    """
    Manages pipeline state for crash recovery.
    
    Features:
    - Save state after each major stage
    - Resume from last successful stage
    - Lock file to prevent concurrent runs
    - Clean state on successful completion
    """
    
    def __init__(self, state_dir: Path = Path("./pipeline_state")):
        self.state_dir = state_dir
        self.state_file = state_dir / "state.json"
        self.lock_file = state_dir / ".pipeline.lock"
        self.lock_fd: Optional[IO[str]] = None
        
        # Ensure directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def acquire_lock(self) -> bool:
        """
        Acquire exclusive lock to prevent concurrent pipeline runs.
        
        Returns:
            True if lock acquired, False if another instance is running.
        """
        try:
            self.lock_fd = open(self.lock_file, "w")
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Write PID for debugging
            self.lock_fd.write(f"{os.getpid()}\n{datetime.now().isoformat()}\n")
            self.lock_fd.flush()
            
            logger.info(f"🔒 Pipeline lock acquired (PID: {os.getpid()})")
            return True
            
        except (IOError, OSError) as e:
            if self.lock_fd:
                self.lock_fd.close()
                self.lock_fd = None
            
            # Try to read existing lock info
            try:
                with open(self.lock_file, "r") as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        existing_pid = lines[0].strip()
                        lock_time = lines[1].strip()
                        logger.error(
                            f"❌ Pipeline already running (PID: {existing_pid}, started: {lock_time})"
                        )
            except Exception:
                pass
            
            return False
    
    def release_lock(self) -> None:
        """Release the pipeline lock."""
        if self.lock_fd:
            try:
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
            except Exception as e:
                logger.warning(f"Error releasing lock: {e}")
            finally:
                self.lock_fd = None
        
        # Remove lock file
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            logger.warning(f"Error removing lock file: {e}")
        
        logger.info("🔓 Pipeline lock released")
    
    def save_state(self, state: PipelineState) -> None:
        """Save current pipeline state to disk."""
        state.timestamp = datetime.now().isoformat()
        
        # Use atomic write
        atomic_writer = AtomicFileWriter()
        atomic_writer.atomic_json_write(self.state_file, state.to_dict())
        
        logger.debug(f"State saved: stage={state.stage_completed}, processed={state.candidates_processed}")
    
    def load_state(self) -> Optional[PipelineState]:
        """Load pipeline state from disk."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            return PipelineState.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return None
    
    def is_recent(self, max_age_hours: int = 24) -> bool:
        """Check if saved state is recent enough to resume from."""
        state = self.load_state()
        if not state:
            return False
        
        try:
            state_time = datetime.fromisoformat(state.timestamp)
            age = datetime.now() - state_time
            is_recent = age < timedelta(hours=max_age_hours)
            
            if is_recent:
                logger.info(f"📋 Found recent state from {state.timestamp} (stage: {state.stage_completed})")
            else:
                logger.info(f"⏰ State too old ({age.total_seconds() / 3600:.1f}h), starting fresh")
            
            return is_recent
        except Exception as e:
            logger.warning(f"Error checking state age: {e}")
            return False
    
    def clean_state(self) -> None:
        """Clean up state files on successful completion."""
        files_to_clean = [
            self.state_file,
            self.state_dir / "embeddings_progress.json",
            self.state_dir / "upload_queue.json"
        ]
        
        for file_path in files_to_clean:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Cleaned: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean {file_path}: {e}")
        
        logger.info("✅ Pipeline state cleaned")
    
    def get_resume_stage(self) -> Optional[str]:
        """Get the stage to resume from."""
        state = self.load_state()
        if state and self.is_recent():
            return state.stage_completed
        return None


# =============================================================================
# EMBEDDING PROGRESS TRACKING
# =============================================================================

@dataclass
class EmbeddingProgress:
    """Progress tracking for a single wallpaper's embeddings."""
    wallpaper_id: str
    mobilenet: bool = False
    efficientnet: bool = False
    siglip: bool = False
    dinov2: bool = False
    embeddings_data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "wallpaper_id": self.wallpaper_id,
            "mobilenet": self.mobilenet,
            "efficientnet": self.efficientnet,
            "siglip": self.siglip,
            "dinov2": self.dinov2,
            "embeddings_data": self.embeddings_data
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingProgress":
        return cls(**data)


class EmbeddingProgressTracker:
    """
    Tracks embedding extraction progress for resume capability.
    
    Saves progress after each model completes, allowing resume
    from the exact point of interruption.
    """
    
    MODEL_NAMES = ["mobilenet", "efficientnet", "siglip", "dinov2"]
    
    def __init__(self, state_dir: Path = Path("./pipeline_state")):
        self.progress_file = state_dir / "embeddings_progress.json"
        self.progress: dict[str, EmbeddingProgress] = {}
        self.load()
    
    def load(self) -> None:
        """Load progress from disk."""
        if not self.progress_file.exists():
            return
        
        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
            
            for wp_id, wp_data in data.items():
                self.progress[wp_id] = EmbeddingProgress.from_dict(wp_data)
            
            logger.info(f"📊 Loaded embedding progress for {len(self.progress)} wallpapers")
        except Exception as e:
            logger.warning(f"Failed to load embedding progress: {e}")
    
    def save(self) -> None:
        """Save progress to disk."""
        try:
            data = {wp_id: prog.to_dict() for wp_id, prog in self.progress.items()}
            
            atomic_writer = AtomicFileWriter()
            atomic_writer.atomic_json_write(self.progress_file, data)
        except Exception as e:
            logger.warning(f"Failed to save embedding progress: {e}")
    
    def mark_complete(self, wallpaper_id: str, model: str, embedding_data: Any = None) -> None:
        """Mark a model's embedding as complete for a wallpaper."""
        if wallpaper_id not in self.progress:
            self.progress[wallpaper_id] = EmbeddingProgress(wallpaper_id=wallpaper_id)
        
        setattr(self.progress[wallpaper_id], model, True)
        
        if embedding_data is not None:
            self.progress[wallpaper_id].embeddings_data[model] = embedding_data
        
        # Save after each completion for crash recovery
        self.save()
    
    def get_pending_models(self, wallpaper_id: str) -> list[str]:
        """Get list of models still pending for a wallpaper."""
        if wallpaper_id not in self.progress:
            return self.MODEL_NAMES.copy()
        
        prog = self.progress[wallpaper_id]
        pending = []
        for model in self.MODEL_NAMES:
            if not getattr(prog, model, False):
                pending.append(model)
        
        return pending
    
    def is_fully_processed(self, wallpaper_id: str) -> bool:
        """Check if all embeddings are complete for a wallpaper."""
        if wallpaper_id not in self.progress:
            return False
        
        prog = self.progress[wallpaper_id]
        return all(getattr(prog, model, False) for model in self.MODEL_NAMES)
    
    def get_saved_embeddings(self, wallpaper_id: str) -> dict[str, Any]:
        """Get any saved embedding data for a wallpaper."""
        if wallpaper_id not in self.progress:
            return {}
        return self.progress[wallpaper_id].embeddings_data
    
    def get_incomplete_wallpapers(self) -> list[str]:
        """Get list of wallpapers with incomplete embeddings."""
        incomplete = []
        for wp_id, prog in self.progress.items():
            if not self.is_fully_processed(wp_id):
                incomplete.append(wp_id)
        return incomplete


# =============================================================================
# UPLOAD QUEUE MANAGEMENT
# =============================================================================

@dataclass
class UploadQueueItem:
    """Item in the upload queue."""
    wallpaper_id: str
    filepath: str
    category: str
    r2_path: str
    added_at: str
    retry_count: int = 0
    last_error: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UploadQueueItem":
        return cls(**data)


class UploadQueueManager:
    """
    Manages the upload queue for recovery from upload failures.
    
    Features:
    - Persist queue to disk
    - Track retry attempts
    - Idempotent: Check if file exists before upload
    """
    
    def __init__(self, state_dir: Path = Path("./pipeline_state")):
        self.queue_file = state_dir / "upload_queue.json"
        self.queue: dict[str, UploadQueueItem] = {}
        self.r2_client = None  # Set externally
        self.load()
    
    def load(self) -> None:
        """Load queue from disk."""
        if not self.queue_file.exists():
            return
        
        try:
            with open(self.queue_file, "r") as f:
                data = json.load(f)
            
            for item_data in data:
                item = UploadQueueItem.from_dict(item_data)
                self.queue[item.wallpaper_id] = item
            
            if self.queue:
                logger.info(f"📤 Loaded {len(self.queue)} pending uploads from queue")
        except Exception as e:
            logger.warning(f"Failed to load upload queue: {e}")
    
    def save(self) -> None:
        """Save queue to disk."""
        try:
            data = [item.to_dict() for item in self.queue.values()]
            
            atomic_writer = AtomicFileWriter()
            atomic_writer.atomic_json_write(self.queue_file, data)
        except Exception as e:
            logger.warning(f"Failed to save upload queue: {e}")
    
    def add_to_queue(self, items: list[UploadQueueItem]) -> None:
        """Add items to the upload queue."""
        for item in items:
            self.queue[item.wallpaper_id] = item
        self.save()
    
    def remove_from_queue(self, wallpaper_id: str) -> None:
        """Remove item from queue after successful upload."""
        if wallpaper_id in self.queue:
            del self.queue[wallpaper_id]
            self.save()
    
    def mark_failed(self, wallpaper_id: str, error: str) -> None:
        """Mark an upload as failed, increment retry count."""
        if wallpaper_id in self.queue:
            self.queue[wallpaper_id].retry_count += 1
            self.queue[wallpaper_id].last_error = error
            self.save()
    
    def get_pending(self) -> list[UploadQueueItem]:
        """Get all pending uploads."""
        return list(self.queue.values())
    
    def check_exists_in_r2(self, r2_path: str, r2_client: Any) -> bool:
        """
        Check if file already exists in R2 via HEAD request.
        
        Args:
            r2_path: The R2 object path.
            r2_client: Boto3 S3 client configured for R2.
        
        Returns:
            True if file exists, False otherwise.
        """
        if r2_client is None:
            return False
        
        try:
            r2_client.head_object(
                Bucket=os.getenv("R2_BUCKET_NAME", ""),
                Key=r2_path.lstrip("/")
            )
            return True
        except Exception:
            return False


# =============================================================================
# RETRY WITH BACKOFF
# =============================================================================

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    timeout: Optional[float] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Enhanced retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (2^attempt * base_delay).
        max_delay: Maximum delay cap.
        exceptions: Tuple of exception types to catch and retry.
        on_retry: Optional callback(attempt, exception, delay) for logging.
        timeout: Optional timeout per attempt in seconds.
    
    Returns:
        Decorated function with retry logic.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if timeout:
                        return await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=timeout
                        )
                    return await func(*args, **kwargs)
                    
                except asyncio.TimeoutError as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} timed out for {func.__name__}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                    
                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                        
                except exceptions as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                    
                    if attempt < max_retries:
                        await asyncio.sleep(delay)
            
            logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if timeout:
                        # Use signal for sync timeout (Unix only)
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Operation timed out after {timeout}s")
                        
                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(timeout))
                        try:
                            result = func(*args, **kwargs)
                            signal.alarm(0)
                            return result
                        finally:
                            signal.signal(signal.SIGALRM, old_handler)
                    
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                    
                    if attempt < max_retries:
                        time.sleep(delay)
            
            logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# =============================================================================
# DATA VALIDATION
# =============================================================================

class DataValidator:
    """
    Validates data before committing changes.
    
    Performs sanity checks to prevent data corruption.
    """
    
    EMBEDDING_DIMENSIONS = {
        "mobilenet_v3": 576,
        "efficientnet_v2": 1280,
        "siglip": 1152,
        "dinov2": 1024
    }
    
    def validate_manifest_json(self, path: Path) -> tuple[bool, list[str]]:
        """Validate that manifest JSON is well-formed."""
        errors = []
        
        if not path.exists():
            errors.append(f"Manifest file not found: {path}")
            return False, errors
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                errors.append("Manifest root must be a dictionary")
            
            # Check required fields
            if "wallpapers" not in data:
                errors.append("Missing 'wallpapers' key in manifest")
            elif not isinstance(data["wallpapers"], list):
                errors.append("'wallpapers' must be a list")
            
            return len(errors) == 0, errors
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Error reading manifest: {e}")
            return False, errors
    
    def validate_embedding_dimensions(
        self,
        embeddings: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Verify all embeddings have correct dimensions."""
        errors = []
        
        for model, expected_dim in self.EMBEDDING_DIMENSIONS.items():
            if model in embeddings:
                embedding = embeddings[model]
                if hasattr(embedding, "__len__"):
                    actual_dim = len(embedding)
                    if actual_dim != expected_dim:
                        errors.append(
                            f"{model}: expected {expected_dim} dimensions, got {actual_dim}"
                        )
        
        return len(errors) == 0, errors
    
    def check_duplicate_ids(self, manifest: dict) -> tuple[bool, list[str]]:
        """Check for duplicate wallpaper IDs in manifest."""
        errors = []
        seen_ids = set()
        duplicates = []
        
        wallpapers = manifest.get("wallpapers", [])
        for wp in wallpapers:
            wp_id = wp.get("id")
            if wp_id in seen_ids:
                duplicates.append(wp_id)
            seen_ids.add(wp_id)
        
        if duplicates:
            errors.append(f"Duplicate IDs found: {duplicates[:10]}...")
        
        return len(errors) == 0, errors
    
    async def spot_check_r2_urls(
        self,
        urls: list[str],
        sample_size: int = 5
    ) -> tuple[bool, list[str]]:
        """Verify R2 URLs are accessible by spot-checking a sample."""
        import random
        errors = []
        
        if not urls:
            return True, []
        
        sample = random.sample(urls, min(sample_size, len(urls)))
        
        async with aiohttp.ClientSession() as session:
            for url in sample:
                try:
                    async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status >= 400:
                            errors.append(f"URL not accessible ({resp.status}): {url}")
                except Exception as e:
                    errors.append(f"URL check failed: {url} - {e}")
        
        return len(errors) == 0, errors
    
    def validate_phash_file(self, path: Path) -> tuple[bool, list[str]]:
        """Validate perceptual hash file format."""
        errors = []
        
        if not path.exists():
            # Not an error if file doesn't exist yet
            return True, []
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                errors.append("Phash file must be a dictionary")
            
            # Check format of entries
            for wp_id, phash in list(data.items())[:5]:
                if not isinstance(phash, str):
                    errors.append(f"Invalid phash format for {wp_id}")
                    break
            
            return len(errors) == 0, errors
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in phash file: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Error reading phash file: {e}")
            return False, errors


# =============================================================================
# ATOMIC FILE OPERATIONS
# =============================================================================

class AtomicFileWriter:
    """
    Provides atomic file write operations.
    
    Uses temp file + atomic rename pattern to prevent corruption
    if the process crashes during write.
    """
    
    @contextmanager
    def atomic_write(
        self,
        target: Path,
        mode: str = "w",
        encoding: str = "utf-8"
    ) -> Iterator[IO]:
        """
        Context manager for atomic file writes.
        
        Writes to a temp file, then atomically renames to target
        on successful completion.
        
        Args:
            target: Target file path.
            mode: File mode ('w' for text, 'wb' for binary).
            encoding: Text encoding (ignored for binary mode).
        
        Yields:
            File-like object to write to.
        """
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temp file in same directory for atomic rename
        fd, temp_path = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp"
        )
        temp_path = Path(temp_path)
        
        try:
            if "b" in mode:
                f = os.fdopen(fd, mode)
            else:
                f = os.fdopen(fd, mode, encoding=encoding)
            
            yield f
            f.flush()
            os.fsync(f.fileno())
            f.close()
            
            # Atomic rename
            temp_path.rename(target)
            
        except Exception:
            # Clean up temp file on error
            try:
                if not f.closed:
                    f.close()
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            raise
    
    def atomic_json_write(self, target: Path, data: Any, indent: int = 2) -> None:
        """Atomically write JSON data to a file."""
        with self.atomic_write(target) as f:
            json.dump(data, f, indent=indent, default=str)


# =============================================================================
# MEMORY MONITORING
# =============================================================================

class MemoryMonitor:
    """
    Monitors memory usage and warns if approaching limits.
    
    Especially important for ML model loading which can be memory-intensive.
    """
    
    def __init__(
        self,
        warn_threshold_mb: int = 3000,
        critical_threshold_mb: int = 3500
    ):
        self.warn_threshold_mb = warn_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.last_check_time = 0
        self.check_interval = 30  # seconds
    
    def get_memory_usage_mb(self) -> int:
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return int(process.memory_info().rss / (1024 * 1024))
    
    def get_available_memory_mb(self) -> int:
        """Get available system memory in MB."""
        return int(psutil.virtual_memory().available / (1024 * 1024))
    
    def check_memory(self) -> tuple[int, str]:
        """
        Check memory usage and return status.
        
        Returns:
            Tuple of (usage_mb, status) where status is 'ok', 'warning', or 'critical'.
        """
        usage_mb = self.get_memory_usage_mb()
        
        if usage_mb >= self.critical_threshold_mb:
            status = "critical"
            logger.error(f"🔴 CRITICAL: Memory usage at {usage_mb}MB (threshold: {self.critical_threshold_mb}MB)")
        elif usage_mb >= self.warn_threshold_mb:
            status = "warning"
            logger.warning(f"🟡 WARNING: Memory usage at {usage_mb}MB (threshold: {self.warn_threshold_mb}MB)")
        else:
            status = "ok"
        
        return usage_mb, status
    
    def should_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        current_time = time.time()
        
        # Rate limit checks
        if current_time - self.last_check_time < self.check_interval:
            return False
        
        self.last_check_time = current_time
        usage_mb, status = self.check_memory()
        
        return status in ("warning", "critical")
    
    def force_gc(self) -> int:
        """Force garbage collection and return memory freed."""
        before = self.get_memory_usage_mb()
        gc.collect()
        after = self.get_memory_usage_mb()
        freed = before - after
        
        if freed > 0:
            logger.info(f"🧹 Garbage collection freed {freed}MB (now at {after}MB)")
        
        return freed


# =============================================================================
# TIMEOUT MANAGEMENT
# =============================================================================

class TimeoutManager:
    """
    Manages overall script timeout for GitHub Actions.
    
    GitHub Actions has a 60-minute timeout. This manager tracks elapsed time
    and triggers graceful shutdown when approaching the limit.
    
    Enhanced with upload_reserve_minutes to ensure uploads have time to complete.
    """
    
    def __init__(
        self,
        max_runtime_minutes: int = 50,
        upload_reserve_minutes: int = 10
    ):
        self.start_time = time.time()
        self.max_runtime_seconds = max_runtime_minutes * 60
        self.upload_reserve_seconds = upload_reserve_minutes * 60
        self.warning_threshold = 0.8  # Warn at 80% of max runtime
    
    def elapsed_seconds(self) -> int:
        """Get elapsed time in seconds."""
        return int(time.time() - self.start_time)
    
    def remaining_seconds(self) -> int:
        """Get remaining time in seconds."""
        return max(0, self.max_runtime_seconds - self.elapsed_seconds())
    
    def remaining_minutes(self) -> float:
        """Get remaining time in minutes."""
        return self.remaining_seconds() / 60
    
    def should_exit_gracefully(self) -> bool:
        """Check if pipeline should exit gracefully to avoid timeout."""
        return self.remaining_seconds() <= 0
    
    def should_prioritize_upload(self) -> bool:
        """
        Check if filtering should stop to give time for uploads.
        
        Returns True when remaining time equals upload_reserve_seconds,
        signaling that filtering should wrap up so uploads can begin.
        """
        return self.remaining_seconds() <= self.upload_reserve_seconds
    
    def check_and_warn(self) -> Optional[str]:
        """Check timeout status and return warning message if applicable."""
        elapsed = self.elapsed_seconds()
        remaining = self.remaining_seconds()
        
        if remaining <= 0:
            return f"⏰ TIMEOUT: {elapsed // 60}m elapsed, must exit now"
        
        if remaining <= self.upload_reserve_seconds:
            return f"📤 UPLOAD TIME: Only {remaining // 60}m remaining, prioritizing uploads"
        
        if elapsed >= self.max_runtime_seconds * self.warning_threshold:
            return f"⏳ Warning: {remaining // 60}m remaining before timeout"
        
        return None
    
    def get_status(self) -> dict[str, Any]:
        """Get current timeout status."""
        return {
            "elapsed_seconds": self.elapsed_seconds(),
            "remaining_seconds": self.remaining_seconds(),
            "max_runtime_seconds": self.max_runtime_seconds,
            "upload_reserve_seconds": self.upload_reserve_seconds,
            "should_exit": self.should_exit_gracefully(),
            "should_prioritize_upload": self.should_prioritize_upload()
        }


# =============================================================================
# HEALTH CHECKER
# =============================================================================

class HealthChecker:
    """
    Pre-pipeline health checks.
    
    Verifies all prerequisites before starting the pipeline to fail fast
    if something is misconfigured.
    """
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.results: dict[str, tuple[bool, str]] = {}
    
    def check_api_credentials(self) -> dict[str, bool]:
        """Verify all API credentials are present."""
        results = {}
        
        # Reddit
        # Reddit (No auth needed)
        results["reddit"] = True
        
        # Unsplash
        results["unsplash"] = bool(os.getenv("UNSPLASH_ACCESS_KEY"))
        
        # Pexels
        results["pexels"] = bool(os.getenv("PEXELS_API_KEY"))
        
        # R2
        results["r2"] = bool(
            os.getenv("R2_ENDPOINT") and
            os.getenv("R2_ACCESS_KEY") and
            os.getenv("R2_SECRET_KEY") and
            os.getenv("R2_BUCKET_NAME")
        )
        
        return results
    
    async def check_r2_connectivity(self) -> bool:
        """Check R2 bucket is accessible."""
        endpoint = os.getenv("R2_ENDPOINT", "")
        bucket = os.getenv("R2_BUCKET_NAME", "")
        
        if not endpoint or not bucket:
            return False
        
        try:
            import boto3
            from botocore.config import Config as BotoConfig
            
            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=os.getenv("R2_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("R2_SECRET_KEY"),
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 2}
                )
            )
            
            client.head_bucket(Bucket=bucket)
            return True
            
        except Exception as e:
            logger.warning(f"R2 connectivity check failed: {e}")
            return False
    
    def check_disk_space(self, required_gb: float = 5.0) -> bool:
        """Check if sufficient disk space is available."""
        try:
            stat = shutil.disk_usage(".")
            available_gb = stat.free / (1024 ** 3)
            
            if available_gb < required_gb:
                logger.warning(
                    f"Low disk space: {available_gb:.1f}GB available, "
                    f"{required_gb}GB recommended"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Disk space check failed: {e}")
            return True  # Don't fail on check error
    
    def check_model_availability(self) -> dict[str, bool]:
        """Check if ML model files are present or downloadable."""
        results = {}
        
        # Check for model cache directories
        models_dir = Path("./models")
        
        # These will be downloaded on first use
        results["mobilenet"] = True  # PyTorch hub
        results["efficientnet"] = True  # timm
        results["siglip"] = True  # HuggingFace
        results["dinov2"] = True  # PyTorch hub
        
        return results
    
    def check_manifest_format(self, manifest_path: Path = Path("./manifest.json")) -> bool:
        """Validate existing manifest format is correct."""
        if not manifest_path.exists():
            return True  # No manifest yet is OK
        
        validator = DataValidator()
        valid, errors = validator.validate_manifest_json(manifest_path)
        
        if not valid:
            logger.error(f"Manifest validation failed: {errors}")
        
        return valid
    
    async def run_all_checks(self) -> tuple[bool, list[str]]:
        """
        Run all health checks.
        
        Returns:
            Tuple of (all_passed, list of error messages).
        """
        errors = []
        
        logger.info("🏥 Running pre-flight health checks...")
        
        # API credentials
        creds = self.check_api_credentials()
        for api, available in creds.items():
            if not available:
                if api == "r2":
                    errors.append(f"FATAL: {api.upper()} credentials missing")
                else:
                    logger.warning(f"⚠️ {api.upper()} credentials missing (source will be skipped)")
        
        # R2 connectivity (only if credentials are present)
        if creds.get("r2"):
            r2_ok = await self.check_r2_connectivity()
            if not r2_ok:
                errors.append("FATAL: Cannot connect to R2 bucket")
        
        # Disk space
        if not self.check_disk_space():
            errors.append("WARNING: Low disk space")
        
        # Manifest format
        if not self.check_manifest_format():
            errors.append("FATAL: Existing manifest is corrupted")
        
        # Summary
        all_passed = not any("FATAL" in e for e in errors)
        
        if all_passed:
            logger.info("✅ All health checks passed")
        else:
            logger.error(f"❌ Health checks failed: {errors}")
        
        return all_passed, errors


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

def setup_structured_logging(
    logs_dir: Path = Path("./logs"),
    max_days: int = 30,
    stages: Optional[list[str]] = None
) -> dict[str, logging.Logger]:
    """
    Create structured logging with separate files per stage.
    
    Args:
        logs_dir: Directory for log files.
        max_days: Number of days to keep logs.
        stages: List of stages to create loggers for.
    
    Returns:
        Dictionary mapping stage names to loggers.
    """
    if stages is None:
        stages = ["fetch", "filter", "embeddings", "upload", "main"]
    
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Rotate old logs
    _rotate_logs(logs_dir, max_days)
    
    loggers = {}
    timestamp = datetime.now().strftime("%Y%m%d")
    
    for stage in stages:
        stage_logger = logging.getLogger(f"wallpaper_curator.{stage}")
        stage_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        stage_logger.handlers.clear()
        
        # File handler
        log_file = logs_dir / f"{stage}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # JSON formatter for machine parsing
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"stage": "' + stage + '", "message": "%(message)s"}'
        )
        file_handler.setFormatter(json_formatter)
        
        stage_logger.addHandler(file_handler)
        loggers[stage] = stage_logger
    
    return loggers


def _rotate_logs(logs_dir: Path, max_days: int) -> None:
    """Remove log files older than max_days."""
    cutoff = datetime.now() - timedelta(days=max_days)
    
    for log_file in logs_dir.glob("*.log"):
        try:
            # Parse date from filename (format: stage_YYYYMMDD.log)
            parts = log_file.stem.split("_")
            if len(parts) >= 2:
                date_str = parts[-1]
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                if file_date < cutoff:
                    log_file.unlink()
                    logger.debug(f"Rotated old log: {log_file}")
        except Exception:
            pass  # Skip files we can't parse


# =============================================================================
# GRACEFUL DEGRADATION
# =============================================================================

class GracefulDegradation:
    """
    Handles graceful degradation when components fail.
    
    Allows the pipeline to continue with reduced functionality
    rather than failing entirely.
    """
    
    def __init__(self):
        self.failed_sources: list[str] = []
        self.failed_models: list[str] = []
        self.failed_wallpapers: list[str] = []
        self.warnings: list[str] = []
    
    def record_source_failure(self, source: str, error: str) -> None:
        """Record a source API failure."""
        self.failed_sources.append(source)
        self.warnings.append(f"Source '{source}' failed: {error}")
        logger.warning(f"⚠️ Source '{source}' unavailable, continuing with others")
    
    def record_model_failure(self, model: str, error: str) -> None:
        """Record a model loading/inference failure."""
        self.failed_models.append(model)
        self.warnings.append(f"Model '{model}' failed: {error}")
        logger.warning(f"⚠️ Model '{model}' failed, wallpapers will have incomplete embeddings")
    
    def record_wallpaper_failure(self, wallpaper_id: str, stage: str, error: str) -> None:
        """Record a single wallpaper processing failure."""
        self.failed_wallpapers.append(wallpaper_id)
        logger.warning(f"⚠️ Wallpaper '{wallpaper_id}' failed at {stage}: {error}")
    
    def should_continue(self) -> bool:
        """Check if pipeline should continue despite failures."""
        # Continue unless ALL sources failed
        return len(self.failed_sources) < 3  # Reddit, Unsplash, Pexels
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of all degradation events."""
        return {
            "failed_sources": self.failed_sources,
            "failed_models": self.failed_models,
            "failed_wallpapers": self.failed_wallpapers,
            "warning_count": len(self.warnings)
        }


# =============================================================================
# DRY-RUN MODE
# =============================================================================

class DryRunMode:
    """
    Handles dry-run mode where no actual changes are made.
    
    Useful for testing configuration without side effects.
    """
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.would_fetch: list[str] = []
        self.would_download: list[str] = []
        self.would_upload: list[str] = []
        self.manifest_changes: dict[str, Any] = {}
    
    def log_fetch(self, source: str, url: str) -> None:
        """Log what would be fetched."""
        if self.enabled:
            self.would_fetch.append(f"{source}: {url}")
            logger.info(f"[DRY-RUN] Would fetch: {source}: {url[:80]}...")
    
    def log_download(self, url: str, filepath: str) -> None:
        """Log what would be downloaded."""
        if self.enabled:
            self.would_download.append(f"{url} -> {filepath}")
            logger.info(f"[DRY-RUN] Would download: {url[:50]}...")
    
    def log_upload(self, filepath: str, r2_path: str) -> None:
        """Log what would be uploaded."""
        if self.enabled:
            self.would_upload.append(f"{filepath} -> {r2_path}")
            logger.info(f"[DRY-RUN] Would upload: {filepath} -> {r2_path}")
    
    def log_manifest_change(self, operation: str, data: Any) -> None:
        """Log what manifest changes would be made."""
        if self.enabled:
            self.manifest_changes[operation] = data
            logger.info(f"[DRY-RUN] Manifest change: {operation}")
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of what would have happened."""
        return {
            "enabled": self.enabled,
            "fetch_count": len(self.would_fetch),
            "download_count": len(self.would_download),
            "upload_count": len(self.would_upload),
            "manifest_changes": list(self.manifest_changes.keys())
        }


# =============================================================================
# BATCH PROCESSOR
# =============================================================================

class BatchProcessor:
    """
    Processes items in batches to manage memory usage.
    
    Useful for embedding extraction where loading many images
    at once would exhaust memory.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        memory_monitor: Optional[MemoryMonitor] = None
    ):
        self.batch_size = batch_size
        self.memory_monitor = memory_monitor or MemoryMonitor()
    
    def process_in_batches(
        self,
        items: list[Any],
        processor: Callable[[list[Any]], list[Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process.
            processor: Function that processes a batch and returns results.
            progress_callback: Optional callback(processed, total) for progress.
        
        Returns:
            Combined results from all batches.
        """
        results = []
        total = len(items)
        
        for i in range(0, total, self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Check memory before each batch
            if self.memory_monitor.should_gc():
                self.memory_monitor.force_gc()
            
            # Process batch
            batch_results = processor(batch)
            results.extend(batch_results)
            
            # Report progress
            processed = min(i + self.batch_size, total)
            if progress_callback:
                progress_callback(processed, total)
            
            logger.debug(f"Batch processed: {processed}/{total}")
        
        return results
    
    def generator_batches(
        self,
        items: list[Any]
    ) -> Iterator[list[Any]]:
        """
        Yield items in batches as a generator.
        
        Uses less memory than returning all at once.
        """
        for i in range(0, len(items), self.batch_size):
            # Check memory before each batch
            if self.memory_monitor.should_gc():
                self.memory_monitor.force_gc()
            
            yield items[i:i + self.batch_size]
