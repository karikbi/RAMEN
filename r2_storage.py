#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Part 3: R2 Storage

Handles batch uploading of approved wallpapers to Cloudflare R2.
Uses boto3 S3-compatible client.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from tqdm import tqdm

logger = logging.getLogger("wallpaper_curator")


@dataclass
class R2Config:
    """Configuration for Cloudflare R2 storage."""
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    custom_domain: Optional[str] = None  # e.g., "wallpapers.yourdomain.com"
    max_workers: int = 5  # Concurrent uploads
    
    @classmethod
    def from_env(cls) -> "R2Config":
        """Load configuration from environment variables."""
        return cls(
            endpoint=os.getenv("R2_ENDPOINT", ""),
            access_key=os.getenv("R2_ACCESS_KEY", ""),
            secret_key=os.getenv("R2_SECRET_KEY", ""),
            bucket_name=os.getenv("R2_BUCKET_NAME", ""),
            custom_domain=os.getenv("R2_CUSTOM_DOMAIN"),
        )
    
    def is_valid(self) -> bool:
        """Check if all required fields are set."""
        return all([self.endpoint, self.access_key, self.secret_key, self.bucket_name])


@dataclass
class UploadResult:
    """Result of a single upload operation."""
    wallpaper_id: str
    success: bool
    r2_path: str
    r2_url: str
    error: Optional[str] = None


class R2Uploader:
    """Handles batch uploading to Cloudflare R2."""
    
    CONTENT_TYPES = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    
    def __init__(self, config: R2Config):
        self.config = config
        self.client = self._create_client()
    
    def _create_client(self):
        """Create boto3 S3 client configured for R2."""
        if not self.config.is_valid():
            logger.warning("R2 configuration incomplete, uploads will be skipped")
            return None
        
        try:
            return boto3.client(
                "s3",
                endpoint_url=self.config.endpoint,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                config=BotoConfig(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "adaptive"}
                )
            )
        except Exception as e:
            logger.error(f"Failed to create R2 client: {e}")
            return None
    
    def _generate_r2_path(
        self,
        wallpaper_id: str,
        category: str,
        filepath: Path,
        date: Optional[datetime] = None
    ) -> str:
        """
        Generate R2 object path.
        
        Format: /{year}/{month:02d}/{category}/wp_{wallpaper_id}.{ext}
        Example: /2025/12/nature/wp_nature_1735108800_4521.jpg
        """
        if date is None:
            date = datetime.now()
        
        ext = filepath.suffix.lower()
        year = date.year
        month = date.month
        
        return f"{year}/{month:02d}/{category}/wp_{wallpaper_id}{ext}"
    
    def _generate_url(self, r2_path: str) -> str:
        """Generate public URL for the uploaded object."""
        if self.config.custom_domain:
            return f"https://{self.config.custom_domain}/{r2_path}"
        else:
            # Standard R2 public URL format
            return f"https://{self.config.bucket_name}.r2.cloudflarestorage.com/{r2_path}"
    
    def _get_content_type(self, filepath: Path) -> str:
        """Get content type for file."""
        ext = filepath.suffix.lower()
        return self.CONTENT_TYPES.get(ext, "application/octet-stream")
    
    def upload_single(
        self,
        filepath: Path,
        wallpaper_id: str,
        category: str,
        date: Optional[datetime] = None
    ) -> UploadResult:
        """
        Upload a single file to R2.
        
        Args:
            filepath: Local path to the file.
            wallpaper_id: Unique wallpaper ID.
            category: Wallpaper category for path organization.
            date: Date for path organization (defaults to now).
        
        Returns:
            UploadResult with success status and URL.
        """
        r2_path = self._generate_r2_path(wallpaper_id, category, filepath, date)
        r2_url = self._generate_url(r2_path)
        
        if self.client is None:
            return UploadResult(
                wallpaper_id=wallpaper_id,
                success=False,
                r2_path=r2_path,
                r2_url=r2_url,
                error="R2 client not initialized"
            )
        
        try:
            content_type = self._get_content_type(filepath)
            
            with open(filepath, "rb") as f:
                self.client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=r2_path,
                    Body=f,
                    ContentType=content_type,
                    CacheControl="public, max-age=31536000",  # 1 year cache
                )
            
            logger.debug(f"Uploaded: {r2_path}")
            
            return UploadResult(
                wallpaper_id=wallpaper_id,
                success=True,
                r2_path=r2_path,
                r2_url=r2_url
            )
        
        except ClientError as e:
            error_msg = str(e)
            logger.error(f"R2 upload failed for {wallpaper_id}: {error_msg}")
            return UploadResult(
                wallpaper_id=wallpaper_id,
                success=False,
                r2_path=r2_path,
                r2_url=r2_url,
                error=error_msg
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected upload error for {wallpaper_id}: {error_msg}")
            return UploadResult(
                wallpaper_id=wallpaper_id,
                success=False,
                r2_path=r2_path,
                r2_url=r2_url,
                error=error_msg
            )
    
    def upload_batch(
        self,
        uploads: list[tuple[Path, str, str]],  # (filepath, wallpaper_id, category)
        show_progress: bool = True
    ) -> list[UploadResult]:
        """
        Upload multiple files in parallel.
        
        Args:
            uploads: List of (filepath, wallpaper_id, category) tuples.
            show_progress: Whether to show progress bar.
        
        Returns:
            List of UploadResult objects.
        """
        if not uploads:
            return []
        
        if self.client is None:
            logger.warning("Skipping R2 uploads: client not initialized")
            return [
                UploadResult(
                    wallpaper_id=wp_id,
                    success=False,
                    r2_path="",
                    r2_url="",
                    error="R2 client not initialized"
                )
                for _, wp_id, _ in uploads
            ]
        
        results = []
        
        logger.info(f"📤 Uploading {len(uploads)} wallpapers to R2...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self.upload_single, fp, wp_id, cat): (fp, wp_id, cat)
                for fp, wp_id, cat in uploads
            }
            
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Uploading"
            ) if show_progress else as_completed(futures)
            
            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    fp, wp_id, cat = futures[future]
                    logger.error(f"Upload future failed for {wp_id}: {e}")
                    results.append(UploadResult(
                        wallpaper_id=wp_id,
                        success=False,
                        r2_path="",
                        r2_url="",
                        error=str(e)
                    ))
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        logger.info(f"✅ Uploaded {successful} wallpapers to R2")
        if failed:
            logger.warning(f"⚠️  {failed} uploads failed")
        
        return results
    
    def verify_upload(self, r2_path: str) -> bool:
        """Verify that an object exists in R2."""
        if self.client is None:
            return False
        
        try:
            self.client.head_object(
                Bucket=self.config.bucket_name,
                Key=r2_path
            )
            return True
        except ClientError:
            return False
    
    def check_exists(self, r2_path: str) -> bool:
        """
        Check if file already exists in R2 (idempotency check).
        
        Uses HEAD request which is very cheap.
        """
        return self.verify_upload(r2_path)
    
    def upload_with_recovery(
        self,
        filepath: Path,
        wallpaper_id: str,
        category: str,
        upload_queue=None,
        date: Optional[datetime] = None
    ) -> UploadResult:
        """
        Upload with idempotency check and queue management for recovery.
        
        Features:
        - Check if file already exists in R2 (skip if exists)
        - Add to queue before upload (for crash recovery)
        - Remove from queue after successful upload
        - Queue persists for retry on next run
        
        Args:
            filepath: Local path to the file.
            wallpaper_id: Unique wallpaper ID.
            category: Wallpaper category for path organization.
            upload_queue: Optional UploadQueueManager for recovery.
            date: Date for path organization (defaults to now).
        
        Returns:
            UploadResult with success status and URL.
        """
        r2_path = self._generate_r2_path(wallpaper_id, category, filepath, date)
        r2_url = self._generate_url(r2_path)
        
        # Idempotency check: skip if already exists
        if self.check_exists(r2_path):
            logger.info(f"⏭️  File already exists in R2, skipping: {wallpaper_id}")
            return UploadResult(
                wallpaper_id=wallpaper_id,
                success=True,
                r2_path=r2_path,
                r2_url=r2_url
            )
        
        # Add to queue before upload (for crash recovery)
        if upload_queue:
            from pipeline_robustness import UploadQueueItem
            upload_queue.add_to_queue([UploadQueueItem(
                wallpaper_id=wallpaper_id,
                filepath=str(filepath),
                category=category,
                r2_path=r2_path,
                added_at=datetime.now().isoformat()
            )])
        
        # Perform upload
        result = self.upload_single(filepath, wallpaper_id, category, date)
        
        # Remove from queue on success
        if result.success and upload_queue:
            upload_queue.remove_from_queue(wallpaper_id)
        elif not result.success and upload_queue:
            upload_queue.mark_failed(wallpaper_id, result.error or "Unknown error")
        
        return result
    
    def upload_batch_with_recovery(
        self,
        uploads: list[tuple[Path, str, str]],  # (filepath, wallpaper_id, category)
        upload_queue=None,
        show_progress: bool = True
    ) -> list[UploadResult]:
        """
        Upload multiple files with idempotency and queue management.
        
        Args:
            uploads: List of (filepath, wallpaper_id, category) tuples.
            upload_queue: Optional UploadQueueManager for recovery.
            show_progress: Whether to show progress bar.
        
        Returns:
            List of UploadResult objects.
        """
        if not uploads:
            return []
        
        if self.client is None:
            logger.warning("Skipping R2 uploads: client not initialized")
            return [
                UploadResult(
                    wallpaper_id=wp_id,
                    success=False,
                    r2_path="",
                    r2_url="",
                    error="R2 client not initialized"
                )
                for _, wp_id, _ in uploads
            ]
        
        results = []
        skipped = 0
        
        logger.info(f"📤 Uploading {len(uploads)} wallpapers to R2 (with idempotency check)...")
        
        # Use sequential uploads for better recovery tracking
        iterator = tqdm(uploads, desc="Uploading") if show_progress else uploads
        
        for filepath, wp_id, category in iterator:
            result = self.upload_with_recovery(
                filepath, wp_id, category, upload_queue
            )
            results.append(result)
            
            if result.success and "already exists" in str(result.error or ""):
                skipped += 1
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        logger.info(f"✅ Uploaded {successful} wallpapers to R2 ({skipped} skipped as duplicates)")
        if failed:
            logger.warning(f"⚠️  {failed} uploads failed (queued for retry)")
        
        return results
    
    def resume_pending_uploads(
        self,
        upload_queue,
        show_progress: bool = True
    ) -> list[UploadResult]:
        """
        Resume any pending uploads from the queue.
        
        Call this at the start of a run to handle any uploads
        that failed in a previous run.
        
        Args:
            upload_queue: UploadQueueManager with pending uploads.
            show_progress: Whether to show progress bar.
        
        Returns:
            List of UploadResult objects for resumed uploads.
        """
        pending = upload_queue.get_pending()
        
        if not pending:
            return []
        
        logger.info(f"📥 Resuming {len(pending)} pending uploads from previous run...")
        
        uploads = [
            (Path(item.filepath), item.wallpaper_id, item.category)
            for item in pending
            if Path(item.filepath).exists()
        ]
        
        if len(uploads) < len(pending):
            missing = len(pending) - len(uploads)
            logger.warning(f"⚠️  {missing} queued files no longer exist, skipping")
        
        return self.upload_batch_with_recovery(uploads, upload_queue, show_progress)


def upload_to_r2(
    approved_wallpapers: list,  # List of ApprovedWallpaper
    config: Optional[R2Config] = None
) -> dict[str, str]:
    """
    Upload approved wallpapers to R2 and return URL mapping.
    
    Args:
        approved_wallpapers: List of ApprovedWallpaper objects from Part 2.
        config: R2 configuration (loads from env if None).
    
    Returns:
        Dictionary mapping wallpaper_id to R2 URL.
    """
    if config is None:
        config = R2Config.from_env()
    
    if not config.is_valid():
        logger.warning("R2 configuration incomplete, skipping uploads")
        return {}
    
    uploader = R2Uploader(config)
    
    # Prepare upload list
    uploads = []
    for wp in approved_wallpapers:
        if wp.filepath and wp.filepath.exists():
            category = wp.metadata.primary_category or "general"
            uploads.append((wp.filepath, wp.id, category))
    
    # Execute batch upload
    results = uploader.upload_batch(uploads)
    
    # Build URL mapping
    url_mapping = {}
    for result in results:
        if result.success:
            url_mapping[result.wallpaper_id] = result.r2_url
    
    return url_mapping


# =============================================================================
# BATCH CONFIGURATION AND STATS
# =============================================================================

@dataclass
class BatchConfig:
    """Configuration for batched R2 operations."""
    batch_size: int = 100              # Files per batch
    inter_batch_delay: float = 1.0     # Seconds between batches
    max_concurrent: int = 5            # Concurrent uploads within batch
    retry_failed: bool = True          # Retry failed uploads at end
    max_retries: int = 3               # Max retry attempts per file


@dataclass
class BatchStats:
    """Statistics for batch operations."""
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped_duplicates: int = 0
    total_bytes: int = 0
    batches_processed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "successful": self.successful,
            "failed": self.failed,
            "skipped_duplicates": self.skipped_duplicates,
            "total_bytes": self.total_bytes,
            "batches_processed": self.batches_processed,
            "duration_sec": (
                (self.end_time - self.start_time).total_seconds()
                if self.start_time and self.end_time else 0
            )
        }


# =============================================================================
# BATCHED R2 UPLOADER
# =============================================================================

class R2BatchUploader:
    """
    Batched uploader for Cloudflare R2 to avoid free tier limits.
    
    Cloudflare R2 free tier limits:
    - 10GB storage
    - 1 million Class A (write) operations/month
    - 10 million Class B (read) operations/month
    
    This uploader:
    - Batches uploads to reduce API call overhead
    - Adds rate limiting between batches
    - Tracks statistics for monitoring
    - Retries failed uploads
    """
    
    def __init__(self, uploader: R2Uploader, config: Optional[BatchConfig] = None):
        self.uploader = uploader
        self.config = config or BatchConfig()
        self.stats = BatchStats()
    
    def upload_batch(
        self,
        uploads: list[tuple[Path, str, str]],  # (filepath, wallpaper_id, category)
        show_progress: bool = True
    ) -> list[UploadResult]:
        """
        Upload files in batches with rate limiting.
        
        Args:
            uploads: List of (filepath, wallpaper_id, category) tuples.
            show_progress: Whether to show progress bar.
        
        Returns:
            List of UploadResult objects.
        """
        import time
        
        if not uploads:
            return []
        
        self.stats = BatchStats(
            total_files=len(uploads),
            start_time=datetime.now()
        )
        
        results = []
        failed_uploads = []
        total_batches = (len(uploads) + self.config.batch_size - 1) // self.config.batch_size
        
        logger.info(f"📦 Starting batched upload: {len(uploads)} files in {total_batches} batches")
        
        for batch_num, batch in enumerate(self._chunk(uploads)):
            batch_num_display = batch_num + 1
            logger.info(f"📤 Processing batch {batch_num_display}/{total_batches} ({len(batch)} files)")
            
            # Process batch
            batch_results = self._process_batch(batch, show_progress)
            
            for result in batch_results:
                results.append(result)
                if result.success:
                    self.stats.successful += 1
                    if result.r2_path:
                        try:
                            # Track bytes uploaded
                            filepath = next(
                                fp for fp, wp_id, _ in batch 
                                if wp_id == result.wallpaper_id
                            )
                            self.stats.total_bytes += filepath.stat().st_size
                        except (StopIteration, OSError):
                            pass
                else:
                    self.stats.failed += 1
                    # Track for retry
                    failed_item = next(
                        (item for item in batch if item[1] == result.wallpaper_id),
                        None
                    )
                    if failed_item:
                        failed_uploads.append(failed_item)
            
            self.stats.batches_processed += 1
            
            # Rate limit between batches (not after last batch)
            if batch_num_display < total_batches:
                logger.debug(f"⏳ Waiting {self.config.inter_batch_delay}s before next batch...")
                time.sleep(self.config.inter_batch_delay)
        
        # Retry failed uploads
        if self.config.retry_failed and failed_uploads:
            logger.info(f"🔄 Retrying {len(failed_uploads)} failed uploads...")
            retry_results = self._retry_failed(failed_uploads)
            
            for result in retry_results:
                # Update the existing result
                for i, existing in enumerate(results):
                    if existing.wallpaper_id == result.wallpaper_id:
                        if result.success:
                            self.stats.failed -= 1
                            self.stats.successful += 1
                        results[i] = result
                        break
        
        self.stats.end_time = datetime.now()
        
        # Log summary
        duration = (self.stats.end_time - self.stats.start_time).total_seconds()
        logger.info(
            f"✅ Batch upload complete: {self.stats.successful}/{self.stats.total_files} successful "
            f"({self.stats.total_bytes / (1024*1024):.1f} MB) in {duration:.1f}s"
        )
        
        if self.stats.failed > 0:
            logger.warning(f"⚠️ {self.stats.failed} uploads failed")
        
        return results
    
    def _chunk(self, items: list) -> list:
        """Split list into chunks of batch_size."""
        for i in range(0, len(items), self.config.batch_size):
            yield items[i:i + self.config.batch_size]
    
    def _process_batch(
        self,
        batch: list[tuple[Path, str, str]],
        show_progress: bool
    ) -> list[UploadResult]:
        """Process a single batch of uploads."""
        results = []
        
        iterator = tqdm(batch, desc="  Uploading", leave=False) if show_progress else batch
        
        for filepath, wp_id, category in iterator:
            result = self.uploader.upload_single(filepath, wp_id, category)
            results.append(result)
        
        return results
    
    def _retry_failed(self, failed_uploads: list[tuple[Path, str, str]]) -> list[UploadResult]:
        """Retry failed uploads with backoff."""
        import time
        
        results = []
        
        for attempt in range(self.config.max_retries):
            if not failed_uploads:
                break
            
            delay = 2 ** attempt  # Exponential backoff
            logger.info(f"  Retry attempt {attempt + 1}/{self.config.max_retries} for {len(failed_uploads)} files...")
            time.sleep(delay)
            
            still_failed = []
            for filepath, wp_id, category in failed_uploads:
                result = self.uploader.upload_single(filepath, wp_id, category)
                results.append(result)
                
                if not result.success:
                    still_failed.append((filepath, wp_id, category))
            
            failed_uploads = still_failed
        
        return results
    
    def get_stats(self) -> dict:
        """Get batch statistics."""
        return self.stats.to_dict()


# =============================================================================
# R2 MANIFEST MANAGER
# =============================================================================

class R2ManifestManager:
    """
    Manages manifest storage on Cloudflare R2.
    
    Stores manifests on R2 instead of GitHub to:
    - Keep manifests private (not exposed in public repo)
    - Reduce GitHub repository size
    - Enable faster access via R2 CDN
    
    Maintains local cache for fast access.
    """
    
    def __init__(
        self,
        r2_client,
        bucket: str,
        manifest_prefix: str = "manifests/",
        custom_domain: Optional[str] = None,
        local_cache_dir: Path = Path("./manifest_cache")
    ):
        self.r2_client = r2_client
        self.bucket = bucket
        self.prefix = manifest_prefix
        self.custom_domain = custom_domain
        self.local_cache_dir = local_cache_dir
        
        # Ensure cache directory exists
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_r2_key(self, filename: str) -> str:
        """Get full R2 key for a manifest file."""
        return f"{self.prefix}{filename}"
    
    def _get_public_url(self, key: str) -> str:
        """Get public URL for an R2 object."""
        if self.custom_domain:
            return f"https://{self.custom_domain}/{key}"
        return f"https://{self.bucket}.r2.cloudflarestorage.com/{key}"
    
    def upload_manifest(
        self,
        manifest_path: Path,
        key: str = "collection.json.gz"
    ) -> str:
        """
        Upload manifest to R2.
        
        Args:
            manifest_path: Local path to manifest file.
            key: R2 object key (filename).
        
        Returns:
            Public URL of uploaded manifest.
        """
        import gzip
        
        if self.r2_client is None:
            raise RuntimeError("R2 client not configured")
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        r2_key = self._get_r2_key(key)
        
        # Read and compress if not already compressed
        if manifest_path.suffix == ".gz":
            with open(manifest_path, "rb") as f:
                data = f.read()
            content_type = "application/gzip"
        else:
            with open(manifest_path, "rb") as f:
                raw_data = f.read()
            data = gzip.compress(raw_data)
            content_type = "application/gzip"
            # Update key to include .gz
            if not key.endswith(".gz"):
                r2_key = f"{r2_key}.gz"
        
        try:
            self.r2_client.put_object(
                Bucket=self.bucket,
                Key=r2_key,
                Body=data,
                ContentType=content_type,
                # Note: Removed ContentEncoding to avoid boto3 checksum mismatch
                CacheControl="public, max-age=3600"  # 1 hour cache
            )
            
            url = self._get_public_url(r2_key)
            logger.info(f"📤 Manifest uploaded to R2: {url}")
            
            # Save to local cache
            cache_path = self.local_cache_dir / key
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if key.endswith(".gz"):
                with open(cache_path, "wb") as f:
                    f.write(data)
            else:
                cache_path = cache_path.with_suffix(cache_path.suffix + ".gz")
                with open(cache_path, "wb") as f:
                    f.write(data)
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to upload manifest to R2: {e}")
            raise
    
    def download_manifest(
        self,
        key: str = "collection.json.gz"
    ) -> Optional[Path]:
        """
        Download manifest from R2 to local cache.
        
        Args:
            key: R2 object key (filename).
        
        Returns:
            Local path to downloaded manifest, or None if not found.
        """
        import gzip
        
        if self.r2_client is None:
            logger.warning("R2 client not configured, using local cache only")
            return self._get_local_cache(key)
        
        r2_key = self._get_r2_key(key)
        cache_path = self.local_cache_dir / key
        
        try:
            logger.info(f"📥 Downloading manifest from R2: {r2_key}")
            
            response = self.r2_client.get_object(
                Bucket=self.bucket,
                Key=r2_key
            )
            
            data = response["Body"].read()
            
            # Save to local cache
            with open(cache_path, "wb") as f:
                f.write(data)
            
            logger.info(f"✅ Manifest downloaded to: {cache_path}")
            return cache_path
            
        except self.r2_client.exceptions.NoSuchKey:
            logger.info("No manifest found in R2, will create new one")
            return None
        except Exception as e:
            logger.warning(f"Failed to download manifest from R2: {e}")
            return self._get_local_cache(key)
    
    def _get_local_cache(self, key: str) -> Optional[Path]:
        """Get manifest from local cache if available."""
        cache_path = self.local_cache_dir / key
        
        if cache_path.exists():
            logger.info(f"Using cached manifest: {cache_path}")
            return cache_path
        
        return None
    
    def upload_delta(self, delta_path: Path) -> str:
        """
        Upload delta file to R2.
        
        Args:
            delta_path: Local path to delta file.
        
        Returns:
            Public URL of uploaded delta.
        """
        # Use the delta filename as the key
        key = f"deltas/{delta_path.name}"
        return self.upload_manifest(delta_path, key)
    
    def list_manifests(self) -> list[str]:
        """
        List all manifests in R2.
        
        Returns:
            List of manifest keys.
        """
        if self.r2_client is None:
            return []
        
        try:
            response = self.r2_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.prefix
            )
            
            manifests = []
            for obj in response.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".json") or key.endswith(".json.gz"):
                    manifests.append(key)
            
            return manifests
            
        except Exception as e:
            logger.warning(f"Failed to list manifests: {e}")
            return []
    
    def delete_old_deltas(self, keep_count: int = 30) -> int:
        """
        Delete old delta files, keeping the most recent ones.
        
        Args:
            keep_count: Number of recent deltas to keep.
        
        Returns:
            Number of deleted files.
        """
        if self.r2_client is None:
            return 0
        
        try:
            response = self.r2_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f"{self.prefix}deltas/"
            )
            
            deltas = []
            for obj in response.get("Contents", []):
                deltas.append({
                    "Key": obj["Key"],
                    "LastModified": obj["LastModified"]
                })
            
            # Sort by date, oldest first
            deltas.sort(key=lambda x: x["LastModified"])
            
            # Delete old ones
            to_delete = deltas[:-keep_count] if len(deltas) > keep_count else []
            
            for delta in to_delete:
                self.r2_client.delete_object(
                    Bucket=self.bucket,
                    Key=delta["Key"]
                )
            
            if to_delete:
                logger.info(f"🗑️ Deleted {len(to_delete)} old delta files")
            
            return len(to_delete)
            
        except Exception as e:
            logger.warning(f"Failed to delete old deltas: {e}")
            return 0


def create_r2_client(config: Optional[R2Config] = None):
    """
    Create a boto3 S3 client configured for Cloudflare R2.
    
    Args:
        config: R2 configuration (loads from env if None).
    
    Returns:
        boto3 S3 client or None if config invalid.
    """
    if config is None:
        config = R2Config.from_env()
    
    if not config.is_valid():
        logger.warning("R2 configuration incomplete")
        return None
    
    try:
        return boto3.client(
            "s3",
            endpoint_url=config.endpoint,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            config=BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"}
            )
        )
    except Exception as e:
        logger.error(f"Failed to create R2 client: {e}")
        return None
