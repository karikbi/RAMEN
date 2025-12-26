#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Part 3: Complete Pipeline

Main orchestrator that runs the complete curation pipeline:
Part 1: Fetching → Part 2: Filtering → Part 3: Storage & Reporting

Enhanced with robustness and recovery capabilities:
- State management and crash recovery
- Lock file to prevent concurrent runs
- Graceful degradation on partial failures
- Timeout awareness for CI environments
- Health checks before starting
- Dry-run mode for testing
"""

import asyncio
import logging
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Part 1 imports
from curate_wallpapers import Config, CandidateWallpaper, fetch_candidates

# Part 2 imports
from pipeline_part2 import (
    FilteringPipeline,
    ApprovedWallpaper,
    FilteringStats,
    FilterConfig,
)
from filters import FilterConfig
from ml_quality_scorer import MLQualityConfig


# Part 3 imports
from r2_storage import (
    R2Config, upload_to_r2, R2Uploader,
    R2BatchUploader, BatchConfig, R2ManifestManager, create_r2_client
)
from manifest_manager import ManifestManager, HashManager, SourceStatsManager
from reporting import PipelineStats, ReportGenerator, generate_report

# Deduplication imports
from dedup_manager import (
    DuplicateIndex, DuplicateChecker, DedupSync,
    DedupConfig, create_dedup_system
)

# Central config
from config_loader import get_config

# Robustness imports
from pipeline_robustness import (
    StateManager,
    PipelineState,
    TimeoutManager,
    MemoryMonitor,
    HealthChecker,
    DataValidator,
    AtomicFileWriter,
    GracefulDegradation,
    DryRunMode,
    EmbeddingProgressTracker,
    UploadQueueManager,
    ErrorCategory,
    PipelineError,
    setup_structured_logging,
)

logger = logging.getLogger("wallpaper_curator")


class WallpaperCurationPipeline:
    """
    Complete wallpaper curation pipeline orchestrator.
    
    Runs all three parts:
    1. Fetching: Download candidates from Reddit, Unsplash, Pexels
    2. Filtering: Hard filters, quality scoring, embeddings, metadata
    3. Storage: R2 upload, manifest update, reporting
    
    Enhanced with robustness features:
    - Crash recovery via state management
    - Lock file to prevent concurrent runs
    - Graceful degradation on partial failures
    - Timeout awareness for CI environments
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        quality_threshold: Optional[float] = None,  # Read from config.yaml if not specified
        skip_upload: bool = False,
        skip_part1: bool = False,
        dry_run: bool = False,
        max_runtime_minutes: int = 50,
        fresh_start: bool = False,
        test_mode: bool = False,
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration.
            quality_threshold: Minimum quality score (default: from config.yaml).
            skip_upload: Skip R2 upload (for testing).
            skip_part1: Skip fetching, use existing candidates.
            dry_run: Dry-run mode - no side effects.
            max_runtime_minutes: Max runtime before graceful exit (for CI).
            fresh_start: Reset deduplication system for clean curation.
        """
        self.config = config or Config()
        
        # Get quality threshold from central config if not specified
        if quality_threshold is None:
            quality_threshold = get_config().get('quality.threshold', 0.40)
        self.quality_threshold = quality_threshold
        self.skip_upload = skip_upload
        self.skip_part1 = skip_part1
        self.dry_run_enabled = dry_run
        self.fresh_start = fresh_start
        self.test_mode = test_mode
        
        # If fresh start, clear local dedup/state files before initialization
        if fresh_start:
            self._clear_dedup_state()
        
        # Statistics
        self.stats = PipelineStats()
        
        # Managers
        self.manifest_manager = ManifestManager()
        self.hash_manager = HashManager()
        self.source_stats = SourceStatsManager()
        
        # Robustness components
        self.state_manager = StateManager()
        # Reserve 5 minutes at the end for uploads/storage (batch uploads are fast)
        self.timeout_manager = TimeoutManager(
            max_runtime_minutes=max_runtime_minutes,
            upload_reserve_minutes=5  # Reduced from 10 - uploads are fast with batching
        )
        self.memory_monitor = MemoryMonitor()
        self.health_checker = HealthChecker()
        self.data_validator = DataValidator()
        self.degradation = GracefulDegradation()
        self.dry_run = DryRunMode(enabled=dry_run)
        self.embedding_tracker = EmbeddingProgressTracker()
        self.upload_queue = UploadQueueManager()
        
        # Deduplication system
        self.dedup_index: Optional[DuplicateIndex] = None
        self.dedup_checker: Optional[DuplicateChecker] = None
        self.dedup_sync: Optional[R2DedupSync] = None
        
        # R2 manifest storage
        self.r2_manifest_manager: Optional[R2ManifestManager] = None
        
        # Batch uploader
        self.batch_uploader: Optional[R2BatchUploader] = None
        
        # Structured logging
        self.stage_loggers = setup_structured_logging()
    
    def _clear_dedup_state(self) -> None:
        """
        Clear all local deduplication state for fresh curation.
        
        Removes:
        - Dedup cache directory
        - Existing hashes file
        - Pipeline state directory
        - Manifest cache
        """
        import shutil
        
        paths_to_clear = [
            Path("./dedup_cache"),
            Path("./existing_hashes.json"),
            Path("./pipeline_state"),
            Path("./manifest_cache"),
            Path("./source_stats.json"),
            Path("manifests/dedup_index.json.gz"),
        ]
        
        logger.info("🧹 Clearing deduplication state for fresh start...")
        
        for path in paths_to_clear:
            try:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                        logger.info(f"  ✓ Removed directory: {path}")
                    else:
                        path.unlink()
                        logger.info(f"  ✓ Removed file: {path}")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to remove {path}: {e}")
        
        logger.info("🔄 Fresh start ready - deduplication system reset")
    
    async def run_part1_fetching(self) -> list[CandidateWallpaper]:
        """Run Part 1: Fetch candidates from all sources."""
        logger.info("\n" + "=" * 70)
        logger.info("PART 1: FETCHING CANDIDATES")
        logger.info("=" * 70)
        
        start = time.time()
        candidates = await fetch_candidates(
            config=self.config,
            test_mode=self.test_mode,
            dedup_checker=self.dedup_checker
        )
        self.stats.fetch_duration_sec = time.time() - start
        
        # Count by source
        for c in candidates:
            if c.source == "reddit":
                self.stats.reddit_candidates += 1
            elif c.source == "unsplash":
                self.stats.unsplash_candidates += 1
            elif c.source == "pexels":
                self.stats.pexels_candidates += 1
        
        self.stats.total_candidates = len(candidates)
        
        return candidates
    
    def run_part2_filtering(
        self,
        candidates: list[CandidateWallpaper]
    ) -> list[ApprovedWallpaper]:
        """Run Part 2: Filter, score, extract embeddings."""
        logger.info("\n" + "=" * 70)
        logger.info("PART 2: FILTERING AND PROCESSING")
        logger.info("=" * 70)
        
        start = time.time()
        
        # Configure filtering with ML quality scoring
        quality_config = MLQualityConfig(threshold=self.quality_threshold)
        
        # Create timeout callback for early exit when upload time approaches
        # Timeout callback no longer needed - pipeline uses proper 3-pass approach
        # that completes all scoring before embedding extraction
        
        # Run filtering pipeline
        pipeline = FilteringPipeline(
            config=self.config,
            quality_config=quality_config,
            dedup_checker=self.dedup_checker
        )
        
        approved = pipeline.process_all(candidates)
        
        # Copy stats
        filter_stats = pipeline.stats
        self.stats.passed_hard_filters = filter_stats.passed_hard_filters
        self.stats.passed_quality_scoring = filter_stats.passed_quality_scoring
        self.stats.rejected_resolution = filter_stats.rejected_resolution
        self.stats.rejected_file_integrity = filter_stats.rejected_file_integrity
        self.stats.rejected_aspect_ratio = filter_stats.rejected_aspect_ratio
        self.stats.rejected_text_detection = filter_stats.rejected_text_detection
        self.stats.rejected_file_size = filter_stats.rejected_file_size
        self.stats.rejected_duplicate = filter_stats.rejected_duplicate
        self.stats.rejected_quality_score = filter_stats.rejected_quality_score
        self.stats.approved_count = len(approved)
        
        # Collect quality scores
        self.stats.quality_scores = [wp.quality_scores.final_score for wp in approved]
        
        # Category distribution
        for wp in approved:
            cat = wp.metadata.primary_category or "unknown"
            self.stats.category_counts[cat] = self.stats.category_counts.get(cat, 0) + 1
        
        self.stats.filter_duration_sec = time.time() - start
        
        # Log if we exited early
        if pipeline.early_exit:
            logger.info("📤 Filtering ended early to prioritize uploads")
        
        return approved
    
    def run_part3_storage(
        self,
        approved: list[ApprovedWallpaper]
    ) -> dict[str, str]:
        """Run Part 3: Upload to R2 and update manifests."""
        logger.info("\n" + "=" * 70)
        logger.info("PART 3: STORAGE AND MANIFEST UPDATE")
        logger.info("=" * 70)
        
        start = time.time()
        r2_urls = {}
        
        # Upload to R2
        if not self.skip_upload:
            r2_config = R2Config.from_env()
            if r2_config.is_valid():
                # Use batch uploader if available
                if self.batch_uploader:
                    logger.info("📦 Using batched upload for R2...")
                    
                    # Prepare upload list
                    uploads = []
                    for wp in approved:
                        if wp.filepath and wp.filepath.exists():
                            category = wp.metadata.primary_category or "general"
                            uploads.append((wp.filepath, wp.id, category))
                    
                    # Execute batched upload
                    results = self.batch_uploader.upload_batch(uploads)
                    
                    # Build URL mapping
                    for result in results:
                        if result.success:
                            r2_urls[result.wallpaper_id] = result.r2_url
                    
                    self.stats.uploaded_count = len(r2_urls)
                    self.stats.upload_failures = len(approved) - len(r2_urls)
                else:
                    # Fallback to original upload
                    r2_urls = upload_to_r2(approved, r2_config)
                    self.stats.uploaded_count = len(r2_urls)
                    self.stats.upload_failures = len(approved) - len(r2_urls)
            else:
                logger.warning("R2 not configured, skipping upload")
        else:
            logger.info("Skipping R2 upload (--skip-upload)")
        
        self.stats.upload_duration_sec = time.time() - start
        
        # Update manifest (will also sync to R2 if configured)
        logger.info("\n📝 Updating manifest...")
        new_count, delta_path = self.manifest_manager.update_manifest(approved, r2_urls)
        
        # Update perceptual hashes
        logger.info("🔐 Updating perceptual hashes...")
        self.hash_manager.update_from_approved(approved)
        
        # Register approved wallpapers in dedup index
        if self.dedup_checker and self.dedup_index:
            logger.info("🔍 Updating dedup index...")
            for wp in approved:
                self.dedup_checker.register(
                    wp_id=wp.id,
                    url=wp.metadata.source_url if hasattr(wp.metadata, 'source_url') else "",
                    filepath=wp.filepath
                )
        
        # Save dedup index to Repo and R2
        if self.dedup_sync and self.dedup_index:
            logger.info("📤 Syncing dedup index (Repo + R2)...")
            self.dedup_sync.sync_index(self.dedup_index)
        
        # Update source statistics
        logger.info("📈 Updating source statistics...")
        self._update_source_stats(approved)
        
        return r2_urls
    
    def _update_source_stats(self, approved: list[ApprovedWallpaper]) -> None:
        """Update source statistics for adaptive balancing."""
        # Group by source and subreddit
        source_counts = {}
        
        for wp in approved:
            if wp.source == "reddit":
                subreddit = wp.metadata.subreddit
                key = f"reddit_{subreddit}"
            else:
                key = wp.source
            
            source_counts[key] = source_counts.get(key, 0) + 1
        
        # Track what we approved
        for key, count in source_counts.items():
            if key.startswith("reddit_"):
                subreddit = key.replace("reddit_", "")
                self.source_stats.update_stats("reddit", count, count, subreddit)
            else:
                self.source_stats.update_stats(key, count, count)
        
        # Store for report
        for key, count in source_counts.items():
            self.stats.source_results[key] = {
                "candidates": count,  # Approximation
                "approved": count
            }
    
    def generate_final_report(self) -> Path:
        """Generate final pipeline report."""
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING REPORT")
        logger.info("=" * 70)
        
        self.stats.end_time = datetime.now()
        return generate_report(self.stats, print_summary=True)
    
    async def run(self) -> list[ApprovedWallpaper]:
        """
        Run the complete pipeline with robustness features.
        
        Includes:
        - Lock file to prevent concurrent runs
        - Health checks before starting
        - State tracking for crash recovery
        - Timeout awareness for CI environments
        - Graceful degradation on partial failures
        
        Returns:
            List of approved wallpapers.
        """
        # Dry-run mode handling
        if self.dry_run_enabled:
            logger.info("🧪 DRY-RUN MODE - No files will be modified")
            return await self._run_dry_mode()
        
        # Acquire lock to prevent concurrent runs
        if not self.state_manager.acquire_lock():
            logger.error("❌ Another pipeline instance is running. Exiting.")
            sys.exit(1)
        
        try:
            return await self._run_with_recovery()
        finally:
            self.state_manager.release_lock()
    
    async def _run_with_recovery(self) -> list[ApprovedWallpaper]:
        """Internal run with recovery support."""
        self.stats.start_time = datetime.now()
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("\n" + "🚀" * 35)
        logger.info("RAMEN WALLPAPER CURATION PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Started at: {self.stats.start_time.isoformat()}")
        logger.info(f"Quality threshold: {self.quality_threshold}")
        logger.info(f"Run ID: {run_id}")
        logger.info("🚀" * 35)
        
        # Run health checks
        logger.info("\n🏥 Running pre-flight health checks...")
        healthy, issues = await self.health_checker.run_all_checks()
        
        if not healthy:
            for issue in issues:
                if "FATAL" in issue:
                    logger.error(f"Health check failed: {issue}")
                    return []
        
        # Initialize R2 client
        r2_config = R2Config.from_env()
        r2_client = create_r2_client(r2_config) if r2_config.is_valid() else None
        
        # Initialize deduplication system
        logger.info("\n🔍 Initializing deduplication system...")
        
        # We always want to use the persistent repo storage
        repo_dedup_path = Path("manifests/dedup_index.json.gz")
        dedup_config = DedupConfig(
            sync_to_r2=True,
            repo_path=repo_dedup_path
        )
        
        # Initialize with or without R2
        self.dedup_index, self.dedup_checker, self.dedup_sync = create_dedup_system(
            r2_client=r2_client if r2_config.bucket_name else None,
            bucket=r2_config.bucket_name if r2_config.bucket_name else "",
            config=dedup_config
        )
        
        if self.dedup_index:
             logger.info(f"Loaded dedup index: {self.dedup_index.get_stats()}")
        
        # Initialize R2 manifest manager
        if r2_client and r2_config.bucket_name:
            self.r2_manifest_manager = R2ManifestManager(
                r2_client=r2_client,
                bucket=r2_config.bucket_name,
                manifest_prefix="manifests/",
                custom_domain=r2_config.custom_domain
            )
            self.manifest_manager.set_r2_manager(self.r2_manifest_manager)
            self.manifest_manager.set_dedup_index(self.dedup_index)
            
            # Download existing manifest from R2
            logger.info("\n📥 Loading manifest from R2...")
            self.r2_manifest_manager.download_manifest()
        
        # Initialize batch uploader
        if r2_client:
            uploader = R2Uploader(r2_config)
            self.batch_uploader = R2BatchUploader(
                uploader=uploader,
                config=BatchConfig(
                    batch_size=100,
                    inter_batch_delay=1.0,
                    retry_failed=True
                )
            )
        
        # Check for resumable state
        existing_state = self.state_manager.load_state()
        resume_stage = None
        
        if existing_state and self.state_manager.is_recent():
            resume_stage = existing_state.stage_completed
            logger.info(f"📋 Resuming from stage: {resume_stage}")
        else:
            # Initialize fresh state
            state = PipelineState.initial(run_id)
            self.state_manager.save_state(state)
        
        # Part 1: Fetching
        candidates = []
        if resume_stage not in ["fetch", "filter", "embed", "upload", "complete"]:
            if self.skip_part1:
                logger.info("Skipping Part 1 (using existing candidates)")
                candidates = self._load_existing_candidates()
            else:
                # Check timeout before starting
                timeout_msg = self.timeout_manager.check_and_warn()
                if timeout_msg and self.timeout_manager.should_exit_gracefully():
                    logger.warning(timeout_msg)
                    return await self._graceful_exit("timeout before fetch")
                
                try:
                    candidates = await self.run_part1_fetching()
                except Exception as e:
                    self.degradation.record_source_failure("all", str(e))
                    logger.error(f"Fetching failed: {e}")
                    return []
            
            # Save state after fetch
            state = PipelineState(
                stage_completed="fetch",
                candidates_processed=len(candidates),
                approved_count=0,
                failed_candidates=[],
                timestamp=datetime.now().isoformat(),
                run_id=run_id
            )
            self.state_manager.save_state(state)
        else:
            # Load existing candidates if resuming
            candidates = self._load_existing_candidates()
        
        if not candidates:
            logger.warning("No candidates to process!")
            return []
        
        # Part 2: Filtering
        approved = []
        if resume_stage not in ["filter", "embed", "upload", "complete"]:
            # Check timeout
            timeout_msg = self.timeout_manager.check_and_warn()
            if timeout_msg:
                logger.warning(timeout_msg)
            
            if self.timeout_manager.should_exit_gracefully():
                return await self._graceful_exit("timeout before filter")
            
            try:
                approved = self.run_part2_filtering(candidates)
            except Exception as e:
                logger.error(f"Filtering failed: {e}")
                state = self.state_manager.load_state()
                if state:
                    state.stage_completed = "fetch"  # Mark fetch as last good stage
                    self.state_manager.save_state(state)
                return []
            
            # Save state after filter
            state = PipelineState(
                stage_completed="filter",
                candidates_processed=len(candidates),
                approved_count=len(approved),
                failed_candidates=list(self.degradation.failed_wallpapers),
                timestamp=datetime.now().isoformat(),
                run_id=run_id
            )
            self.state_manager.save_state(state)
        
        if not approved:
            logger.warning("No wallpapers passed filtering!")
            self.generate_final_report()
            return []
        
        # Part 3: Storage
        if resume_stage not in ["upload", "complete"]:
            # Check timeout
            if self.timeout_manager.should_exit_gracefully():
                return await self._graceful_exit("timeout before upload")
            
            try:
                self.run_part3_storage(approved)
            except Exception as e:
                logger.error(f"Storage failed: {e}")
                # State preserved for retry
                return approved  # Return what we have
            
            # Save state after storage
            state = PipelineState(
                stage_completed="upload",
                candidates_processed=len(candidates),
                approved_count=len(approved),
                failed_candidates=list(self.degradation.failed_wallpapers),
                timestamp=datetime.now().isoformat(),
                run_id=run_id
            )
            self.state_manager.save_state(state)
        
        # Validate before finalizing
        logger.info("\n🔍 Validating data integrity...")
        await self._validate_data()
        
        # Final report
        self.generate_final_report()
        
        # Clean state on success
        self.state_manager.clean_state()
        
        logger.info("\n" + "🎉" * 35)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total candidates: {self.stats.total_candidates}")
        logger.info(f"Approved: {self.stats.approved_count}")
        logger.info(f"Uploaded: {self.stats.uploaded_count}")
        
        # Show degradation summary if any failures
        if self.degradation.failed_sources or self.degradation.failed_wallpapers:
            summary = self.degradation.get_summary()
            logger.warning(f"⚠️ Degradation: {summary}")
        
        logger.info("🎉" * 35)
        
        return approved
    
    async def _run_dry_mode(self) -> list[ApprovedWallpaper]:
        """Run in dry-run mode with no side effects."""
        logger.info("\n" + "🧪" * 35)
        logger.info("DRY-RUN MODE SUMMARY")
        logger.info("=" * 70)
        
        # Simulate fetch
        logger.info("\n📱 [DRY-RUN] Would fetch candidates from:")
        logger.info("  - Reddit: r/wallpapers, r/EarthPorn, r/Amoledbackgrounds...")
        logger.info("  - Unsplash: Curated collections")
        logger.info("  - Pexels: Curated endpoint")
        
        # Load existing candidates if available
        if self.skip_part1:
            candidates = self._load_existing_candidates()
            if candidates:
                logger.info(f"\n🖼️ [DRY-RUN] Found {len(candidates)} existing candidates")
                logger.info("  Would process through filtering pipeline")
                logger.info("  Would extract embeddings from 4 models")
                
                for c in candidates[:5]:
                    self.dry_run.log_fetch(c.source, c.url or "(local file)")
        
        # Show what would be uploaded
        logger.info("\n☁️ [DRY-RUN] Upload simulation:")
        logger.info("  - Would upload approved wallpapers to R2")
        logger.info("  - Would update manifest.json")
        logger.info("  - Would update phash database")
        
        summary = self.dry_run.get_summary()
        logger.info(f"\n📊 Dry-run summary: {summary}")
        
        logger.info("\n🧪" * 35)
        
        return []
    
    async def _graceful_exit(self, reason: str) -> list[ApprovedWallpaper]:
        """Exit gracefully, saving state for resume."""
        logger.warning(f"⏰ Graceful exit triggered: {reason}")
        logger.info("Pipeline state saved - will resume on next run")
        
        # Generate partial report
        self.generate_final_report()
        
        return []
    
    async def _validate_data(self) -> bool:
        """Validate data integrity before finalizing."""
        all_valid = True
        
        # Check manifest
        manifest_path = Path("./manifest.json")
        if manifest_path.exists():
            valid, errors = self.data_validator.validate_manifest_json(manifest_path)
            if not valid:
                logger.error(f"Manifest validation failed: {errors}")
                all_valid = False
        
        # Check for duplicate IDs
        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Handle both list format and wrapped format {"wallpapers": [...]}
            if isinstance(manifest, list):
                manifest = {"wallpapers": manifest}
            elif isinstance(manifest, dict) and "wallpapers" not in manifest:
                manifest = {"wallpapers": []}
            valid, errors = self.data_validator.check_duplicate_ids(manifest)
            if not valid:
                logger.error(f"Duplicate ID check failed: {errors}")
                all_valid = False
        
        return all_valid
    
    def _load_existing_candidates(self) -> list[CandidateWallpaper]:
        """Load candidates from temp/candidates/ directory."""
        candidates = []
        candidates_dir = self.config.candidates_dir
        
        if not candidates_dir.exists():
            return candidates
        
        for filepath in candidates_dir.iterdir():
            if filepath.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                # Parse filename to extract info
                name = filepath.stem
                parts = name.split("_")
                
                if len(parts) >= 2:
                    source = parts[0]
                    wp_id = "_".join(parts[1:])
                else:
                    source = "unknown"
                    wp_id = name
                
                candidates.append(CandidateWallpaper(
                    id=wp_id,
                    source=source,
                    filepath=filepath,
                    url="",
                    title=name,
                    artist="Unknown"
                ))
        
        logger.info(f"Loaded {len(candidates)} existing candidates")
        return candidates


async def main(
    quality_threshold: Optional[float] = None,  # Read from config.yaml if not specified
    skip_upload: bool = False,
    skip_part1: bool = False,
    dry_run: bool = False,
    max_runtime_minutes: int = 50,
    fresh_start: bool = False,
    test_mode: bool = False,
) -> list[ApprovedWallpaper]:
    """
    Run the complete wallpaper curation pipeline.
    
    Args:
        quality_threshold: Minimum quality score (default: from config.yaml).
        skip_upload: Skip R2 upload for testing.
        skip_part1: Skip fetching, use existing candidates.
        dry_run: Dry-run mode with no side effects.
        max_runtime_minutes: Max runtime before graceful exit.
        fresh_start: Reset deduplication system for clean curation.
    
    Returns:
        List of approved wallpapers.
    """
    pipeline = WallpaperCurationPipeline(
        quality_threshold=quality_threshold,
        skip_upload=skip_upload,
        skip_part1=skip_part1,
        dry_run=dry_run,
        max_runtime_minutes=max_runtime_minutes,
        fresh_start=fresh_start,
        test_mode=test_mode,
    )
    
    return await pipeline.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAMEN Wallpaper Curation Pipeline - Production-grade with robustness features"
    )
    parser.add_argument(
        "--quality-threshold", "-q",
        type=float,
        default=None,  # Will read from config.yaml
        help="Minimum quality score to approve (default: from config.yaml)"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip R2 upload (for testing)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching, use existing candidates in temp/candidates/"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - simulate pipeline without side effects"
    )
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=50,
        help="Maximum runtime in minutes before graceful exit (default: 50)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force run even if another instance appears to be running"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Fresh start - reset deduplication system and start clean (ignores previous runs)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode - fetch only 10 wallpapers (5 Reddit, 2 Unsplash, 3 Pexels) for quick testing"
    )

    
    args = parser.parse_args()
    
    # Configure debug logging if requested
    if args.debug:
        logging.getLogger("wallpaper_curator").setLevel(logging.DEBUG)
        # Also set root logger to see all debug output
        logging.basicConfig(level=logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Handle --force flag by removing stale lock file
    if args.force:
        lock_file = Path("./pipeline_state/.pipeline.lock")
        if lock_file.exists():
            lock_file.unlink()
            logger.info("🔓 Removed stale lock file (--force)")
    
    # Log fresh start mode
    if args.fresh:
        logger.info("🔄 Fresh start mode enabled - resetting deduplication system")
    
    approved = asyncio.run(main(
        quality_threshold=args.quality_threshold,
        skip_upload=args.skip_upload,
        skip_part1=args.skip_fetch,
        dry_run=args.dry_run,
        max_runtime_minutes=args.max_runtime,
        fresh_start=args.fresh,
        test_mode=args.test_mode,
    ))
    
    if approved:
        print(f"\n✅ Complete! Approved {len(approved)} wallpapers.")
    else:
        print("\n⚠️ Pipeline completed with no approved wallpapers.")

