#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Part 2: Filtering Pipeline

Main pipeline that processes candidates through hard filters, quality scoring,
embedding extraction, and metadata generation.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Import Part 1 types
from curate_wallpapers import CandidateWallpaper, Config

# Import Part 2 modules
from filters import HardFilters, FilterConfig, FilterResult
from ml_quality_scorer import MLQualityScorer, MLQualityConfig, MLQualityScore
from embeddings import EmbeddingExtractor, EmbeddingSet
from metadata_generator import MetadataGenerator, WallpaperMetadata


logger = logging.getLogger("wallpaper_curator")


@dataclass
class ApprovedWallpaper:
    """Represents a fully processed and approved wallpaper."""
    id: str
    source: str
    filepath: Path
    url: str
    title: str
    artist: str
    
    # Part 2 additions
    quality_scores: MLQualityScore = field(default_factory=MLQualityScore)
    embeddings: EmbeddingSet = field(default_factory=EmbeddingSet)
    metadata: WallpaperMetadata = field(default_factory=WallpaperMetadata)
    phash: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "source": self.source,
            "filepath": str(self.filepath),
            "url": self.url,
            "title": self.title,
            "artist": self.artist,
            "quality_score": self.quality_scores.final_score,
            "phash": self.phash,
            "metadata": self.metadata.to_dict(),
            "embeddings": self.embeddings.to_dict(),
        }


@dataclass
class FilteringStats:
    """Statistics from the filtering pipeline."""
    total_candidates: int = 0
    passed_hard_filters: int = 0
    passed_quality_scoring: int = 0
    final_approved: int = 0
    
    # Rejection reasons
    rejected_resolution: int = 0
    rejected_file_integrity: int = 0
    rejected_aspect_ratio: int = 0
    rejected_text_detection: int = 0
    rejected_file_size: int = 0
    rejected_duplicate: int = 0
    rejected_quality_score: int = 0
    
    def log_summary(self) -> None:
        """Log filtering summary."""
        logger.info("\n" + "=" * 60)
        logger.info("FILTERING COMPLETE - STATISTICS")
        logger.info("=" * 60)
        logger.info(f"  Total Candidates:     {self.total_candidates}")
        logger.info(f"  Passed Hard Filters:  {self.passed_hard_filters}")
        logger.info(f"  Passed Quality Score: {self.passed_quality_scoring}")
        logger.info(f"  Final Approved:       {self.final_approved}")
        logger.info("-" * 60)
        logger.info("Rejection Breakdown:")
        logger.info(f"  - Resolution:      {self.rejected_resolution}")
        logger.info(f"  - File Integrity:  {self.rejected_file_integrity}")
        logger.info(f"  - Aspect Ratio:    {self.rejected_aspect_ratio}")
        logger.info(f"  - Text Detection:  {self.rejected_text_detection}")
        logger.info(f"  - File Size:       {self.rejected_file_size}")
        logger.info(f"  - Duplicate:       {self.rejected_duplicate}")
        logger.info(f"  - Quality Score:   {self.rejected_quality_score}")
        logger.info("=" * 60)


class FilteringPipeline:
    """
    Main filtering pipeline that processes candidates through:
    1. Hard filters (auto-reject)
    2. ML-based quality scoring (using SigLIP)
    3. Embedding extraction (for approved images only)
    4. Metadata generation
    """
    
    def __init__(
        self,
        config: Config,
        filter_config: Optional[FilterConfig] = None,
        quality_config: Optional[MLQualityConfig] = None,
    ):
        self.config = config
        self.filter_config = filter_config or FilterConfig()
        self.quality_config = quality_config or MLQualityConfig(
            threshold=config.quality_threshold
        )
        
        # Initialize components
        self.hard_filters = HardFilters(self.filter_config)
        self.embedding_extractor = EmbeddingExtractor()
        self.ml_quality_scorer = MLQualityScorer(
            embedding_extractor=self.embedding_extractor,
            config=self.quality_config
        )
        self.metadata_generator = MetadataGenerator()
        
        # Statistics
        self.stats = FilteringStats()
        
        # Track new hashes for saving
        self.new_hashes: dict[str, str] = {}

    
    def _categorize_rejection(self, reason: str) -> None:
        """Categorize rejection reason for statistics."""
        reason_lower = reason.lower()
        if "resolution" in reason_lower:
            self.stats.rejected_resolution += 1
        elif "corrupt" in reason_lower or "format" in reason_lower:
            self.stats.rejected_file_integrity += 1
        elif "aspect" in reason_lower:
            self.stats.rejected_aspect_ratio += 1
        elif "text" in reason_lower:
            self.stats.rejected_text_detection += 1
        elif "size" in reason_lower or "small" in reason_lower or "large" in reason_lower:
            self.stats.rejected_file_size += 1
        elif "duplicate" in reason_lower:
            self.stats.rejected_duplicate += 1
    
    def process_candidate(self, candidate: CandidateWallpaper) -> Optional[ApprovedWallpaper]:
        """
        Process a single candidate through the full pipeline.
        
        Args:
            candidate: CandidateWallpaper from Part 1.
        
        Returns:
            ApprovedWallpaper if passes all checks, None otherwise.
        """
        if candidate.filepath is None or not candidate.filepath.exists():
            logger.warning(f"Candidate {candidate.id} has no valid filepath")
            return None
        
        # Step 1: Hard Filters (includes watermark cropping for Reddit)
        filter_result = self.hard_filters.apply_all_filters(
            candidate.filepath,
            candidate.id,
            source=candidate.source  # Pass source for watermark detection
        )
        
        if not filter_result.passed:
            self._categorize_rejection(filter_result.reason or "unknown")
            self.hard_filters.reject_candidate(
                candidate.filepath,
                filter_result.reason or "unknown"
            )
            logger.debug(f"Rejected {candidate.id}: {filter_result.reason}")
            return None
        
        self.stats.passed_hard_filters += 1
        
        # Step 2: ML Quality Scoring with SOURCE-AWARE weights
        # Reddit: balanced (aesthetic 40%, technical 20%, wallpaper 40%)
        # Unsplash/Pexels: focus on wallpaper suitability (80%)
        ml_score, siglip_embedding = self.ml_quality_scorer.score_for_source(
            candidate.filepath,
            source=candidate.source
        )

        
        # Step 3: Quality Check
        if ml_score.final_score < self.quality_config.threshold:
            self.stats.rejected_quality_score += 1
            self.hard_filters.reject_candidate(
                candidate.filepath,
                f"Quality score {ml_score.final_score:.3f} < {self.quality_config.threshold} "
                f"(aes={ml_score.aesthetic_score:.2f}, tech={ml_score.technical_score:.2f}, wall={ml_score.wallpaper_score:.2f})"
            )
            return None
        
        self.stats.passed_quality_scoring += 1
        
        # Step 4: Extract Remaining Embeddings (ONLY FOR APPROVED IMAGES)
        # SigLIP already extracted, now get MobileNet, EfficientNet, DINOv2
        logger.info(f"Approved {candidate.id} (score={ml_score.final_score:.3f}) - extracting embeddings")
        embeddings = EmbeddingSet()
        embeddings.siglip = siglip_embedding
        embeddings.mobilenet_v3 = self.embedding_extractor.extract_mobilenet(candidate.filepath)
        embeddings.efficientnet_v2 = self.embedding_extractor.extract_efficientnet(candidate.filepath)
        embeddings.dinov2 = self.embedding_extractor.extract_dinov2(candidate.filepath)

        
        # Step 5: Generate Metadata
        metadata = self.metadata_generator.generate_metadata(
            filepath=candidate.filepath,
            title=candidate.title,
            artist=candidate.artist,
            source=candidate.source,
            source_metadata=candidate.metadata,
            quality_score=ml_score.final_score
        )
        
        # Step 6: Move to approved directory
        approved_path = self.config.approved_dir / candidate.filepath.name
        try:
            shutil.copy2(str(candidate.filepath), str(approved_path))
            candidate.filepath.unlink()  # Remove from candidates
        except Exception as e:
            logger.error(f"Failed to move approved file: {e}")
            return None
        
        # Store hash for later
        if filter_result.phash:
            self.new_hashes[candidate.id] = filter_result.phash
        
        self.stats.final_approved += 1
        
        return ApprovedWallpaper(
            id=metadata.id or candidate.id,
            source=candidate.source,
            filepath=approved_path,
            url=candidate.url,
            title=candidate.title,
            artist=candidate.artist,
            quality_scores=ml_score,
            embeddings=embeddings,
            metadata=metadata,
            phash=filter_result.phash
        )

    

    def process_all(
        self,
        candidates: list[CandidateWallpaper],
        show_progress: bool = True
    ) -> list[ApprovedWallpaper]:
        """
        Process all candidates through the filtering pipeline.
        
        Args:
            candidates: List of CandidateWallpaper from Part 1.
            show_progress: Whether to show tqdm progress bar.
        
        Returns:
            List of ApprovedWallpaper objects.
        """
        self.stats.total_candidates = len(candidates)
        approved = []
        
        logger.info(f"\n🔍 Processing {len(candidates)} candidates through filters...")
        
        iterator = tqdm(candidates, desc="Filtering") if show_progress else candidates
        
        for candidate in iterator:
            try:
                result = self.process_candidate(candidate)
                if result:
                    approved.append(result)
            except Exception as e:
                logger.error(f"Error processing {candidate.id}: {e}")
                continue
        
        # Save new hashes
        if self.new_hashes:
            self.hard_filters.save_hashes(self.new_hashes)
            logger.info(f"Saved {len(self.new_hashes)} new hashes")
        
        # Log statistics
        self.stats.log_summary()
        
        return approved
    
    def save_approved_manifest(
        self,
        approved: list[ApprovedWallpaper],
        output_path: Path = Path("./approved_manifest.json")
    ) -> None:
        """Save approved wallpapers to a JSON manifest."""
        manifest = {
            "version": "1.0",
            "count": len(approved),
            "wallpapers": [wp.to_dict() for wp in approved]
        }
        
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Saved manifest to {output_path}")


def run_part2_pipeline(
    candidates: list[CandidateWallpaper],
    config: Optional[Config] = None,
    quality_threshold: float = 0.55  # Lowered for softmax-based ML scoring
) -> list[ApprovedWallpaper]:
    """
    Run the Part 2 filtering pipeline.
    
    Args:
        candidates: List of candidates from Part 1.
        config: Pipeline configuration (uses default if None).
        quality_threshold: Minimum quality score to accept.
    
    Returns:
        List of approved wallpapers with embeddings and metadata.
    """
    logger.info("=" * 60)
    logger.info("RAMEN Wallpaper Curation Pipeline - Part 2: Filtering")
    logger.info("=" * 60)
    
    if config is None:
        config = Config()
    
    # Override quality threshold if specified
    config.quality_threshold = quality_threshold
    
    # ML Quality config
    quality_config = MLQualityConfig(threshold=quality_threshold)
    
    # Run pipeline
    pipeline = FilteringPipeline(
        config=config,
        quality_config=quality_config
    )
    
    approved = pipeline.process_all(candidates)
    
    # Save manifest
    if approved:
        pipeline.save_approved_manifest(approved)
    
    logger.info(f"\n✅ Part 2 complete. {len(approved)} wallpapers approved.")
    
    return approved


# For standalone testing
if __name__ == "__main__":
    import asyncio
    from curate_wallpapers import main as fetch_candidates
    
    # Run Part 1 to get candidates
    print("Running Part 1: Fetching candidates...")
    candidates = asyncio.run(fetch_candidates())
    
    # Run Part 2 to filter and process
    print("\nRunning Part 2: Filtering and processing...")
    approved = run_part2_pipeline(candidates)
    
    print(f"\n🎉 Pipeline complete!")
    print(f"   Fetched: {len(candidates)} candidates")
    print(f"   Approved: {len(approved)} wallpapers")
