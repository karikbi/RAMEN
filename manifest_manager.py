#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Part 3: Manifest Management

Handles manifest generation, updates, delta files, and source statistics.

Enhanced with:
- R2 storage integration for private manifest storage
- Deduplication index integration
"""

import base64
import gzip
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from r2_storage import R2ManifestManager
    from dedup_manager import DuplicateIndex

logger = logging.getLogger("wallpaper_curator")


@dataclass
class ManifestConfig:
    """Configuration for manifest management."""
    manifests_dir: Path = Path("./manifests")
    collection_file: str = "collection.json"
    compressed: bool = True
    hashes_file: Path = Path("./existing_hashes.json")
    source_stats_file: Path = Path("./source_stats.json")


class ManifestManager:
    """
    Manages wallpaper collection manifests and delta files.
    
    Enhanced with:
    - R2 storage for manifests (optional)
    - Dedup index integration
    """
    
    def __init__(self, config: Optional[ManifestConfig] = None):
        self.config = config or ManifestConfig()
        self.config.manifests_dir.mkdir(parents=True, exist_ok=True)
        
        # R2 integration (optional)
        self.r2_manager: Optional["R2ManifestManager"] = None
        
        # Dedup index integration (optional)
        self.dedup_index: Optional["DuplicateIndex"] = None
    
    def _to_native(self, value: Any) -> Any:
        """
        Convert numpy types to native Python types for JSON serialization.
        
        Handles: np.float32, np.float64, np.int32, np.int64, np.bool_, etc.
        """
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    def _safe_focal_point(self, focal_point: tuple) -> list:
        """
        Convert focal point tuple to a JSON-serializable list of floats.
        
        Handles numpy float types that may be in the tuple.
        """
        return [float(self._to_native(x)) for x in focal_point]
    
    def set_r2_manager(self, r2_manager: "R2ManifestManager") -> None:
        """
        Enable R2 storage for manifests.
        
        When enabled, manifests are uploaded to R2 after saving locally.
        """
        self.r2_manager = r2_manager
        logger.info("R2 manifest storage enabled")
    
    def set_dedup_index(self, dedup_index: "DuplicateIndex") -> None:
        """
        Set dedup index for cross-run ID tracking.
        
        When enabled, approved wallpaper IDs are registered in the dedup index.
        """
        self.dedup_index = dedup_index
        logger.info("Dedup index integration enabled")
    
    def _encode_embedding(self, embedding: Optional[np.ndarray]) -> Optional[str]:
        """Encode numpy array to base64 string."""
        if embedding is None:
            return None
        return base64.b64encode(embedding.tobytes()).decode("utf-8")
    
    def _create_manifest_entry(
        self,
        wallpaper,  # ApprovedWallpaper
        r2_url: str = ""
    ) -> dict[str, Any]:
        """
        Create a manifest entry for a wallpaper.
        
        Args:
            wallpaper: ApprovedWallpaper object.
            r2_url: R2 storage URL if uploaded.
        
        Returns:
            Dictionary representing manifest entry.
        
        Note:
            All values are converted to native Python types via _to_native()
            to ensure JSON serialization works correctly (no numpy float32 errors).
        """
        meta = wallpaper.metadata
        embeddings = wallpaper.embeddings
        
        return {
            "id": wallpaper.id,
            "title": wallpaper.title,
            "url": r2_url or str(wallpaper.filepath),
            "r2_url": r2_url or "",  # Explicit R2 URL for validation
            "category": meta.primary_category,
            "primary_category": meta.primary_category,  # Explicit for validation
            "subcategories": meta.subcategories,
            
            # Color analysis - convert numpy types
            "colors": meta.color_palette,
            "dominant_hue": self._to_native(meta.dominant_hue),
            "color_diversity": round(float(self._to_native(meta.color_diversity)), 3),
            "brightness": self._to_native(meta.brightness),
            "contrast": round(float(self._to_native(meta.contrast_ratio)), 2),
            "is_dark_mode_friendly": self._to_native(meta.is_dark_mode_friendly),
            
            # Quality metrics - convert numpy types
            "quality_score": round(float(self._to_native(wallpaper.quality_scores.final_score)), 4),
            "quality_tier": meta.quality_tier,
            "aesthetic_score": round(float(self._to_native(meta.aesthetic_score)), 2),
            
            # ML classification metadata - convert numpy types
            "ml_category": meta.ml_category,
            "ml_confidence": round(float(self._to_native(meta.ml_confidence)), 3),
            
            "dimensions": {
                "width": self._to_native(meta.width),
                "height": self._to_native(meta.height)
            },
            "aspect_ratio": round(float(self._to_native(meta.aspect_ratio)), 2),
            "file_size": self._to_native(meta.file_size),
            "date_added": datetime.now().isoformat() + "Z",
            "source": wallpaper.source,
            "source_metadata": {
                "subreddit": meta.subreddit,
                "upvotes": self._to_native(meta.upvotes),
                "post_url": meta.post_url,
                "source_url": meta.source_url,
            },
            "artist": wallpaper.artist,
            "artist_url": meta.artist_url,
            "license": meta.license_type,
            "embeddings": {
                "mobilenet_v4": self._encode_embedding(embeddings.mobilenet_v4),
                "efficientnet_v2": self._encode_embedding(embeddings.efficientnet_v2),
                "siglip": self._encode_embedding(embeddings.siglip),
                "dinov3": self._encode_embedding(embeddings.dinov3),
            },
            "composition": {
                "type": meta.composition_type,
                "symmetry": round(float(self._to_native(meta.symmetry_score)), 2),
                "depth": round(float(self._to_native(meta.depth_score)), 2),
                "complexity": meta.complexity_level,
                "focal_point": self._safe_focal_point(meta.focal_point),
                "focal_point_method": meta.focal_point_method,
            },
            "mood_tags": meta.mood_tags,
            "style_tags": meta.style_tags,
            "metadata_version": "1.1"  # Version for new fields
        }
    
    def load_collection(self) -> list[dict]:
        """Load existing collection from manifest file."""
        collection_path = self.config.manifests_dir / self.config.collection_file
        
        if not collection_path.exists():
            # Check for compressed version
            compressed_path = collection_path.with_suffix(".json.gz")
            if compressed_path.exists():
                try:
                    with gzip.open(compressed_path, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                        return data.get("wallpapers", [])
                except Exception as e:
                    logger.error(f"Failed to load compressed collection: {e}")
            return []
        
        try:
            with open(collection_path, "r") as f:
                data = json.load(f)
                return data.get("wallpapers", [])
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            return []
    
    def save_collection(self, wallpapers: list[dict]) -> None:
        """Save collection to manifest file."""
        # Sort by date_added descending
        wallpapers.sort(key=lambda x: x.get("date_added", ""), reverse=True)
        
        manifest = {
            "version": "1.0",
            "updated": datetime.now().isoformat() + "Z",
            "count": len(wallpapers),
            "wallpapers": wallpapers
        }
        
        if self.config.compressed:
            output_path = self.config.manifests_dir / (self.config.collection_file + ".gz")
            with gzip.open(output_path, "wt", encoding="utf-8") as f:
                json.dump(manifest, f)
            logger.info(f"Saved compressed collection to {output_path}")
        else:
            output_path = self.config.manifests_dir / self.config.collection_file
            with open(output_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Saved collection to {output_path}")
        
        # Upload to R2 if configured
        if self.r2_manager:
            try:
                r2_url = self.r2_manager.upload_manifest(output_path)
                logger.info(f"📤 Manifest synced to R2: {r2_url}")
            except Exception as e:
                logger.error(f"Failed to upload manifest to R2: {e}")
                # Continue - local save succeeded
    
    def save_delta(self, new_entries: list[dict]) -> Path:
        """Save delta file with only new entries."""
        date_str = datetime.now().strftime("%Y_%m_%d")
        delta_filename = f"delta_{date_str}.json"
        delta_path = self.config.manifests_dir / delta_filename
        
        delta = {
            "version": "1.0",
            "created": datetime.now().isoformat() + "Z",
            "count": len(new_entries),
            "wallpapers": new_entries
        }
        
        # Compress delta files too
        if self.config.compressed:
            delta_path = delta_path.with_suffix(".json.gz")
            with gzip.open(delta_path, "wt", encoding="utf-8") as f:
                json.dump(delta, f)
        else:
            with open(delta_path, "w") as f:
                json.dump(delta, f, indent=2)
        
        logger.info(f"Saved delta file: {delta_path}")
        return delta_path
    
    def update_manifest(
        self,
        approved_wallpapers: list,  # List of ApprovedWallpaper
        r2_urls: dict[str, str] = None
    ) -> tuple[int, Path]:
        """
        Update manifest with new approved wallpapers.
        
        Args:
            approved_wallpapers: List of ApprovedWallpaper objects.
            r2_urls: Optional mapping of wallpaper_id to R2 URL.
        
        Returns:
            Tuple of (number of new entries, delta file path).
        """
        if r2_urls is None:
            r2_urls = {}
        
        # Load existing collection
        existing = self.load_collection()
        existing_ids = {w["id"] for w in existing}
        
        # Create new entries
        new_entries = []
        for wp in approved_wallpapers:
            if wp.id not in existing_ids:
                r2_url = r2_urls.get(wp.id, "")
                entry = self._create_manifest_entry(wp, r2_url)
                new_entries.append(entry)
        
        if not new_entries:
            logger.info("No new wallpapers to add to manifest")
            return 0, Path()
        
        # Combine and save
        all_wallpapers = new_entries + existing
        self.save_collection(all_wallpapers)
        
        # Save delta
        delta_path = self.save_delta(new_entries)
        
        # Upload delta to R2 if configured
        if self.r2_manager and delta_path.exists():
            try:
                self.r2_manager.upload_delta(delta_path)
            except Exception as e:
                logger.warning(f"Failed to upload delta to R2: {e}")
        
        # Update dedup index with new IDs
        if self.dedup_index:
            for wp in approved_wallpapers:
                self.dedup_index.add_id(wp.id)
        
        logger.info(f"Added {len(new_entries)} new wallpapers to manifest")
        
        return len(new_entries), delta_path


class HashManager:
    """Manages perceptual hash storage for deduplication."""
    
    def __init__(self, hashes_file: Path = Path("./existing_hashes.json")):
        self.hashes_file = hashes_file
        self.hashes: dict[str, str] = {}
        self._load()
    
    def _load(self) -> None:
        """Load existing hashes from file."""
        if self.hashes_file.exists():
            try:
                with open(self.hashes_file, "r") as f:
                    self.hashes = json.load(f)
                logger.debug(f"Loaded {len(self.hashes)} existing hashes")
            except Exception as e:
                logger.warning(f"Failed to load hashes: {e}")
    
    def save(self) -> None:
        """Save hashes to file."""
        try:
            with open(self.hashes_file, "w") as f:
                json.dump(self.hashes, f, indent=2)
            logger.debug(f"Saved {len(self.hashes)} hashes")
        except Exception as e:
            logger.error(f"Failed to save hashes: {e}")
    
    def add_hashes(self, new_hashes: dict[str, str]) -> None:
        """Add new hashes and save."""
        self.hashes.update(new_hashes)
        self.save()
        logger.info(f"Added {len(new_hashes)} new hashes (total: {len(self.hashes)})")
    
    def update_from_approved(self, approved_wallpapers: list) -> None:
        """Update hashes from approved wallpapers."""
        new_hashes = {}
        for wp in approved_wallpapers:
            if wp.phash:
                new_hashes[wp.id] = wp.phash
        
        if new_hashes:
            self.add_hashes(new_hashes)


class SourceStatsManager:
    """Manages source acceptance statistics for adaptive balancing."""
    
    def __init__(self, stats_file: Path = Path("./source_stats.json")):
        self.stats_file = stats_file
        self.stats: dict[str, dict] = {}
        self._load()
    
    def _load(self) -> None:
        """Load existing statistics."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r") as f:
                    self.stats = json.load(f)
            except Exception:
                self.stats = {}
    
    def save(self) -> None:
        """Save statistics to file."""
        try:
            with open(self.stats_file, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save source stats: {e}")
    
    def update_stats(
        self,
        source: str,
        candidates: int,
        approved: int,
        subreddit: str = None
    ) -> None:
        """
        Update statistics for a source.
        
        Args:
            source: Source name (reddit, unsplash, pexels).
            candidates: Number of candidates fetched.
            approved: Number approved.
            subreddit: Optional subreddit name for Reddit sources.
        """
        key = f"{source}_{subreddit}" if subreddit else source
        
        if key not in self.stats:
            self.stats[key] = {
                "total_candidates": 0,
                "total_approved": 0,
                "history": []
            }
        
        self.stats[key]["total_candidates"] += candidates
        self.stats[key]["total_approved"] += approved
        
        # Add to 30-day rolling history
        rate = approved / candidates if candidates > 0 else 0
        self.stats[key]["history"].append({
            "date": datetime.now().isoformat(),
            "candidates": candidates,
            "approved": approved,
            "rate": round(rate, 4)
        })
        
        # Keep only last 30 entries
        self.stats[key]["history"] = self.stats[key]["history"][-30:]
        
        # Calculate rolling acceptance rate
        recent = self.stats[key]["history"]
        total_cand = sum(h["candidates"] for h in recent)
        total_appr = sum(h["approved"] for h in recent)
        self.stats[key]["rolling_rate"] = round(
            total_appr / total_cand if total_cand > 0 else 0, 4
        )
        
        self.save()
    
    def get_recommendations(self) -> dict[str, str]:
        """
        Get recommendations for source allocation adjustments.
        
        Returns:
            Dictionary with source recommendations.
        """
        recommendations = {}
        
        for source, data in self.stats.items():
            rate = data.get("rolling_rate", 0)
            
            if rate >= 0.30:
                recommendations[source] = "increase"
            elif rate >= 0.10:
                recommendations[source] = "maintain"
            elif rate >= 0.05:
                recommendations[source] = "reduce"
            else:
                recommendations[source] = "consider_removing"
        
        return recommendations
