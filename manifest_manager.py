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
        Create an optimized manifest entry for a wallpaper.
        
        Optimizations applied:
        - Short key names (cat, qs, hue, etc.) - saves ~40% on keys
        - Arrays instead of objects for dimensions [w, h]
        - Removed duplicate fields (category/primary_category)
        - Omit empty/null values
        - Unix timestamp for dates
        - Short embedding keys (mn4, en2, sig, din)
        
        Args:
            wallpaper: ApprovedWallpaper object.
            r2_url: R2 storage URL if uploaded.
        
        Returns:
            Dictionary representing optimized manifest entry.
        """
        meta = wallpaper.metadata
        embeddings = wallpaper.embeddings
        
        # Base entry with short keys
        entry = {
            "id": wallpaper.id,
            "t": wallpaper.title,  # title
            "cat": meta.primary_category,  # category
            "src": wallpaper.source,  # source
            "ts": int(datetime.now().timestamp()),  # timestamp (unix)
            
            # Dimensions as array [width, height] - saves ~20 bytes per entry
            "dim": [self._to_native(meta.width), self._to_native(meta.height)],
            "ar": round(float(self._to_native(meta.aspect_ratio)), 2),  # aspect_ratio
            "sz": self._to_native(meta.file_size),  # file_size
            
            # Quality (short keys)
            "qs": round(float(self._to_native(wallpaper.quality_scores.final_score)), 3),  # quality_score
            "qt": meta.quality_tier,  # quality_tier
            "aes": round(float(self._to_native(meta.aesthetic_score)), 1),  # aesthetic
            
            # Colors (short keys)
            "col": meta.color_palette,  # colors
            "hue": self._to_native(meta.dominant_hue),  # dominant_hue
            "bri": self._to_native(meta.brightness),  # brightness
            "ctr": round(float(self._to_native(meta.contrast_ratio)), 1),  # contrast
            "dark": self._to_native(meta.is_dark_mode_friendly),  # is_dark_mode_friendly
            
            # Composition as compact object
            "comp": {
                "t": meta.composition_type,  # type
                "s": round(float(self._to_native(meta.symmetry_score)), 2),  # symmetry
                "d": round(float(self._to_native(meta.depth_score)), 1),  # depth
                "c": meta.complexity_level,  # complexity
                "fp": self._safe_focal_point(meta.focal_point),  # focal_point
            },
            
            # Tags (already short)
            "mood": meta.mood_tags,
            "style": meta.style_tags,
            
            # Embeddings with short keys
            "emb": {
                "mn4": self._encode_embedding(embeddings.mobilenet_v4),  # mobilenet_v4
                "en2": self._encode_embedding(embeddings.efficientnet_v2),  # efficientnet_v2
                "sig": self._encode_embedding(embeddings.siglip),  # siglip
                "din": self._encode_embedding(embeddings.dinov3),  # dinov3
            },
            
            "v": 2  # manifest version (short)
        }
        
        # URL - only include if exists (save bytes on empty strings)
        if r2_url:
            entry["url"] = r2_url
        elif wallpaper.filepath:
            entry["url"] = str(wallpaper.filepath)
        
        # Subcategories - only if non-empty
        if meta.subcategories:
            entry["sub"] = meta.subcategories
        
        # ML category - only if different from primary
        if meta.ml_category and meta.ml_category != meta.primary_category:
            entry["mlc"] = meta.ml_category
            entry["mlcf"] = round(float(self._to_native(meta.ml_confidence)), 2)
        
        # Source metadata - only include non-empty values
        src_meta = {}
        if meta.subreddit:
            src_meta["sr"] = meta.subreddit
        if self._to_native(meta.upvotes):
            src_meta["up"] = self._to_native(meta.upvotes)
        if meta.post_url:
            src_meta["pu"] = meta.post_url
        if meta.source_url:
            src_meta["su"] = meta.source_url
        if src_meta:
            entry["srcm"] = src_meta
        
        # Artist - only if known
        if wallpaper.artist and wallpaper.artist.lower() not in ["unknown", ""]:
            entry["art"] = wallpaper.artist
            if meta.artist_url:
                entry["artu"] = meta.artist_url
        
        # License - only if specified
        if meta.license_type:
            entry["lic"] = meta.license_type
        
        # Color diversity - only if meaningful
        if meta.color_diversity and self._to_native(meta.color_diversity) > 0:
            entry["cdiv"] = round(float(self._to_native(meta.color_diversity)), 2)
        
        return entry
    
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
                # Use the output filename as the R2 key
                r2_url = self.r2_manager.upload_manifest(output_path, key=output_path.name)
                logger.info(f"📤 Manifest synced to R2: {r2_url}")
            except Exception as e:
                logger.error(f"Failed to upload manifest to R2: {e}")
                # Continue - local save succeeded
    
    def _collect_existing_manifest_names(self, base_name: str, ext: str) -> set[str]:
        """Collect manifest filenames locally and on R2 to avoid collisions."""
        existing: set[str] = set()
        pattern = f"{base_name}*{ext}"
        for path in self.config.manifests_dir.glob(pattern):
            existing.add(path.name)
        
        if self.r2_manager:
            try:
                for key in self.r2_manager.list_manifests():
                    name = Path(key).name
                    if name.startswith(base_name) and name.endswith(ext):
                        existing.add(name)
            except Exception as e:
                logger.warning(f"Failed to list manifests from R2: {e}")
        
        return existing
    
    def _get_next_manifest_filename(self, date_str: str, test_mode: bool = False) -> str:
        """
        Get the next available manifest filename for today.
        
        Uses run numbering: date.json.gz, date-2.json.gz, date-3.json.gz, etc.
        For test mode: test_date.json.gz, test_date-2.json.gz, etc.
        
        Looks at both local files and existing R2 manifests so that CI runs do not
        overwrite manual runs from the same day.
        """
        prefix = "test_" if test_mode else ""
        base_name = f"{prefix}{date_str}"
        ext = ".json.gz" if self.config.compressed else ".json"
        
        existing = self._collect_existing_manifest_names(base_name, ext)
        if not existing:
            return f"{base_name}{ext}"
        
        max_run = 1 if f"{base_name}{ext}" in existing else 0
        suffix_len = len(ext)
        for name in existing:
            if name.startswith(f"{base_name}-") and name.endswith(ext):
                try:
                    run_part = name[len(base_name) + 1:-suffix_len]
                    max_run = max(max_run, int(run_part))
                except ValueError:
                    continue
        
        next_run = max_run + 1 if max_run else 1
        if next_run > 100:
            logger.warning(f"Too many manifest files for {date_str}, overwriting last")
        return f"{base_name}-{next_run}{ext}" if next_run > 1 else f"{base_name}{ext}"
    
    def _save_date_manifest(
        self, 
        date_str: str, 
        wallpapers: list[dict],
        test_mode: bool = False
    ) -> Path:
        """
        Save manifest for a specific date with run numbering to temp directory.
        
        Saves to temp/manifests/ to avoid Git tracking (R2-only storage).
        Creates new file for each run: date.json.gz, date-2.json.gz, etc.
        Test runs use prefix: test_date.json.gz
        """
        # Use temp directory for R2-only storage (not tracked by Git)
        temp_manifests_dir = Path("./temp/manifests")
        temp_manifests_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_filename = self._get_next_manifest_filename(date_str, test_mode)
        manifest_path = temp_manifests_dir / manifest_filename
        
        manifest = {
            "version": "1.2",
            "date": date_str,
            "created": datetime.now().isoformat() + "Z",
            "count": len(wallpapers),
            "is_test": test_mode,
            "wallpapers": wallpapers
        }
        
        if self.config.compressed:
            with gzip.open(manifest_path, "wt", encoding="utf-8") as f:
                json.dump(manifest, f)
        else:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        
        mode_label = "🧪 TEST" if test_mode else "📅"
        logger.info(f"{mode_label} Saved manifest to temp (R2-only): {manifest_path} ({len(wallpapers)} wallpapers)")
        return manifest_path
    
    def update_manifest(
        self,
        approved_wallpapers: list,  # List of ApprovedWallpaper
        r2_urls: dict[str, str] = None,
        test_mode: bool = False
    ) -> tuple[int, Path]:
        """
        Update manifest with new approved wallpapers using date-wise storage.
        
        Each pipeline run creates a new file:
        - Production: 2025_12_28.json.gz, 2025_12_28-2.json.gz, etc.
        - Test mode:  test_2025_12_28.json.gz, test_2025_12_28-2.json.gz, etc.
        
        This avoids:
        - Override issues in CI/CD (each run has its own file)
        - Heavy file loading/merging (no need to load existing data)
        - Test data mixing with production data
        
        Args:
            approved_wallpapers: List of ApprovedWallpaper objects.
            r2_urls: Optional mapping of wallpaper_id to R2 URL.
            test_mode: If True, prefix filename with "test_".
        
        Returns:
            Tuple of (number of entries, manifest file path).
        """
        if r2_urls is None:
            r2_urls = {}
        
        if not approved_wallpapers:
            logger.info("No wallpapers to add to manifest")
            return 0, Path()
        
        # Create entries for all approved wallpapers
        entries = []
        for wp in approved_wallpapers:
            r2_url = r2_urls.get(wp.id, "")
            entry = self._create_manifest_entry(wp, r2_url)
            entries.append(entry)
        
        # Save to new date-wise file (with run numbering)
        date_str = datetime.now().strftime("%Y_%m_%d")
        manifest_path = self._save_date_manifest(date_str, entries, test_mode)
        
        # Upload to R2 if configured
        if self.r2_manager and manifest_path.exists():
            try:
                # Use the manifest filename as the R2 key (date-wise naming)
                # Test runs go to test/ subfolder in R2
                r2_key = f"test/{manifest_path.name}" if test_mode else manifest_path.name
                self.r2_manager.upload_manifest(manifest_path, key=r2_key)
                logger.info(f"📤 Manifest synced to R2: {r2_key}")
            except Exception as e:
                logger.warning(f"Failed to upload manifest to R2: {e}")
        
        # Update dedup index with new IDs (skip for test mode)
        if self.dedup_index and not test_mode:
            for wp in approved_wallpapers:
                self.dedup_index.add_id(wp.id)
        
        logger.info(f"✅ Added {len(entries)} wallpapers to manifest")
        
        return len(entries), manifest_path


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
