#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Deduplication Manager

Smart multi-layer duplicate detection system to prevent:
1. Re-fetching the same URLs
2. Re-processing the same wallpaper IDs
3. Visual duplicates via perceptual hash
4. Exact duplicates via content hash (SHA256)

Uses Bloom filter for memory-efficient initial screening.
Syncs dedup index to Cloudflare R2 for persistence across runs.
"""

import base64
import gzip
import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Set, Tuple

import imagehash
from PIL import Image

# Try to import pybloom_live for Bloom filter, make it optional
try:
    from pybloom_live import BloomFilter
    BLOOM_AVAILABLE = True
except ImportError:
    BLOOM_AVAILABLE = False

logger = logging.getLogger("wallpaper_curator")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DedupConfig:
    """Configuration for deduplication system."""
    # URL tracking
    track_urls: bool = True
    max_urls: int = 100000
    
    # Perceptual hash
    phash_threshold: int = 10  # Hamming distance (lower = stricter)
    
    # Content hash (SHA256)
    content_hash_enabled: bool = True
    
    # Bloom filter settings
    bloom_expected_items: int = 50000
    bloom_false_positive_rate: float = 0.01
    
    # R2 sync
    sync_to_r2: bool = True
    r2_index_key: str = "dedup/index.json.gz"
    
    # Local cache
    local_cache_dir: Path = field(default_factory=lambda: Path("./dedup_cache"))


# =============================================================================
# DUPLICATE INDEX
# =============================================================================

@dataclass
class DuplicateIndex:
    """
    Centralized index for all deduplication data.
    
    Tracks:
    - seen_urls: Source URLs already fetched
    - seen_ids: Wallpaper IDs (reddit_xxx, unsplash_xxx, pexels_xxx)
    - phash_index: Perceptual hash → wallpaper_id mapping
    - content_hashes: SHA256 → wallpaper_id mapping
    """
    seen_urls: Set[str] = field(default_factory=set)
    seen_ids: Set[str] = field(default_factory=set)
    phash_index: dict[str, str] = field(default_factory=dict)  # phash_hex → wp_id
    content_hashes: dict[str, str] = field(default_factory=dict)  # sha256 → wp_id
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
            "seen_urls": list(self.seen_urls),
            "seen_ids": list(self.seen_ids),
            "phash_index": self.phash_index,
            "content_hashes": self.content_hashes,
            "stats": {
                "url_count": len(self.seen_urls),
                "id_count": len(self.seen_ids),
                "phash_count": len(self.phash_index),
                "content_hash_count": len(self.content_hashes),
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DuplicateIndex":
        """Deserialize from dictionary."""
        return cls(
            seen_urls=set(data.get("seen_urls", [])),
            seen_ids=set(data.get("seen_ids", [])),
            phash_index=data.get("phash_index", {}),
            content_hashes=data.get("content_hashes", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            version=data.get("version", "1.0"),
        )
    
    def add_url(self, url: str) -> None:
        """Add a URL to the seen set."""
        self.seen_urls.add(url)
    
    def add_id(self, wp_id: str) -> None:
        """Add a wallpaper ID to the seen set."""
        self.seen_ids.add(wp_id)
    
    def add_phash(self, phash_hex: str, wp_id: str) -> None:
        """Add a perceptual hash to the index."""
        self.phash_index[phash_hex] = wp_id
    
    def add_content_hash(self, sha256: str, wp_id: str) -> None:
        """Add a content hash to the index."""
        self.content_hashes[sha256] = wp_id
    
    def has_url(self, url: str) -> bool:
        """Check if URL has been seen."""
        return url in self.seen_urls
    
    def has_id(self, wp_id: str) -> bool:
        """Check if wallpaper ID has been seen."""
        return wp_id in self.seen_ids
    
    def find_similar_phash(self, phash: imagehash.ImageHash, threshold: int = 10) -> Tuple[bool, Optional[str], int]:
        """
        Find visually similar image by perceptual hash.
        
        Args:
            phash: Perceptual hash of the image to check.
            threshold: Maximum Hamming distance to consider a match.
        
        Returns:
            Tuple of (is_duplicate, matching_wp_id, distance)
        """
        phash_str = str(phash)
        
        for existing_hex, wp_id in self.phash_index.items():
            try:
                existing_phash = imagehash.hex_to_hash(existing_hex)
                distance = phash - existing_phash
                if distance <= threshold:
                    return True, wp_id, distance
            except Exception:
                continue
        
        return False, None, -1
    
    def get_stats(self) -> dict[str, int]:
        """Get statistics about the index."""
        return {
            "urls": len(self.seen_urls),
            "ids": len(self.seen_ids),
            "phashes": len(self.phash_index),
            "content_hashes": len(self.content_hashes),
        }


# =============================================================================
# DUPLICATE CHECKER
# =============================================================================

class DuplicateChecker:
    """
    Fast duplicate checking with multiple strategies.
    
    Provides layered checking from fastest to slowest:
    1. URL check (instant)
    2. ID check (instant)
    3. Bloom filter (fast, probabilistic)
    4. Content hash (requires file)
    5. Perceptual hash (requires image processing)
    """
    
    def __init__(self, index: DuplicateIndex, config: Optional[DedupConfig] = None):
        self.index = index
        self.config = config or DedupConfig()
        self.bloom: Optional[BloomFilter] = None
        
        # Initialize Bloom filter if available
        if BLOOM_AVAILABLE:
            self._init_bloom_filter()
    
    def _init_bloom_filter(self) -> None:
        """Initialize Bloom filter with existing data."""
        try:
            self.bloom = BloomFilter(
                capacity=self.config.bloom_expected_items,
                error_rate=self.config.bloom_false_positive_rate
            )
            
            # Add existing IDs to bloom filter
            for wp_id in self.index.seen_ids:
                self.bloom.add(wp_id)
            
            # Add existing URLs to bloom filter
            for url in self.index.seen_urls:
                self.bloom.add(url)
            
            logger.debug(f"Bloom filter initialized with {len(self.index.seen_ids)} IDs and {len(self.index.seen_urls)} URLs")
        except Exception as e:
            logger.warning(f"Failed to initialize Bloom filter: {e}")
            self.bloom = None
    
    def quick_check(self, wp_id: str, url: str = "") -> bool:
        """
        Fast probabilistic check using Bloom filter.
        
        Returns True if DEFINITELY NOT seen (can proceed).
        Returns False if POSSIBLY seen (need full check).
        """
        if self.bloom is None:
            return True  # Skip if bloom not available
        
        # If neither in bloom filter, definitely not seen
        if wp_id not in self.bloom and (not url or url not in self.bloom):
            return True
        
        return False  # Possibly seen, need full check
    
    def check_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Check if URL has been seen before.
        
        Args:
            url: Source URL to check.
        
        Returns:
            Tuple of (is_duplicate, None) - URL doesn't give us the matching ID
        """
        if not self.config.track_urls or not url:
            return False, None
        
        if self.index.has_url(url):
            return True, None
        
        return False, None
    
    def check_id(self, wp_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if wallpaper ID has been processed before.
        
        Args:
            wp_id: Wallpaper ID (e.g., "reddit_abc123").
        
        Returns:
            Tuple of (is_duplicate, matching_id)
        """
        if self.index.has_id(wp_id):
            return True, wp_id
        
        return False, None
    
    def check_content(self, filepath: Path) -> Tuple[bool, Optional[str]]:
        """
        Check for exact duplicate via SHA256 content hash.
        
        Args:
            filepath: Path to the file to check.
        
        Returns:
            Tuple of (is_duplicate, matching_wp_id)
        """
        if not self.config.content_hash_enabled:
            return False, None
        
        if not filepath.exists():
            return False, None
        
        try:
            sha256 = self._compute_sha256(filepath)
            
            if sha256 in self.index.content_hashes:
                matching_id = self.index.content_hashes[sha256]
                logger.debug(f"Content hash match: {filepath.name} matches {matching_id}")
                return True, matching_id
            
            return False, None
        except Exception as e:
            logger.warning(f"Content hash check failed for {filepath}: {e}")
            return False, None
    
    def check_visual(self, img: Image.Image, candidate_id: str) -> Tuple[bool, Optional[str], int]:
        """
        Check for visual duplicate via perceptual hash.
        
        Args:
            img: PIL Image to check.
            candidate_id: ID of the candidate (for logging).
        
        Returns:
            Tuple of (is_duplicate, matching_wp_id, hamming_distance)
        """
        try:
            phash = imagehash.phash(img)
            
            is_dup, match_id, distance = self.index.find_similar_phash(
                phash, 
                threshold=self.config.phash_threshold
            )
            
            if is_dup:
                logger.debug(f"Visual match: {candidate_id} matches {match_id} (distance: {distance})")
            
            return is_dup, match_id, distance
        except Exception as e:
            logger.warning(f"Visual hash check failed for {candidate_id}: {e}")
            return False, None, -1
    
    def full_check(
        self,
        wp_id: str,
        url: str = "",
        filepath: Optional[Path] = None,
        img: Optional[Image.Image] = None
    ) -> Tuple[bool, Optional[str], str]:
        """
        Perform full duplicate check through all layers.
        
        Args:
            wp_id: Wallpaper ID.
            url: Source URL (optional).
            filepath: Path to file (optional, for content hash).
            img: PIL Image (optional, for visual hash).
        
        Returns:
            Tuple of (is_duplicate, matching_id, reason)
        """
        # Layer 1: URL check
        if url:
            is_dup, _ = self.check_url(url)
            if is_dup:
                return True, None, f"URL already fetched: {url[:50]}..."
        
        # Layer 2: ID check
        is_dup, match_id = self.check_id(wp_id)
        if is_dup:
            return True, match_id, f"ID already processed: {wp_id}"
        
        # Layer 3: Content hash (if file available)
        if filepath and filepath.exists():
            is_dup, match_id = self.check_content(filepath)
            if is_dup:
                return True, match_id, f"Exact content match: {match_id}"
        
        # Layer 4: Visual hash (if image available)
        if img:
            is_dup, match_id, distance = self.check_visual(img, wp_id)
            if is_dup:
                return True, match_id, f"Visual match: {match_id} (distance: {distance})"
        
        return False, None, ""
    
    def register(
        self,
        wp_id: str,
        url: str = "",
        filepath: Optional[Path] = None,
        img: Optional[Image.Image] = None
    ) -> None:
        """
        Register a new wallpaper in the dedup index.
        
        Call this after successfully processing a wallpaper.
        
        Args:
            wp_id: Wallpaper ID.
            url: Source URL.
            filepath: Path to file (for content hash).
            img: PIL Image (for visual hash).
        """
        # Register URL
        if url:
            self.index.add_url(url)
            if self.bloom:
                self.bloom.add(url)
        
        # Register ID
        self.index.add_id(wp_id)
        if self.bloom:
            self.bloom.add(wp_id)
        
        # Register content hash
        if filepath and filepath.exists():
            try:
                sha256 = self._compute_sha256(filepath)
                self.index.add_content_hash(sha256, wp_id)
            except Exception as e:
                logger.warning(f"Failed to compute content hash for {wp_id}: {e}")
        
        # Register visual hash
        if img:
            try:
                phash = imagehash.phash(img)
                self.index.add_phash(str(phash), wp_id)
            except Exception as e:
                logger.warning(f"Failed to compute visual hash for {wp_id}: {e}")
    
    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        """Compute SHA256 hash of file content."""
        sha256_hash = hashlib.sha256()
        
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()


# =============================================================================
# R2 SYNC FOR DEDUP INDEX
# =============================================================================

class R2DedupSync:
    """
    Syncs deduplication index to/from Cloudflare R2.
    
    Enables persistence of dedup data across pipeline runs
    when running on ephemeral environments like GitHub Actions.
    """
    
    def __init__(
        self,
        r2_client,
        bucket: str,
        index_key: str = "dedup/index.json.gz",
        local_cache_dir: Path = Path("./dedup_cache")
    ):
        self.r2_client = r2_client
        self.bucket = bucket
        self.index_key = index_key
        self.local_cache_dir = local_cache_dir
        self.local_cache_path = local_cache_dir / "index.json.gz"
        
        # Ensure cache directory exists
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_index(self) -> Optional[DuplicateIndex]:
        """
        Download dedup index from R2.
        
        Returns:
            DuplicateIndex if found, None otherwise.
        """
        if self.r2_client is None:
            logger.warning("R2 client not configured, skipping index download")
            return self._load_local_cache()
        
        try:
            logger.info(f"📥 Downloading dedup index from R2: {self.index_key}")
            
            response = self.r2_client.get_object(
                Bucket=self.bucket,
                Key=self.index_key
            )
            
            # Read and decompress
            compressed_data = response["Body"].read()
            json_data = gzip.decompress(compressed_data).decode("utf-8")
            data = json.loads(json_data)
            
            index = DuplicateIndex.from_dict(data)
            stats = index.get_stats()
            
            logger.info(
                f"✅ Loaded dedup index: {stats['urls']} URLs, {stats['ids']} IDs, "
                f"{stats['phashes']} phashes, {stats['content_hashes']} content hashes"
            )
            
            # Save to local cache
            self._save_local_cache(index)
            
            return index
            
        except self.r2_client.exceptions.NoSuchKey:
            logger.info("No existing dedup index found in R2, starting fresh")
            return DuplicateIndex()
        except Exception as e:
            logger.warning(f"Failed to download dedup index from R2: {e}")
            return self._load_local_cache()
    
    def upload_index(self, index: DuplicateIndex) -> bool:
        """
        Upload dedup index to R2.
        
        Args:
            index: DuplicateIndex to upload.
        
        Returns:
            True if successful, False otherwise.
        """
        if self.r2_client is None:
            logger.warning("R2 client not configured, saving to local cache only")
            self._save_local_cache(index)
            return False
        
        try:
            logger.info(f"📤 Uploading dedup index to R2: {self.index_key}")
            
            # Serialize and compress
            json_data = json.dumps(index.to_dict())
            compressed_data = gzip.compress(json_data.encode("utf-8"))
            
            self.r2_client.put_object(
                Bucket=self.bucket,
                Key=self.index_key,
                Body=compressed_data,
                ContentType="application/gzip"
                # Note: Removed ContentEncoding to avoid boto3 checksum mismatch
                # The file is stored as compressed bytes (.gz extension indicates compression)
            )
            
            stats = index.get_stats()
            logger.info(
                f"✅ Uploaded dedup index: {stats['urls']} URLs, {stats['ids']} IDs, "
                f"{stats['phashes']} phashes, {stats['content_hashes']} content hashes"
            )
            
            # Save to local cache as well
            self._save_local_cache(index)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload dedup index to R2: {e}")
            self._save_local_cache(index)
            return False
    
    def _save_local_cache(self, index: DuplicateIndex) -> None:
        """Save index to local cache file."""
        try:
            json_data = json.dumps(index.to_dict())
            compressed_data = gzip.compress(json_data.encode("utf-8"))
            
            with open(self.local_cache_path, "wb") as f:
                f.write(compressed_data)
            
            logger.debug(f"Saved dedup index to local cache: {self.local_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save local cache: {e}")
    
    def _load_local_cache(self) -> Optional[DuplicateIndex]:
        """Load index from local cache if available."""
        if not self.local_cache_path.exists():
            logger.debug("No local cache found")
            return DuplicateIndex()
        
        try:
            with open(self.local_cache_path, "rb") as f:
                compressed_data = f.read()
            
            json_data = gzip.decompress(compressed_data).decode("utf-8")
            data = json.loads(json_data)
            
            index = DuplicateIndex.from_dict(data)
            logger.info(f"Loaded dedup index from local cache")
            return index
        except Exception as e:
            logger.warning(f"Failed to load local cache: {e}")
            return DuplicateIndex()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_dedup_system(
    r2_client=None,
    bucket: str = "",
    config: Optional[DedupConfig] = None
) -> Tuple[DuplicateIndex, DuplicateChecker, Optional[R2DedupSync]]:
    """
    Factory function to create the complete dedup system.
    
    Args:
        r2_client: Optional boto3 S3 client configured for R2.
        bucket: R2 bucket name.
        config: Deduplication configuration.
    
    Returns:
        Tuple of (DuplicateIndex, DuplicateChecker, R2DedupSync or None)
    """
    config = config or DedupConfig()
    
    # Create R2 sync if client provided
    r2_sync = None
    if r2_client and bucket:
        r2_sync = R2DedupSync(
            r2_client=r2_client,
            bucket=bucket,
            index_key=config.r2_index_key,
            local_cache_dir=config.local_cache_dir
        )
        
        # Download existing index
        index = r2_sync.download_index() or DuplicateIndex()
    else:
        index = DuplicateIndex()
        logger.info("R2 not configured, using fresh dedup index")
    
    # Create checker
    checker = DuplicateChecker(index, config)
    
    return index, checker, r2_sync
