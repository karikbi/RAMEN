#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Part 2: Hard Filters

Contains hard filter checks for resolution, file integrity, aspect ratio,
text detection, and file size validation.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import imagehash
from PIL import Image

# Try to import pytesseract, but make it optional
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger("wallpaper_curator")


@dataclass
class FilterResult:
    """Result of applying filters to a candidate."""
    passed: bool
    reason: Optional[str] = None
    width: int = 0
    height: int = 0
    file_size: int = 0
    aspect_ratio: float = 0.0
    phash: Optional[str] = None


@dataclass  
class FilterConfig:
    """Configuration for hard filters."""
    min_width: int = 2560
    min_height: int = 1440
    min_file_size: int = 200 * 1024  # 200KB
    max_file_size: int = 15 * 1024 * 1024  # 15MB
    min_aspect_ratio: float = 0.5  # Portrait minimum (e.g., 9:16)
    max_aspect_ratio: float = 3.0  # Ultra-wide maximum (e.g., 21:9)
    max_text_coverage: float = 0.50  # 50% (relaxed from 30% to reduce false positives)
    hash_similarity_threshold: int = 10  # Hamming distance (lower = more similar)
    rejected_dir: Path = Path("./temp/rejected")


class HardFilters:
    """Applies hard filters to reject unsuitable wallpaper candidates."""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.config.rejected_dir.mkdir(parents=True, exist_ok=True)
        self.existing_hashes: dict[str, str] = {}
        self._load_existing_hashes()
    
    def _load_existing_hashes(self) -> None:
        """Load existing collection's perceptual hashes."""
        hash_file = Path("./existing_hashes.json")
        if hash_file.exists():
            try:
                with open(hash_file, "r") as f:
                    self.existing_hashes = json.load(f)
                logger.info(f"Loaded {len(self.existing_hashes)} existing hashes")
            except Exception as e:
                logger.warning(f"Failed to load existing hashes: {e}")
    
    def save_hashes(self, new_hashes: dict[str, str]) -> None:
        """Save updated hashes to file."""
        self.existing_hashes.update(new_hashes)
        hash_file = Path("./existing_hashes.json")
        try:
            with open(hash_file, "w") as f:
                json.dump(self.existing_hashes, f)
        except Exception as e:
            logger.error(f"Failed to save hashes: {e}")
    
    def check_file_integrity(self, filepath: Path) -> Tuple[bool, Optional[Image.Image], str]:
        """Verify image can be opened and is valid format."""
        try:
            img = Image.open(filepath)
            img.verify()
            # Re-open after verify (verify closes the file)
            img = Image.open(filepath)
            img.load()  # Force load to catch truncated images
            
            if img.format not in ["JPEG", "PNG", "WEBP"]:
                return False, None, f"Invalid format: {img.format}"
            
            return True, img, ""
        except Exception as e:
            return False, None, f"Corrupted file: {str(e)}"
    
    def check_resolution(self, img: Image.Image) -> Tuple[bool, str]:
        """Check minimum resolution requirements."""
        width, height = img.size
        if width < self.config.min_width or height < self.config.min_height:
            return False, f"Resolution too low: {width}x{height} (min: {self.config.min_width}x{self.config.min_height})"
        return True, ""
    
    def check_file_size(self, filepath: Path) -> Tuple[bool, int, str]:
        """Check file size is within acceptable range."""
        size = filepath.stat().st_size
        if size < self.config.min_file_size:
            return False, size, f"File too small: {size / 1024:.1f}KB"
        if size > self.config.max_file_size:
            return False, size, f"File too large: {size / (1024*1024):.1f}MB"
        return True, size, ""
    
    def check_aspect_ratio(self, img: Image.Image) -> Tuple[bool, float, str]:
        """Check aspect ratio is within acceptable range."""
        width, height = img.size
        aspect = width / height
        
        # Accept both landscape and portrait
        if aspect < 1:  # Portrait
            aspect = 1 / aspect
        
        if aspect < self.config.min_aspect_ratio:
            return False, aspect, f"Aspect ratio too narrow: {aspect:.2f}"
        if aspect > self.config.max_aspect_ratio:
            return False, aspect, f"Aspect ratio too wide: {aspect:.2f}"
        return True, aspect, ""
    
    def check_text_coverage(self, img: Image.Image) -> Tuple[bool, float, str]:
        """Detect text using OCR and calculate coverage."""
        if not TESSERACT_AVAILABLE:
            return True, 0.0, ""  # Skip if tesseract not available
        
        try:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize for faster OCR
            max_size = 800
            ratio = min(max_size / img.width, max_size / img.height)
            if ratio < 1:
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                img_resized = img
            
            # Run OCR
            data = pytesseract.image_to_data(img_resized, output_type=pytesseract.Output.DICT)
            
            # Calculate text coverage
            total_pixels = img_resized.width * img_resized.height
            text_pixels = 0
            
            for i, conf in enumerate(data["conf"]):
                if int(conf) > 60:  # Confidence threshold (raised from 30 to reduce false positives)
                    w = data["width"][i]
                    h = data["height"][i]
                    text_pixels += w * h
            
            coverage = text_pixels / total_pixels
            
            if coverage > self.config.max_text_coverage:
                return False, coverage, f"Too much text: {coverage*100:.1f}%"
            return True, coverage, ""
        except Exception as e:
            logger.debug(f"OCR check failed: {e}")
            return True, 0.0, ""  # Pass if OCR fails
    
    def check_duplicate(self, img: Image.Image, candidate_id: str) -> Tuple[bool, Optional[str], str]:
        """Check for duplicates using perceptual hash."""
        try:
            phash = imagehash.phash(img)
            phash_str = str(phash)
            
            # Compare against existing hashes
            for existing_id, existing_hash in self.existing_hashes.items():
                existing_phash = imagehash.hex_to_hash(existing_hash)
                distance = phash - existing_phash
                
                if distance <= self.config.hash_similarity_threshold:
                    return False, phash_str, f"Duplicate of {existing_id} (distance: {distance})"
            
            return True, phash_str, ""
        except Exception as e:
            logger.warning(f"Hash check failed: {e}")
            return True, None, ""
    
    def reject_candidate(self, filepath: Path, reason: str) -> None:
        """Move rejected candidate to rejected folder with reason logged."""
        try:
            rejected_path = self.config.rejected_dir / filepath.name
            shutil.move(str(filepath), str(rejected_path))
            logger.debug(f"Rejected {filepath.name}: {reason}")
        except Exception as e:
            logger.error(f"Failed to move rejected file: {e}")
    
    def apply_all_filters(self, filepath: Path, candidate_id: str) -> FilterResult:
        """Apply all hard filters to a candidate."""
        # File size check
        size_ok, file_size, reason = self.check_file_size(filepath)
        if not size_ok:
            return FilterResult(passed=False, reason=reason, file_size=file_size)
        
        # File integrity
        integrity_ok, img, reason = self.check_file_integrity(filepath)
        if not integrity_ok or img is None:
            return FilterResult(passed=False, reason=reason)
        
        width, height = img.size
        
        # Resolution
        res_ok, reason = self.check_resolution(img)
        if not res_ok:
            return FilterResult(passed=False, reason=reason, width=width, height=height)
        
        # Aspect ratio
        aspect_ok, aspect, reason = self.check_aspect_ratio(img)
        if not aspect_ok:
            return FilterResult(passed=False, reason=reason, width=width, height=height, aspect_ratio=aspect)
        
        # Text detection
        text_ok, text_coverage, reason = self.check_text_coverage(img)
        if not text_ok:
            return FilterResult(passed=False, reason=reason, width=width, height=height, aspect_ratio=aspect)
        
        # Duplicate check
        dup_ok, phash, reason = self.check_duplicate(img, candidate_id)
        if not dup_ok:
            return FilterResult(passed=False, reason=reason, width=width, height=height, 
                              aspect_ratio=aspect, phash=phash)
        
        return FilterResult(
            passed=True,
            width=width,
            height=height,
            file_size=file_size,
            aspect_ratio=aspect,
            phash=phash
        )
