#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Watermark Detection and Cropping

Detects and removes Reddit watermarks (black bar with "Posted in r/..." text)
from the bottom of images.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("wallpaper_curator")


class WatermarkCropper:
    """Detects and crops watermarks from wallpaper images."""
    
    # Reddit watermark characteristics
    WATERMARK_HEIGHT_RANGE = (35, 70)  # pixels
    DARK_THRESHOLD = 40  # RGB value threshold for "black" bar
    MIN_DARK_RATIO = 0.85  # Minimum percentage of dark pixels in bar
    
    def __init__(self):
        pass
    
    def detect_reddit_watermark(self, img: Image.Image) -> Optional[int]:
        """
        Detect Reddit watermark bar at bottom of image.
        
        The Reddit watermark is a black bar (~40-60px) at the bottom containing
        white text like "Posted in r/wallpaper by u/username"
        
        Args:
            img: PIL Image to check
            
        Returns:
            Height of watermark bar to crop, or None if not detected
        """
        try:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            width, height = img.size
            
            # Only check bottom portion
            max_check_height = min(80, height // 10)
            
            # Convert bottom portion to numpy array
            bottom_region = img.crop((0, height - max_check_height, width, height))
            bottom_array = np.array(bottom_region)
            
            # For each row from bottom, check if it's a dark bar
            for check_height in range(self.WATERMARK_HEIGHT_RANGE[0], 
                                      min(self.WATERMARK_HEIGHT_RANGE[1] + 1, max_check_height)):
                # Get the strip at this height from bottom
                strip_start = max_check_height - check_height
                strip = bottom_array[strip_start:, :, :]
                
                # Calculate darkness: all RGB channels should be dark
                is_dark = np.all(strip < self.DARK_THRESHOLD, axis=2)
                dark_ratio = np.mean(is_dark)
                
                if dark_ratio >= self.MIN_DARK_RATIO:
                    logger.debug(f"Detected Reddit watermark: {check_height}px bar (dark ratio: {dark_ratio:.2%})")
                    return check_height
            
            return None
            
        except Exception as e:
            logger.debug(f"Watermark detection failed: {e}")
            return None
    
    def crop_watermark(self, img: Image.Image, watermark_height: int) -> Image.Image:
        """
        Crop watermark from bottom of image.
        
        Args:
            img: PIL Image to crop
            watermark_height: Height in pixels to crop from bottom
            
        Returns:
            Cropped PIL Image
        """
        width, height = img.size
        return img.crop((0, 0, width, height - watermark_height))
    
    def process_image(self, filepath: Path, source: str = "") -> Tuple[bool, Optional[Path]]:
        """
        Process an image, detecting and cropping watermarks if present.
        
        Args:
            filepath: Path to the image file
            source: Source identifier (e.g., 'reddit', 'unsplash')
            
        Returns:
            Tuple of (was_cropped, new_filepath)
            - was_cropped: True if watermark was detected and removed
            - new_filepath: Path to processed image (same as input if no crop)
        """
        # Only check Reddit images for watermarks
        if source.lower() != "reddit":
            return False, filepath
        
        try:
            img = Image.open(filepath)
            
            # Detect watermark
            watermark_height = self.detect_reddit_watermark(img)
            
            if watermark_height is None:
                return False, filepath
            
            # Crop watermark
            cropped_img = self.crop_watermark(img, watermark_height)
            
            # Save back to same file - handle different formats
            suffix = filepath.suffix.lower()
            if suffix in ['.jpg', '.jpeg']:
                cropped_img.save(filepath, 'JPEG', quality=95)
            elif suffix == '.png':
                cropped_img.save(filepath, 'PNG')
            elif suffix == '.webp':
                cropped_img.save(filepath, 'WEBP', quality=95)
            else:
                cropped_img.save(filepath)
            
            logger.info(f"Cropped {watermark_height}px watermark from {filepath.name}")
            return True, filepath
            
        except Exception as e:
            logger.warning(f"Failed to process watermark for {filepath}: {e}")
            return False, filepath
    
    def process_batch(
        self,
        filepaths: list[Path],
        sources: list[str]
    ) -> dict:
        """
        Process multiple images for watermark removal.
        
        Args:
            filepaths: List of image file paths
            sources: List of source identifiers (same order as filepaths)
            
        Returns:
            Statistics dict with crop counts
        """
        stats = {
            "total": len(filepaths),
            "reddit": 0,
            "cropped": 0,
            "failed": 0
        }
        
        for filepath, source in zip(filepaths, sources):
            if source.lower() == "reddit":
                stats["reddit"] += 1
                try:
                    was_cropped, _ = self.process_image(filepath, source)
                    if was_cropped:
                        stats["cropped"] += 1
                except Exception:
                    stats["failed"] += 1
        
        logger.info(f"Watermark processing: {stats['cropped']}/{stats['reddit']} Reddit images cropped")
        return stats
