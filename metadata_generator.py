#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Part 2: Metadata Generation

Generates rich metadata for approved wallpapers including color palettes,
categories, tags, and composition analysis.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from sklearn.cluster import KMeans

logger = logging.getLogger("wallpaper_curator")


@dataclass
class WallpaperMetadata:
    """Complete metadata for an approved wallpaper."""
    # Core identification
    id: str = ""
    title: str = ""
    date_added: str = ""
    version: int = 1
    
    # Visual characteristics
    color_palette: list[str] = field(default_factory=list)  # 5 hex colors
    dominant_hue: int = 0  # 0-360 degrees
    brightness: int = 0  # 0-255
    contrast_ratio: float = 0.0
    color_diversity: float = 0.0
    
    # Scene understanding
    primary_category: str = ""
    subcategories: list[str] = field(default_factory=list)
    scene_elements: list[str] = field(default_factory=list)
    time_of_day: str = ""
    weather: str = ""
    season: str = ""
    
    # Composition analysis
    composition_type: str = ""
    symmetry_score: float = 0.0
    depth_score: float = 0.0
    complexity_level: str = ""
    focal_point: tuple[float, float] = (0.5, 0.5)
    
    # Aesthetic properties
    mood_tags: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    quality_tier: str = "Standard"  # Premium or Standard
    aesthetic_score: float = 0.0
    
    # Technical metadata
    width: int = 0
    height: int = 0
    file_size: int = 0
    aspect_ratio: float = 0.0
    format: str = ""
    exif_data: dict = field(default_factory=dict)
    storage_url: str = ""
    
    # Attribution
    source: str = ""
    source_url: str = ""
    artist: str = ""
    artist_url: str = ""
    license_type: str = ""
    copyright: str = ""
    
    # Reddit-specific
    subreddit: str = ""
    upvotes: int = 0
    post_url: str = ""
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


class MetadataGenerator:
    """Generates rich metadata for wallpaper images."""
    
    # Category definitions for auto-categorization
    CATEGORIES = {
        "nature": ["landscape", "mountain", "forest", "ocean", "beach", "lake", "river", "waterfall"],
        "urban": ["city", "skyline", "street", "architecture", "building", "bridge"],
        "space": ["galaxy", "nebula", "stars", "planet", "cosmos", "aurora"],
        "abstract": ["geometric", "pattern", "gradient", "fractal", "minimalist"],
        "art": ["illustration", "digital art", "painting", "concept art"],
        "technology": ["cyberpunk", "sci-fi", "futuristic", "neon"],
    }
    
    SCENE_ELEMENTS = [
        "water", "sky", "clouds", "mountains", "trees", "grass", "flowers",
        "sun", "moon", "stars", "snow", "rain", "fog", "city", "buildings",
        "road", "path", "people", "animals", "vehicles"
    ]
    
    MOODS = [
        "calm", "peaceful", "serene", "dramatic", "energetic", "mysterious",
        "romantic", "melancholic", "inspiring", "cozy", "epic", "surreal"
    ]
    
    STYLES = [
        "gruvbox", "nord", "dracula", "monokai", "solarized", "catppuccin",
        "cyberpunk", "vaporwave", "minimalist", "vintage", "modern", "natural"
    ]
    
    TIMES_OF_DAY = ["dawn", "morning", "noon", "afternoon", "dusk", "evening", "night"]
    WEATHERS = ["clear", "cloudy", "overcast", "rainy", "snowy", "foggy", "stormy"]
    SEASONS = ["spring", "summer", "autumn", "winter"]
    
    def __init__(self):
        pass
    
    def _lab_to_hex(self, lab_color: np.ndarray) -> str:
        """Convert LAB color to hex string."""
        # LAB to BGR
        lab_pixel = np.uint8([[[lab_color[0], lab_color[1], lab_color[2]]]])
        bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
        b, g, r = bgr_pixel[0][0]
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def extract_color_palette(self, bgr: np.ndarray, n_colors: int = 5) -> tuple[list[str], int, float]:
        """
        Extract dominant colors using KMeans in LAB space.
        
        Returns:
            Tuple of (hex_colors, dominant_hue, color_diversity)
        """
        # Convert to LAB
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        
        # Reshape and sample
        pixels = lab.reshape(-1, 3).astype(np.float32)
        sample_size = min(10000, len(pixels))
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        samples = pixels[indices]
        
        # Cluster
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(samples)
        
        # Convert to hex
        hex_colors = []
        for center in kmeans.cluster_centers_:
            hex_colors.append(self._lab_to_hex(center))
        
        # Calculate dominant hue from largest cluster
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_idx = labels[np.argmax(counts)]
        dominant_lab = kmeans.cluster_centers_[dominant_idx]
        
        # A and B channels give hue
        a, b = dominant_lab[1] - 128, dominant_lab[2] - 128
        hue = int(np.arctan2(b, a) * 180 / np.pi) % 360
        
        # Color diversity: spread of cluster centers
        center_distances = []
        for i, c1 in enumerate(kmeans.cluster_centers_):
            for c2 in kmeans.cluster_centers_[i+1:]:
                center_distances.append(np.linalg.norm(c1 - c2))
        diversity = np.mean(center_distances) / 100 if center_distances else 0
        
        return hex_colors, hue, min(diversity, 1.0)
    
    def calculate_brightness_contrast(self, gray: np.ndarray) -> tuple[int, float]:
        """Calculate average brightness and RMS contrast."""
        brightness = int(gray.mean())
        contrast = float(gray.std())
        return brightness, contrast
    
    def detect_composition_type(self, gray: np.ndarray) -> tuple[str, float, float]:
        """Detect composition type and calculate symmetry/depth scores."""
        h, w = gray.shape
        
        # Check symmetry
        left = gray[:, :w//2]
        right = np.fliplr(gray[:, w//2:])
        if left.shape == right.shape:
            symmetry = 1.0 - np.mean(np.abs(left.astype(float) - right.astype(float))) / 255
        else:
            symmetry = 0.5
        
        # Check for rule of thirds
        edges = cv2.Canny(gray, 50, 150)
        h3, w3 = h // 3, w // 3
        
        # Check intersection areas
        intersections = [
            edges[h3-20:h3+20, w3-20:w3+20],
            edges[h3-20:h3+20, 2*w3-20:2*w3+20],
            edges[2*h3-20:2*h3+20, w3-20:w3+20],
            edges[2*h3-20:2*h3+20, 2*w3-20:2*w3+20],
        ]
        thirds_score = sum(i.sum() for i in intersections if i.size > 0) / (255 * 40 * 40 * 4 + 1)
        
        # Detect leading lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=min(h,w)//3, maxLineGap=10)
        has_leading = lines is not None and len(lines) > 3
        
        # Determine composition type
        if symmetry > 0.7:
            comp_type = "symmetrical"
        elif thirds_score > 0.1:
            comp_type = "rule_of_thirds"
        elif has_leading:
            comp_type = "leading_lines"
        else:
            comp_type = "centered"
        
        # Depth score (variance in edge density across horizontal bands)
        bands = [edges[:h//3, :], edges[h//3:2*h//3, :], edges[2*h//3:, :]]
        densities = [b.mean() for b in bands]
        depth = np.std(densities) / 50
        
        return comp_type, symmetry, min(depth, 1.0)
    
    def calculate_complexity(self, gray: np.ndarray) -> str:
        """Determine image complexity level."""
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = edges.sum() / (255 * edges.size)
        
        if edge_ratio < 0.05:
            return "minimal"
        elif edge_ratio < 0.15:
            return "low"
        elif edge_ratio < 0.30:
            return "medium"
        else:
            return "high"
    
    def find_focal_point(self, gray: np.ndarray) -> tuple[float, float]:
        """Find focal point using gradient magnitude."""
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        h, w = gray.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        total = magnitude.sum() + 1e-6
        cx = (x_coords * magnitude).sum() / total / w
        cy = (y_coords * magnitude).sum() / total / h
        
        return (float(cx), float(cy))
    
    def extract_exif(self, filepath) -> dict:
        """Extract EXIF data from image."""
        exif_data = {}
        try:
            img = Image.open(filepath)
            if hasattr(img, '_getexif') and img._getexif():
                for tag_id, value in img._getexif().items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, (str, int, float)):
                        exif_data[tag] = value
        except Exception:
            pass
        return exif_data
    
    def infer_category_from_metadata(self, title: str, source_metadata: dict) -> tuple[str, list[str]]:
        """Infer category from title and source metadata."""
        title_lower = title.lower()
        
        # Check subreddit-based inference
        subreddit = source_metadata.get("subreddit", "").lower()
        if "earthporn" in subreddit:
            return "nature", ["landscape", "photography"]
        elif "cityporn" in subreddit:
            return "urban", ["cityscape", "architecture"]
        elif "amoled" in subreddit:
            return "abstract", ["dark", "amoled"]
        elif "minimal" in subreddit:
            return "abstract", ["minimalist"]
        
        # Check title keywords
        for category, keywords in self.CATEGORIES.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return category, [keyword]
        
        return "general", []
    
    def generate_id(self, category: str) -> str:
        """Generate unique wallpaper ID."""
        timestamp = int(time.time())
        random_suffix = random.randint(1000, 9999)
        return f"{category}_{timestamp}_{random_suffix}"
    
    def generate_metadata(
        self,
        filepath: Path,
        title: str,
        artist: str,
        source: str,
        source_metadata: dict,
        quality_score: float = 0.0
    ) -> WallpaperMetadata:
        """
        Generate complete metadata for a wallpaper.
        
        Args:
            filepath: Path to the image file.
            title: Wallpaper title.
            artist: Artist/photographer name.
            source: Source platform (reddit/unsplash/pexels).
            source_metadata: Additional metadata from source.
            quality_score: Quality score from scoring system.
        
        Returns:
            WallpaperMetadata with all fields populated.
        """
        metadata = WallpaperMetadata()
        
        try:
            # Load image
            bgr = cv2.imread(str(filepath))
            if bgr is None:
                raise ValueError("Failed to load image")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            
            # Basic info
            h, w = bgr.shape[:2]
            file_stat = filepath.stat()
            
            # Infer category
            primary_cat, subcats = self.infer_category_from_metadata(title, source_metadata)
            
            # Generate ID
            metadata.id = self.generate_id(primary_cat)
            metadata.title = title
            metadata.date_added = time.strftime("%Y-%m-%d")
            
            # Colors
            colors, hue, diversity = self.extract_color_palette(bgr)
            metadata.color_palette = colors
            metadata.dominant_hue = hue
            metadata.color_diversity = diversity
            
            # Brightness/contrast
            brightness, contrast = self.calculate_brightness_contrast(gray)
            metadata.brightness = brightness
            metadata.contrast_ratio = contrast
            
            # Categories
            metadata.primary_category = primary_cat
            metadata.subcategories = subcats
            
            # Composition
            comp_type, symmetry, depth = self.detect_composition_type(gray)
            metadata.composition_type = comp_type
            metadata.symmetry_score = symmetry
            metadata.depth_score = depth
            metadata.complexity_level = self.calculate_complexity(gray)
            metadata.focal_point = self.find_focal_point(gray)
            
            # Quality tier
            metadata.quality_tier = "Premium" if quality_score >= 0.90 else "Standard"
            metadata.aesthetic_score = quality_score
            
            # Mood inference based on brightness and colors
            if brightness < 80:
                metadata.mood_tags = ["dark", "mysterious"]
            elif brightness > 180:
                metadata.mood_tags = ["bright", "cheerful"]
            else:
                metadata.mood_tags = ["balanced"]
            
            # Technical
            metadata.width = w
            metadata.height = h
            metadata.file_size = file_stat.st_size
            metadata.aspect_ratio = w / h
            metadata.format = filepath.suffix.upper().replace(".", "")
            metadata.exif_data = self.extract_exif(filepath)
            
            # Attribution
            metadata.source = source
            metadata.artist = artist
            
            # Source-specific
            if source == "reddit":
                metadata.subreddit = source_metadata.get("subreddit", "")
                metadata.upvotes = source_metadata.get("upvotes", 0)
                metadata.post_url = source_metadata.get("post_url", "")
            elif source == "unsplash":
                metadata.artist_url = source_metadata.get("photographer_url", "")
                metadata.source_url = source_metadata.get("unsplash_url", "")
                metadata.license_type = "Unsplash License"
            elif source == "pexels":
                metadata.artist_url = source_metadata.get("photographer_url", "")
                metadata.source_url = source_metadata.get("pexels_url", "")
                metadata.license_type = "Pexels License"
        
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
        
        return metadata
