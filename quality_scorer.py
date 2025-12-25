#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Part 2: Quality Scoring (DEPRECATED)

DEPRECATED: This module uses hand-coded heuristics that don't work well.
Use ml_quality_scorer.py instead, which uses SigLIP for semantic quality assessment.

Kept for reference and potential hybrid scoring in the future.
"""


import logging
from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

logger = logging.getLogger("wallpaper_curator")


@dataclass
class QualityScores:
    """Breakdown of quality scores by component."""
    visual_quality: float = 0.0
    composition: float = 0.0
    aesthetic_appeal: float = 0.0
    wallpaper_suitability: float = 0.0
    final_score: float = 0.0
    
    # Sub-scores for debugging
    sharpness: float = 0.0
    exposure: float = 0.0
    color_accuracy: float = 0.0
    artifacts: float = 0.0
    rule_of_thirds: float = 0.0
    balance: float = 0.0
    depth: float = 0.0
    leading_lines: float = 0.0
    color_harmony: float = 0.0
    contrast: float = 0.0
    visual_interest: float = 0.0
    subject_centering: float = 0.0
    brightness: float = 0.0
    texture_busyness: float = 0.0


@dataclass
class QualityConfig:
    """Configuration for quality scoring."""
    visual_weight: float = 0.40
    composition_weight: float = 0.30
    aesthetic_weight: float = 0.20
    suitability_weight: float = 0.10
    quality_threshold: float = 0.55  # Lowered for softmax-based ML scoring


class QualityScorer:
    """Calculates multi-factor quality scores for wallpaper candidates."""
    
    def __init__(self, config: QualityConfig = None):
        self.config = config or QualityConfig()
    
    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))
    
    def _load_image_cv(self, filepath) -> Tuple[np.ndarray, np.ndarray]:
        """Load image as BGR and grayscale numpy arrays."""
        img_bgr = cv2.imread(str(filepath))
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {filepath}")
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return img_bgr, img_gray
    
    # =========================================================================
    # VISUAL QUALITY (40% weight)
    # =========================================================================
    
    def calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        # Normalize: 100-2000 is typical range for good images
        score = self._clamp((variance - 100) / 1900)
        return score
    
    def calculate_exposure(self, gray: np.ndarray) -> float:
        """Analyze histogram for exposure quality."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        
        # Check for blown highlights (>250) and crushed shadows (<5)
        blown = hist[250:].sum()
        crushed = hist[:5].sum()
        
        # Penalize extremes
        blown_penalty = min(blown * 5, 0.5)
        crushed_penalty = min(crushed * 5, 0.5)
        
        # Check for good distribution (peak in middle range)
        mid_range = hist[50:200].sum()
        
        score = 0.7 * mid_range + 0.3 * (1 - blown_penalty - crushed_penalty)
        return self._clamp(score)
    
    def calculate_color_accuracy(self, bgr: np.ndarray) -> float:
        """Check for color casts via channel balance."""
        b, g, r = cv2.split(bgr)
        
        # Calculate mean of each channel
        means = [b.mean(), g.mean(), r.mean()]
        overall_mean = np.mean(means)
        
        # Check deviation from balanced (color cast detection)
        deviations = [abs(m - overall_mean) for m in means]
        max_deviation = max(deviations)
        
        # Lower deviation = better color balance
        score = 1.0 - self._clamp(max_deviation / 50)
        return score
    
    def calculate_artifacts(self, bgr: np.ndarray) -> float:
        """Detect JPEG compression artifacts via DCT analysis."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        # Sample 8x8 blocks and check for blockiness
        h, w = gray.shape
        block_size = 8
        
        if h < 64 or w < 64:
            return 1.0  # Too small to detect artifacts
        
        # Calculate horizontal and vertical gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Check for periodic patterns at 8-pixel intervals (JPEG blocks)
        x_power = np.abs(np.fft.fft(grad_x.mean(axis=0)))
        
        # Look for spike at frequency corresponding to 8-pixel blocks
        freq_8 = len(x_power) // 8
        if freq_8 > 0 and freq_8 < len(x_power):
            artifact_strength = x_power[freq_8] / (x_power.mean() + 1e-6)
            score = 1.0 - self._clamp(artifact_strength / 10)
        else:
            score = 1.0
        
        return score
    
    def score_visual_quality(self, bgr: np.ndarray, gray: np.ndarray) -> Tuple[float, dict]:
        """Calculate overall visual quality score."""
        sharpness = self.calculate_sharpness(gray)
        exposure = self.calculate_exposure(gray)
        color_acc = self.calculate_color_accuracy(bgr)
        artifacts = self.calculate_artifacts(bgr)
        
        score = (sharpness + exposure + color_acc + artifacts) / 4
        
        return self._clamp(score), {
            "sharpness": sharpness,
            "exposure": exposure,
            "color_accuracy": color_acc,
            "artifacts": artifacts
        }
    
    # =========================================================================
    # COMPOSITION (30% weight)
    # =========================================================================
    
    def calculate_rule_of_thirds(self, gray: np.ndarray) -> float:
        """Check if focal regions align with rule of thirds intersections."""
        h, w = gray.shape
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Rule of thirds intersection points
        thirds_x = [w // 3, 2 * w // 3]
        thirds_y = [h // 3, 2 * h // 3]
        
        # Check edge density near intersection points
        radius = min(h, w) // 10
        total_score = 0
        
        for tx in thirds_x:
            for ty in thirds_y:
                # Extract region around intersection
                x1, x2 = max(0, tx - radius), min(w, tx + radius)
                y1, y2 = max(0, ty - radius), min(h, ty + radius)
                region = edges[y1:y2, x1:x2]
                
                # Higher edge density = more likely subject is there
                density = region.sum() / (255 * region.size + 1e-6)
                total_score += density
        
        # Normalize
        return self._clamp(total_score * 2)
    
    def calculate_balance(self, gray: np.ndarray) -> float:
        """Compare left vs right, top vs bottom intensity distribution."""
        h, w = gray.shape
        
        # Split into quadrants
        left = gray[:, :w//2].mean()
        right = gray[:, w//2:].mean()
        top = gray[:h//2, :].mean()
        bottom = gray[h//2:, :].mean()
        
        # Calculate symmetry (lower difference = more balanced)
        lr_diff = abs(left - right) / 255
        tb_diff = abs(top - bottom) / 255
        
        score = 1.0 - (lr_diff + tb_diff) / 2
        return self._clamp(score)
    
    def calculate_depth(self, gray: np.ndarray) -> float:
        """Estimate depth through edge density in different regions."""
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide into 3 horizontal bands (foreground, middle, background)
        bands = [
            edges[2*h//3:, :],  # Foreground (bottom)
            edges[h//3:2*h//3, :],  # Middle
            edges[:h//3, :]  # Background (top)
        ]
        
        densities = [b.sum() / (255 * b.size + 1e-6) for b in bands]
        
        # Good depth: different densities in different layers
        variance = np.var(densities)
        score = self._clamp(variance * 100)
        return score
    
    def calculate_leading_lines(self, gray: np.ndarray) -> float:
        """Detect diagonal lines using Hough transform."""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=min(gray.shape)//4, maxLineGap=10)
        
        if lines is None:
            return 0.3  # No strong lines found
        
        diagonal_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = abs(np.arctan((y2-y1)/(x2-x1)) * 180 / np.pi)
                # Count lines between 20-70 degrees as "leading"
                if 20 < angle < 70:
                    diagonal_count += 1
        
        score = self._clamp(diagonal_count / 10)
        return score
    
    def score_composition(self, gray: np.ndarray) -> Tuple[float, dict]:
        """Calculate overall composition score."""
        thirds = self.calculate_rule_of_thirds(gray)
        balance = self.calculate_balance(gray)
        depth = self.calculate_depth(gray)
        lines = self.calculate_leading_lines(gray)
        
        score = (thirds + balance + depth + lines) / 4
        
        return self._clamp(score), {
            "rule_of_thirds": thirds,
            "balance": balance,
            "depth": depth,
            "leading_lines": lines
        }
    
    # =========================================================================
    # AESTHETIC APPEAL (20% weight)
    # =========================================================================
    
    def extract_dominant_colors(self, bgr: np.ndarray, n_colors: int = 5) -> np.ndarray:
        """Extract dominant colors using KMeans in LAB space."""
        # Convert to LAB
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        
        # Reshape and sample for speed
        pixels = lab.reshape(-1, 3)
        sample_size = min(10000, len(pixels))
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        samples = pixels[indices].astype(np.float32)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(samples)
        
        return kmeans.cluster_centers_
    
    def calculate_color_harmony(self, bgr: np.ndarray) -> float:
        """Score color harmony based on color relationships."""
        try:
            colors_lab = self.extract_dominant_colors(bgr, 5)
            
            # Convert LAB centers to approximate hues
            # L is 0-100, A and B are roughly -128 to 127
            hues = []
            for color in colors_lab:
                a, b = color[1] - 128, color[2] - 128
                hue = np.arctan2(b, a) * 180 / np.pi
                hues.append(hue % 360)
            
            # Check for complementary (180°) or analogous (30°) relationships
            harmonies = 0
            for i, h1 in enumerate(hues):
                for h2 in hues[i+1:]:
                    diff = abs(h1 - h2)
                    if diff > 180:
                        diff = 360 - diff
                    
                    # Complementary
                    if 150 < diff < 210:
                        harmonies += 1
                    # Analogous
                    elif diff < 45:
                        harmonies += 0.5
            
            score = self._clamp(harmonies / 5)
            return score
        except Exception:
            return 0.5
    
    def calculate_contrast_score(self, gray: np.ndarray) -> float:
        """Calculate RMS contrast and score optimal range."""
        pixels = gray.astype(np.float64).flatten()
        rms_contrast = np.std(pixels)
        
        # Optimal range: 40-80
        if rms_contrast < 20:
            score = rms_contrast / 20 * 0.5
        elif rms_contrast < 40:
            score = 0.5 + (rms_contrast - 20) / 40
        elif rms_contrast <= 80:
            score = 1.0
        else:
            score = 1.0 - (rms_contrast - 80) / 100
        
        return self._clamp(score)
    
    def calculate_visual_interest(self, gray: np.ndarray) -> float:
        """Higher std dev = more visual interest."""
        std = gray.std()
        # Normalize: 30-70 is good range
        score = self._clamp((std - 20) / 50)
        return score
    
    def score_aesthetic(self, bgr: np.ndarray, gray: np.ndarray) -> Tuple[float, dict]:
        """Calculate overall aesthetic score."""
        harmony = self.calculate_color_harmony(bgr)
        contrast = self.calculate_contrast_score(gray)
        interest = self.calculate_visual_interest(gray)
        
        score = (harmony + contrast + interest) / 3
        
        return self._clamp(score), {
            "color_harmony": harmony,
            "contrast": contrast,
            "visual_interest": interest
        }
    
    # =========================================================================
    # WALLPAPER SUITABILITY (10% weight)
    # =========================================================================
    
    def calculate_subject_centering(self, gray: np.ndarray) -> float:
        """Score higher if main subject is off-center (better for wallpapers)."""
        # Use saliency detection approximation via gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        h, w = gray.shape
        
        # Find center of mass of gradient
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        total = gradient_mag.sum() + 1e-6
        
        cx = (x_coords * gradient_mag).sum() / total
        cy = (y_coords * gradient_mag).sum() / total
        
        # Distance from center (normalized)
        center_x, center_y = w / 2, h / 2
        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Off-center is better for wallpapers
        score = self._clamp(dist / max_dist * 2)
        return score
    
    def calculate_brightness_suitability(self, gray: np.ndarray) -> float:
        """Score brightness for wallpaper use (40-70% optimal)."""
        mean_brightness = gray.mean() / 255
        
        # Optimal: 0.4-0.7
        if 0.4 <= mean_brightness <= 0.7:
            score = 1.0
        elif mean_brightness < 0.4:
            score = mean_brightness / 0.4
        else:
            score = (1.0 - mean_brightness) / 0.3
        
        return self._clamp(score)
    
    def calculate_texture_busyness(self, gray: np.ndarray) -> float:
        """Penalize very busy textures (hard to read icons over)."""
        # Use FFT to detect high-frequency content
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # Create mask for high frequencies (outer 30%)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        high_freq_mask = dist > 0.35 * min(h, w)
        
        high_freq_energy = magnitude[high_freq_mask].sum()
        total_energy = magnitude.sum() + 1e-6
        
        hf_ratio = high_freq_energy / total_energy
        
        # Lower HF ratio = less busy = better for wallpaper
        score = 1.0 - self._clamp(hf_ratio * 3)
        return score
    
    def score_suitability(self, gray: np.ndarray) -> Tuple[float, dict]:
        """Calculate wallpaper suitability score."""
        centering = self.calculate_subject_centering(gray)
        brightness = self.calculate_brightness_suitability(gray)
        busyness = self.calculate_texture_busyness(gray)
        
        score = (centering + brightness + busyness) / 3
        
        return self._clamp(score), {
            "subject_centering": centering,
            "brightness": brightness,
            "texture_busyness": busyness
        }
    
    # =========================================================================
    # MAIN SCORING FUNCTION
    # =========================================================================
    
    def calculate_quality_score(self, filepath) -> QualityScores:
        """
        Calculate comprehensive quality score for a wallpaper candidate.
        
        Args:
            filepath: Path to the image file.
        
        Returns:
            QualityScores dataclass with all component scores.
        """
        scores = QualityScores()
        
        try:
            bgr, gray = self._load_image_cv(filepath)
            
            # Visual Quality (40%)
            visual, visual_details = self.score_visual_quality(bgr, gray)
            scores.visual_quality = visual
            scores.sharpness = visual_details["sharpness"]
            scores.exposure = visual_details["exposure"]
            scores.color_accuracy = visual_details["color_accuracy"]
            scores.artifacts = visual_details["artifacts"]
            
            # Composition (30%)
            comp, comp_details = self.score_composition(gray)
            scores.composition = comp
            scores.rule_of_thirds = comp_details["rule_of_thirds"]
            scores.balance = comp_details["balance"]
            scores.depth = comp_details["depth"]
            scores.leading_lines = comp_details["leading_lines"]
            
            # Aesthetic Appeal (20%)
            aesthetic, aes_details = self.score_aesthetic(bgr, gray)
            scores.aesthetic_appeal = aesthetic
            scores.color_harmony = aes_details["color_harmony"]
            scores.contrast = aes_details["contrast"]
            scores.visual_interest = aes_details["visual_interest"]
            
            # Wallpaper Suitability (10%)
            suit, suit_details = self.score_suitability(gray)
            scores.wallpaper_suitability = suit
            scores.subject_centering = suit_details["subject_centering"]
            scores.brightness = suit_details["brightness"]
            scores.texture_busyness = suit_details["texture_busyness"]
            
            # Final weighted score
            scores.final_score = (
                scores.visual_quality * self.config.visual_weight +
                scores.composition * self.config.composition_weight +
                scores.aesthetic_appeal * self.config.aesthetic_weight +
                scores.wallpaper_suitability * self.config.suitability_weight
            )
            
        except Exception as e:
            logger.error(f"Quality scoring failed for {filepath}: {e}")
            scores.final_score = 0.0
        
        return scores
