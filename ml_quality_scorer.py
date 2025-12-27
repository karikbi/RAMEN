#!/usr/bin/env python3
"""
ML-Based Quality Scorer using Aesthetic Predictor V2.5 + SigLIP

Hybrid scoring approach:
1. Aesthetic Appeal - Aesthetic Predictor V2.5 (SigLIP-based, 1-10 scale)
2. Technical Quality - SigLIP text-image alignment (sharp vs blurry)
3. Wallpaper Suitability - SigLIP text-image alignment (wallpaper vs not)

Aesthetic Predictor V2.5 is trained specifically on SigLIP embeddings,
providing reliable aesthetic quality scores on a 1-10 scale. Scores of
5.5+ indicate good aesthetics, with 6.5+ being premium quality.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("wallpaper_curator")


@dataclass
class MLQualityScore:
    """
    Quality assessment result from ML scoring.
    
    Attributes:
        final_score: Overall quality score (0-1), main decision value
        aesthetic_score: Aesthetic V2.5 score (normalized 0-1 from 1-10 scale)
        technical_score: SigLIP-based technical quality check
        wallpaper_score: SigLIP-based wallpaper suitability check
        raw_aesthetic: Original V2.5 score (1-10 scale)
        quality_tier: Quality tier (premium/standard/acceptable/low)
        confidence: How certain the model is
        details: Raw data for debugging
    """
    final_score: float = 0.0
    aesthetic_score: float = 0.0
    technical_score: float = 0.0
    wallpaper_score: float = 0.0
    raw_aesthetic: float = 0.0  # Original 1-10 V2.5 score
    quality_tier: str = "low"  # premium/standard/acceptable/low
    confidence: float = 0.0
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "final_score": self.final_score,
            "aesthetic_score": self.aesthetic_score,
            "technical_score": self.technical_score,
            "wallpaper_score": self.wallpaper_score,
            "raw_aesthetic": self.raw_aesthetic,
            "quality_tier": self.quality_tier,
            "confidence": self.confidence,
            "details": self.details
        }


@dataclass
class MLQualityConfig:
    """Configuration for ML quality scoring."""
    # Threshold for approval (normalized 0-1 scale)
    # V2.5 scores 1-10, normalized to 0-1. Threshold 0.61 = 5.5/10 (good quality)
    # Quality tiers: 6.5+ Premium (0.72), 5.5-6.5 Standard (0.61), 4.0-5.5 Acceptable (0.44), <4.0 Low
    threshold: float = 0.61
    
    # SigLIP temperature for softmax (for technical/wallpaper checks)
    temperature: float = 0.07
    
    # Component weights - aesthetic is the primary driver
    aesthetic_weight: float = 0.60  # Aesthetic Predictor V2.5 (most reliable)
    technical_weight: float = 0.15  # SigLIP technical check
    wallpaper_weight: float = 0.25  # SigLIP wallpaper suitability


# Source-specific weight configurations
# Professional sources need less aesthetic checking (already curated)
SOURCE_WEIGHTS = {
    # Reddit: Community curated, needs aesthetic verification
    "reddit": {
        "aesthetic": 0.60,
        "technical": 0.15,
        "wallpaper": 0.25,
    },
    # Unsplash: Professional stock, already aesthetic, focus on suitability
    "unsplash": {
        "aesthetic": 0.40,
        "technical": 0.10,
        "wallpaper": 0.50,
    },
    # Pexels: Similar to Unsplash
    "pexels": {
        "aesthetic": 0.40,
        "technical": 0.10,
        "wallpaper": 0.50,
    },
    # Default fallback
    "default": {
        "aesthetic": 0.55,
        "technical": 0.15,
        "wallpaper": 0.30,
    }
}


class MLQualityScorer:
    """
    Hybrid ML-based quality scorer using:
    - Aesthetic Predictor V2.5 for aesthetic quality (SigLIP-based, 1-10 scale)
    - SigLIP for technical quality and wallpaper suitability checks
    
    This approach uses the correct vector space:
    - V2.5 is trained specifically on SigLIP embeddings (not CLIP)
    - SigLIP's text-image alignment handles specific yes/no checks
    """
    
    def __init__(
        self, 
        embedding_extractor=None,
        config: Optional[MLQualityConfig] = None
    ):
        """
        Initialize ML quality scorer.
        
        Args:
            embedding_extractor: EmbeddingExtractor instance (for SigLIP access)
            config: Quality scoring configuration
        """
        self.config = config or MLQualityConfig()
        self._extractor = embedding_extractor
        
        # Aesthetic Predictor V2.5 (SigLIP-based)
        self._aesthetic_model = None
        self._aesthetic_processor = None
        
        # SigLIP (for technical/wallpaper checks)
        self._siglip_model = None
        self._siglip_processor = None
        self._device = None
    
    def _ensure_aesthetic_loaded(self) -> bool:
        """Lazy load Aesthetic Predictor V2.5 (SigLIP-based)."""
        if self._aesthetic_model is not None:
            return True
        
        try:
            from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
            import torch
            
            # Load V2.5 model specifically trained for SigLIP embeddings
            self._aesthetic_model, self._aesthetic_processor = convert_v2_5_from_siglip(
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Move to appropriate device
            if torch.cuda.is_available():
                self._aesthetic_model = self._aesthetic_model.cuda()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._aesthetic_model = self._aesthetic_model.to("mps")
            
            self._aesthetic_model.eval()
            logger.info("Loaded Aesthetic Predictor V2.5 (SigLIP-based)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Aesthetic Predictor V2.5: {e}")
            return False
    
    def _ensure_siglip_loaded(self) -> bool:
        """Lazy load SigLIP model for technical/wallpaper checks."""
        if self._siglip_model is not None:
            return True
            
        if self._extractor is not None:
            # Use extractor's model
            model, processor = self._extractor._load_siglip()
            if model is not None:
                self._siglip_model = model
                self._siglip_processor = processor
                self._device = self._extractor.device
                return True
        
        # Try to load directly
        try:
            from transformers import AutoProcessor, AutoModel
            import torch
            
            # SigLIP 2: Better localization, multilingual, officially stable
            model_name = "google/siglip2-large-patch16-384"
            self._siglip_processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            self._siglip_model = AutoModel.from_pretrained(model_name)
            
            # Detect device
            if torch.cuda.is_available():
                self._device = "cuda"
                self._siglip_model = self._siglip_model.cuda()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
                self._siglip_model = self._siglip_model.to("mps")
            else:
                self._device = "cpu"
            
            self._siglip_model.eval()
            logger.info("Loaded SigLIP 2 Large")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SigLIP 2 model: {e}")
            return False
    
    def _score_aesthetic_v25(self, image) -> Tuple[float, float]:
        """
        Score aesthetic quality using Aesthetic Predictor V2.5 (SigLIP-based).
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (normalized_score 0-1, raw_score 1-10)
        """
        import torch
        
        # Process image using V2.5 processor
        pixel_values = self._aesthetic_processor(images=image, return_tensors="pt").pixel_values
        
        # Move to same device as model
        device = next(self._aesthetic_model.parameters()).device
        pixel_values = pixel_values.to(device)
        
        with torch.no_grad():
            # Get aesthetic score (1-10 scale)
            raw_score = self._aesthetic_model(pixel_values).logits.squeeze().item()
        
        # Clamp to valid range and normalize to 0-1
        raw_score = max(1.0, min(10.0, raw_score))
        normalized = (raw_score - 1.0) / 9.0  # Map 1-10 to 0-1
        
        return normalized, raw_score
    
    def _compute_siglip_similarity(
        self, 
        image_features, 
        positive_prompt: str, 
        negative_prompt: str
    ) -> Tuple[float, float, float]:
        """
        Compute SigLIP similarity for binary yes/no checks.
        
        Returns softmax probability that image matches positive prompt.
        """
        import torch
        import torch.nn.functional as F
        
        # Encode prompts
        text_inputs = self._siglip_processor(
            text=[positive_prompt, negative_prompt],
            padding=True,
            return_tensors="pt"
        )
        
        if self._device == "cuda":
            text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
        elif self._device == "mps":
            text_inputs = {k: v.to("mps") for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self._siglip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarities
            similarities = torch.matmul(image_features, text_features.T)[0]
            pos_sim = similarities[0].item()
            neg_sim = similarities[1].item()
            
            # Softmax to get probability
            logits = similarities / self.config.temperature
            probs = F.softmax(logits, dim=0)
            score = probs[0].item()
        
        return score, pos_sim, neg_sim
    
    def score(self, filepath) -> MLQualityScore:
        """
        Score an image's quality using Aesthetic V2.5 + SigLIP approach.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            MLQualityScore with all component scores
        """
        result = MLQualityScore()
        
        # Load both models
        aesthetic_loaded = self._ensure_aesthetic_loaded()
        siglip_loaded = self._ensure_siglip_loaded()
        
        if not aesthetic_loaded and not siglip_loaded:
            logger.warning("No ML models available, using default passing score")
            result.final_score = 0.61  # Default to threshold (5.5/10 normalized)
            result.raw_aesthetic = 5.5
            result.quality_tier = "standard"
            result.confidence = 0.0
            return result
        
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
            
            # Load image
            img = Image.open(filepath)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # 1. AESTHETIC SCORE (V2.5 - primary signal)
            if aesthetic_loaded:
                result.aesthetic_score, result.raw_aesthetic = self._score_aesthetic_v25(img)
            else:
                # Fallback if V2.5 fails (5.5/10 = 0.61 normalized)
                result.aesthetic_score = 0.61
                result.raw_aesthetic = 5.5
            
            # 2. TECHNICAL SCORE (SigLIP - is it sharp/clear?)
            # 3. WALLPAPER SCORE (SigLIP - is it suitable as wallpaper?)
            if siglip_loaded:
                # Resize for SigLIP
                img_siglip = img.resize((384, 384), Image.Resampling.LANCZOS)
                image_inputs = self._siglip_processor(images=img_siglip, return_tensors="pt")
                
                if self._device == "cuda":
                    image_inputs = {k: v.cuda() for k, v in image_inputs.items()}
                elif self._device == "mps":
                    image_inputs = {k: v.to("mps") for k, v in image_inputs.items()}
                
                with torch.no_grad():
                    image_features = self._siglip_model.get_image_features(**image_inputs)
                    image_features = F.normalize(image_features, dim=-1)
                
                # Technical quality check
                result.technical_score, _, _ = self._compute_siglip_similarity(
                    image_features,
                    "a sharp clear high resolution photograph",
                    "a blurry noisy low quality pixelated image"
                )
                
                # Wallpaper suitability check
                result.wallpaper_score, _, _ = self._compute_siglip_similarity(
                    image_features,
                    "a beautiful desktop wallpaper background image",
                    "a screenshot meme text overlay not a wallpaper"
                )
            else:
                # Fallback if SigLIP fails
                result.technical_score = 0.5
                result.wallpaper_score = 0.5
            
            # Compute weighted final score
            result.final_score = (
                result.aesthetic_score * self.config.aesthetic_weight +
                result.technical_score * self.config.technical_weight +
                result.wallpaper_score * self.config.wallpaper_weight
            )
            
            # Confidence based on score spread
            scores = [result.aesthetic_score, result.technical_score, result.wallpaper_score]
            result.confidence = 1.0 - np.std(scores)
            
            # Set quality tier based on raw aesthetic score
            result.quality_tier = get_quality_tier(result.raw_aesthetic)
            
            result.details = {
                "aesthetic_loaded": aesthetic_loaded,
                "siglip_loaded": siglip_loaded
            }
            
            logger.info(
                f"Quality: V2.5={result.raw_aesthetic:.1f}/10 ({result.quality_tier}), "
                f"tech={result.technical_score:.2f}, wall={result.wallpaper_score:.2f} -> "
                f"FINAL={result.final_score:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            result.final_score = 0.75
            result.confidence = 0.0
            return result
    
    def score_with_embedding(self, filepath) -> Tuple[MLQualityScore, Optional[np.ndarray]]:
        """
        Score quality AND return SigLIP embedding.
        
        More efficient than scoring then extracting separately.
        
        Returns:
            Tuple of (MLQualityScore, embedding or None)
        """
        result = MLQualityScore()
        embedding = None
        
        aesthetic_loaded = self._ensure_aesthetic_loaded()
        siglip_loaded = self._ensure_siglip_loaded()
        
        if not aesthetic_loaded and not siglip_loaded:
            result.final_score = 0.61  # Default to threshold (5.5/10 normalized)
            result.raw_aesthetic = 5.5
            result.quality_tier = "standard"
            return result, None
        
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
            
            img = Image.open(filepath)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Aesthetic V2.5 score
            if aesthetic_loaded:
                result.aesthetic_score, result.raw_aesthetic = self._score_aesthetic_v25(img)
            else:
                result.aesthetic_score = 0.61
                result.raw_aesthetic = 5.5
            
            # SigLIP for technical, wallpaper, and embedding
            if siglip_loaded:
                img_siglip = img.resize((384, 384), Image.Resampling.LANCZOS)
                image_inputs = self._siglip_processor(images=img_siglip, return_tensors="pt")
                
                if self._device == "cuda":
                    image_inputs = {k: v.cuda() for k, v in image_inputs.items()}
                elif self._device == "mps":
                    image_inputs = {k: v.to("mps") for k, v in image_inputs.items()}
                
                with torch.no_grad():
                    image_features = self._siglip_model.get_image_features(**image_inputs)
                    image_features = F.normalize(image_features, dim=-1)
                    
                    # Save embedding
                    embedding = image_features.cpu().numpy()[0].astype(np.float16)
                
                result.technical_score, _, _ = self._compute_siglip_similarity(
                    image_features,
                    "a sharp clear high resolution photograph",
                    "a blurry noisy low quality pixelated image"
                )
                
                result.wallpaper_score, _, _ = self._compute_siglip_similarity(
                    image_features,
                    "a beautiful desktop wallpaper background image",
                    "a screenshot meme text overlay not a wallpaper"
                )
            else:
                result.technical_score = 0.5
                result.wallpaper_score = 0.5
            
            result.final_score = (
                result.aesthetic_score * self.config.aesthetic_weight +
                result.technical_score * self.config.technical_weight +
                result.wallpaper_score * self.config.wallpaper_weight
            )
            
            result.confidence = 1.0 - np.std([
                result.aesthetic_score, 
                result.technical_score, 
                result.wallpaper_score
            ])
            
            # Set quality tier
            result.quality_tier = get_quality_tier(result.raw_aesthetic)
            
            logger.info(
                f"Quality: V2.5={result.raw_aesthetic:.1f}/10 ({result.quality_tier}), "
                f"tech={result.technical_score:.2f}, wall={result.wallpaper_score:.2f} -> "
                f"FINAL={result.final_score:.3f}"
            )
            
            return result, embedding
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            result.final_score = 0.75
            return result, None
    
    def score_for_source(
        self, 
        filepath, 
        source: str
    ) -> Tuple[MLQualityScore, Optional[np.ndarray]]:
        """
        Score quality with source-specific weights.
        
        Different sources have different quality guarantees:
        - Reddit: Community curated, aesthetic check important
        - Unsplash/Pexels: Professional stock, focus on wallpaper suitability
        
        Args:
            filepath: Path to image file
            source: Source name ("reddit", "unsplash", "pexels")
            
        Returns:
            Tuple of (MLQualityScore, embedding or None)
        """
        result = MLQualityScore()
        embedding = None
        
        # Get source-specific weights
        source_key = source.lower() if source.lower() in SOURCE_WEIGHTS else "default"
        weights = SOURCE_WEIGHTS[source_key]
        
        aesthetic_loaded = self._ensure_aesthetic_loaded()
        siglip_loaded = self._ensure_siglip_loaded()
        
        if not aesthetic_loaded and not siglip_loaded:
            result.final_score = 0.61  # Default to threshold (5.5/10 normalized)
            result.raw_aesthetic = 5.5
            result.quality_tier = "standard"
            return result, None
        
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
            
            img = Image.open(filepath)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Aesthetic V2.5 score (primary signal)
            if aesthetic_loaded:
                result.aesthetic_score, result.raw_aesthetic = self._score_aesthetic_v25(img)
            else:
                result.aesthetic_score = 0.61
                result.raw_aesthetic = 5.5
            
            # SigLIP for technical and wallpaper checks
            if siglip_loaded:
                img_siglip = img.resize((384, 384), Image.Resampling.LANCZOS)
                image_inputs = self._siglip_processor(images=img_siglip, return_tensors="pt")
                
                if self._device == "cuda":
                    image_inputs = {k: v.cuda() for k, v in image_inputs.items()}
                elif self._device == "mps":
                    image_inputs = {k: v.to("mps") for k, v in image_inputs.items()}
                
                with torch.no_grad():
                    image_features = self._siglip_model.get_image_features(**image_inputs)
                    image_features = F.normalize(image_features, dim=-1)
                    embedding = image_features.cpu().numpy()[0].astype(np.float16)
                
                result.technical_score, _, _ = self._compute_siglip_similarity(
                    image_features,
                    "a sharp clear high resolution photograph",
                    "a blurry noisy low quality pixelated image"
                )
                
                result.wallpaper_score, _, _ = self._compute_siglip_similarity(
                    image_features,
                    "a beautiful desktop wallpaper background image",
                    "a screenshot meme text overlay not a wallpaper"
                )
            else:
                result.technical_score = 0.5
                result.wallpaper_score = 0.5
            
            # Apply SOURCE-SPECIFIC weights
            result.final_score = (
                result.aesthetic_score * weights["aesthetic"] +
                result.technical_score * weights["technical"] +
                result.wallpaper_score * weights["wallpaper"]
            )
            
            result.confidence = 1.0 - np.std([
                result.aesthetic_score, 
                result.technical_score, 
                result.wallpaper_score
            ])
            
            # Set quality tier
            result.quality_tier = get_quality_tier(result.raw_aesthetic)
            
            result.details = {"source": source, "weights": weights}
            
            logger.info(
                f"Quality [{source}]: V2.5={result.raw_aesthetic:.1f}/10 ({result.quality_tier})×{weights['aesthetic']}, "
                f"tech={result.technical_score:.2f}×{weights['technical']}, "
                f"wall={result.wallpaper_score:.2f}×{weights['wallpaper']} -> "
                f"FINAL={result.final_score:.3f}"
            )
            
            return result, embedding
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            result.final_score = 0.75
            return result, None


def get_quality_tier(raw_aesthetic: float) -> str:
    """
    Get quality tier based on Aesthetic Predictor V2.5 score.
    
    Aesthetic Predictor V2.5 Scale (1-10):
    - 6.5-10: Premium (feature in "Best", show first)
    - 5.5-6.5: Standard (general catalog)
    - 4.0-5.5: Acceptable (lower priority)
    - <4.0: Low (filter out or "All" category only)
    
    Args:
        raw_aesthetic: V2.5 aesthetic score (1-10 scale)
        
    Returns:
        Quality tier string: "premium", "standard", "acceptable", or "low"
    """
    if raw_aesthetic >= 6.5:
        return "premium"
    elif raw_aesthetic >= 5.5:
        return "standard"
    elif raw_aesthetic >= 4.0:
        return "acceptable"
    else:
        return "low"
