#!/usr/bin/env python3
"""
ML-Based Quality Scorer using SigLIP

Replaces hand-coded heuristics with semantic understanding.
Evaluates images based on:
1. Aesthetic Appeal - Is it beautiful/pleasing?
2. Technical Quality - Is it sharp/clear?
3. Wallpaper Suitability - Would it work well as a wallpaper?

Uses SigLIP's text-image alignment to score against quality prompts.
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
        aesthetic_score: How beautiful/pleasing the image looks
        technical_score: Technical quality (sharpness, clarity)
        wallpaper_score: Suitability as a wallpaper
        confidence: How certain the model is (spread of scores)
        details: Raw similarities for debugging
    """
    final_score: float = 0.0
    aesthetic_score: float = 0.0
    technical_score: float = 0.0
    wallpaper_score: float = 0.0
    confidence: float = 0.0
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "final_score": self.final_score,
            "aesthetic_score": self.aesthetic_score,
            "technical_score": self.technical_score,
            "wallpaper_score": self.wallpaper_score,
            "confidence": self.confidence,
            "details": self.details
        }


@dataclass
class MLQualityConfig:
    """Configuration for ML quality scoring."""
    threshold: float = 0.55  # Approval threshold (0.5 = neutral)
    temperature: float = 0.07  # Softmax temperature (lower = more discriminative)
    
    # Default weights for each aspect (used for Reddit/community sources)
    aesthetic_weight: float = 0.40  # How beautiful it is
    technical_weight: float = 0.20  # Sharp, clear, no artifacts
    wallpaper_weight: float = 0.40  # Works well as wallpaper


# Source-specific weight configurations
SOURCE_WEIGHTS = {
    # Reddit: Community curated, need to check all aspects
    "reddit": {
        "aesthetic": 0.40,
        "technical": 0.20,
        "wallpaper": 0.40,
    },
    # Unsplash: Professional stock photos, already high quality
    # Focus on wallpaper suitability since quality is guaranteed
    "unsplash": {
        "aesthetic": 0.15,  # Already curated
        "technical": 0.05,  # Professionals handle this
        "wallpaper": 0.80,  # Main question: will it work as wallpaper?
    },
    # Pexels: Similar to Unsplash, professional stock
    "pexels": {
        "aesthetic": 0.15,
        "technical": 0.05,
        "wallpaper": 0.80,
    },
    # Default fallback
    "default": {
        "aesthetic": 0.35,
        "technical": 0.15,
        "wallpaper": 0.50,
    }
}


class MLQualityScorer:
    """
    ML-based quality scorer using SigLIP embeddings.
    
    Philosophy: Reddit/Unsplash users share wallpapers they love and feel 
    proud of. Our job is to verify:
    1. It's aesthetically pleasing (beautiful, well-composed)
    2. It's technically good (not blurry, no artifacts)
    3. It works as a wallpaper (right vibe, usable)
    
    We're NOT trying to be harsh critics - we trust the community curation.
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
        self._model = None
        self._processor = None
        self._device = None
    
    def _ensure_model_loaded(self):
        """Lazy load SigLIP model."""
        if self._model is not None:
            return True
            
        if self._extractor is not None:
            # Use extractor's model
            model, processor = self._extractor._load_siglip()
            if model is not None:
                self._model = model
                self._processor = processor
                self._device = self._extractor.device
                return True
        
        # Try to load directly
        try:
            from transformers import AutoProcessor, AutoModel
            import torch
            
            model_name = "google/siglip-large-patch16-384"
            self._processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            self._model = AutoModel.from_pretrained(model_name)
            
            # Detect device
            if torch.cuda.is_available():
                self._device = "cuda"
                self._model = self._model.cuda()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
                self._model = self._model.to("mps")
            else:
                self._device = "cpu"
            
            self._model.eval()
            logger.info("MLQualityScorer: Loaded SigLIP-Large")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SigLIP model: {e}")
            return False
    
    def _compute_similarity(
        self, 
        image_features, 
        positive_prompt: str, 
        negative_prompt: str
    ) -> Tuple[float, float, float]:
        """
        Compute similarity to positive/negative prompts and return score.
        
        Returns:
            Tuple of (score, positive_sim, negative_sim)
        """
        import torch
        import torch.nn.functional as F
        
        # Encode prompts
        text_inputs = self._processor(
            text=[positive_prompt, negative_prompt],
            padding=True,
            return_tensors="pt"
        )
        
        if self._device == "cuda":
            text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
        elif self._device == "mps":
            text_inputs = {k: v.to("mps") for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self._model.get_text_features(**text_inputs)
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
        Score an image's quality using SigLIP semantic understanding.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            MLQualityScore with all component scores
        """
        result = MLQualityScore()
        
        if not self._ensure_model_loaded():
            # Model failed to load - give benefit of doubt
            logger.warning("ML model unavailable, using default passing score")
            result.final_score = 0.75
            result.confidence = 0.0
            return result
        
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
            
            # Load and process image
            img = Image.open(filepath)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((384, 384), Image.Resampling.LANCZOS)
            
            image_inputs = self._processor(images=img, return_tensors="pt")
            
            if self._device == "cuda":
                image_inputs = {k: v.cuda() for k, v in image_inputs.items()}
            elif self._device == "mps":
                image_inputs = {k: v.to("mps") for k, v in image_inputs.items()}
            
            with torch.no_grad():
                image_features = self._model.get_image_features(**image_inputs)
                image_features = F.normalize(image_features, dim=-1)
            
            # 1. AESTHETIC SCORE - Is it beautiful?
            aesthetic_score, aes_pos, aes_neg = self._compute_similarity(
                image_features,
                positive_prompt="a beautiful stunning gorgeous photograph with amazing composition",
                negative_prompt="an ugly poorly composed unappealing photograph"
            )
            result.aesthetic_score = aesthetic_score
            
            # 2. TECHNICAL SCORE - Is it sharp/clear?
            technical_score, tech_pos, tech_neg = self._compute_similarity(
                image_features,
                positive_prompt="a sharp clear high resolution photograph",
                negative_prompt="a blurry noisy low quality pixelated image"
            )
            result.technical_score = technical_score
            
            # 3. WALLPAPER SCORE - Would it work as wallpaper?
            wallpaper_score, wall_pos, wall_neg = self._compute_similarity(
                image_features,
                positive_prompt="a beautiful desktop wallpaper background image",
                negative_prompt="a screenshot meme text overlay not a wallpaper"
            )
            result.wallpaper_score = wallpaper_score
            
            # Compute weighted final score
            result.final_score = (
                aesthetic_score * self.config.aesthetic_weight +
                technical_score * self.config.technical_weight +
                wallpaper_score * self.config.wallpaper_weight
            )
            
            # Confidence = how spread out the scores are (low variance = confident)
            scores = [aesthetic_score, technical_score, wallpaper_score]
            result.confidence = 1.0 - np.std(scores)
            
            # Store details for debugging
            result.details = {
                "aesthetic": {"score": aesthetic_score, "pos": aes_pos, "neg": aes_neg},
                "technical": {"score": technical_score, "pos": tech_pos, "neg": tech_neg},
                "wallpaper": {"score": wallpaper_score, "pos": wall_pos, "neg": wall_neg}
            }
            
            logger.info(
                f"Quality: aesthetic={aesthetic_score:.3f}, "
                f"technical={technical_score:.3f}, "
                f"wallpaper={wallpaper_score:.3f} -> "
                f"FINAL={result.final_score:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            # Give benefit of doubt on error
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
        
        if not self._ensure_model_loaded():
            result.final_score = 0.75
            return result, None
        
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
            
            # Load and process image
            img = Image.open(filepath)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((384, 384), Image.Resampling.LANCZOS)
            
            image_inputs = self._processor(images=img, return_tensors="pt")
            
            if self._device == "cuda":
                image_inputs = {k: v.cuda() for k, v in image_inputs.items()}
            elif self._device == "mps":
                image_inputs = {k: v.to("mps") for k, v in image_inputs.items()}
            
            with torch.no_grad():
                image_features = self._model.get_image_features(**image_inputs)
                image_features = F.normalize(image_features, dim=-1)
                
                # Save embedding
                embedding = image_features.cpu().numpy()[0].astype(np.float16)
            
            # Score all aspects
            result.aesthetic_score, _, _ = self._compute_similarity(
                image_features,
                "a beautiful stunning gorgeous photograph with amazing composition",
                "an ugly poorly composed unappealing photograph"
            )
            
            result.technical_score, _, _ = self._compute_similarity(
                image_features,
                "a sharp clear high resolution photograph",
                "a blurry noisy low quality pixelated image"
            )
            
            result.wallpaper_score, _, _ = self._compute_similarity(
                image_features,
                "a beautiful desktop wallpaper background image",
                "a screenshot meme text overlay not a wallpaper"
            )
            
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
            
            logger.info(
                f"Quality: aes={result.aesthetic_score:.2f}, "
                f"tech={result.technical_score:.2f}, "
                f"wall={result.wallpaper_score:.2f} -> "
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
        - Reddit: Community curated, need balanced checks
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
        
        if not self._ensure_model_loaded():
            result.final_score = 0.75
            return result, None
        
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
            
            # Load and process image
            img = Image.open(filepath)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((384, 384), Image.Resampling.LANCZOS)
            
            image_inputs = self._processor(images=img, return_tensors="pt")
            
            if self._device == "cuda":
                image_inputs = {k: v.cuda() for k, v in image_inputs.items()}
            elif self._device == "mps":
                image_inputs = {k: v.to("mps") for k, v in image_inputs.items()}
            
            with torch.no_grad():
                image_features = self._model.get_image_features(**image_inputs)
                image_features = F.normalize(image_features, dim=-1)
                embedding = image_features.cpu().numpy()[0].astype(np.float16)
            
            # Score all aspects
            result.aesthetic_score, _, _ = self._compute_similarity(
                image_features,
                "a beautiful stunning gorgeous photograph with amazing composition",
                "an ugly poorly composed unappealing photograph"
            )
            
            result.technical_score, _, _ = self._compute_similarity(
                image_features,
                "a sharp clear high resolution photograph",
                "a blurry noisy low quality pixelated image"
            )
            
            result.wallpaper_score, _, _ = self._compute_similarity(
                image_features,
                "a beautiful desktop wallpaper background image",
                "a screenshot meme text overlay not a wallpaper"
            )
            
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
            
            result.details = {"source": source, "weights": weights}
            
            logger.info(
                f"Quality [{source}]: aes={result.aesthetic_score:.2f}×{weights['aesthetic']}, "
                f"tech={result.technical_score:.2f}×{weights['technical']}, "
                f"wall={result.wallpaper_score:.2f}×{weights['wallpaper']} -> "
                f"FINAL={result.final_score:.3f}"
            )
            
            return result, embedding
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            result.final_score = 0.75
            return result, None

