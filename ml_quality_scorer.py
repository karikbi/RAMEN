#!/usr/bin/env python3
"""
ML-Based Quality Scorer using LAION Aesthetic Predictor + SigLIP

Hybrid scoring approach:
1. Aesthetic Appeal - LAION Aesthetic Predictor V2 (trained on human ratings, 1-10 scale)
2. Technical Quality - SigLIP text-image alignment (sharp vs blurry)
3. Wallpaper Suitability - SigLIP text-image alignment (wallpaper vs not)

LAION Aesthetic Predictor was trained on millions of human aesthetic ratings from
the AVA dataset, providing reliable aesthetic quality scores on a 1-10 scale.
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
        aesthetic_score: LAION aesthetic score (normalized 0-1 from 1-10 scale)
        technical_score: SigLIP-based technical quality check
        wallpaper_score: SigLIP-based wallpaper suitability check
        raw_aesthetic: Original LAION score (1-10 scale)
        confidence: How certain the model is
        details: Raw data for debugging
    """
    final_score: float = 0.0
    aesthetic_score: float = 0.0
    technical_score: float = 0.0
    wallpaper_score: float = 0.0
    raw_aesthetic: float = 0.0  # Original 1-10 LAION score
    confidence: float = 0.0
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "final_score": self.final_score,
            "aesthetic_score": self.aesthetic_score,
            "technical_score": self.technical_score,
            "wallpaper_score": self.wallpaper_score,
            "raw_aesthetic": self.raw_aesthetic,
            "confidence": self.confidence,
            "details": self.details
        }


@dataclass
class MLQualityConfig:
    """Configuration for ML quality scoring."""
    # Threshold for approval (0.5 = 5/10 on LAION scale, reasonable minimum)
    threshold: float = 0.50
    
    # SigLIP temperature for softmax (only used for technical/wallpaper checks)
    temperature: float = 0.07
    
    # Component weights - aesthetic is now the primary driver
    aesthetic_weight: float = 0.60  # LAION aesthetic (most reliable)
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
    - LAION Aesthetic Predictor V2 for aesthetic quality (1-10 human-rated scale)
    - SigLIP for technical quality and wallpaper suitability checks
    
    This approach combines the best of both worlds:
    - LAION's human-trained aesthetic model provides reliable 1-10 scores
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
        
        # LAION Aesthetic Predictor
        self._laion_model = None
        self._laion_processor = None
        
        # SigLIP (for technical/wallpaper checks)
        self._siglip_model = None
        self._siglip_processor = None
        self._device = None
    
    def _ensure_laion_loaded(self) -> bool:
        """Lazy load LAION Aesthetic Predictor V2."""
        if self._laion_model is not None:
            return True
        
        try:
            from aesthetics_predictor import AestheticsPredictorV2Linear
            from transformers import CLIPProcessor
            
            model_name = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
            
            self._laion_model = AestheticsPredictorV2Linear.from_pretrained(model_name)
            self._laion_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
            # Move to appropriate device
            import torch
            if torch.cuda.is_available():
                self._laion_model = self._laion_model.cuda()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._laion_model = self._laion_model.to("mps")
            
            self._laion_model.eval()
            logger.info("Loaded LAION Aesthetic Predictor V2")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LAION Aesthetic Predictor: {e}")
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
            
            model_name = "google/siglip-large-patch16-384"
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
            logger.info("Loaded SigLIP-Large")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SigLIP model: {e}")
            return False
    
    def _score_aesthetic_laion(self, image) -> Tuple[float, float]:
        """
        Score aesthetic quality using LAION Aesthetic Predictor.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (normalized_score 0-1, raw_score 1-10)
        """
        import torch
        
        inputs = self._laion_processor(images=image, return_tensors="pt")
        
        # Move to same device as model
        device = next(self._laion_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get aesthetic score (1-10 scale)
            raw_score = self._laion_model(**inputs).logits.item()
        
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
        Score an image's quality using hybrid LAION + SigLIP approach.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            MLQualityScore with all component scores
        """
        result = MLQualityScore()
        
        # Load both models
        laion_loaded = self._ensure_laion_loaded()
        siglip_loaded = self._ensure_siglip_loaded()
        
        if not laion_loaded and not siglip_loaded:
            logger.warning("No ML models available, using default passing score")
            result.final_score = 0.75
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
            
            # 1. AESTHETIC SCORE (LAION - primary signal)
            if laion_loaded:
                result.aesthetic_score, result.raw_aesthetic = self._score_aesthetic_laion(img)
            else:
                # Fallback if LAION fails
                result.aesthetic_score = 0.5
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
            
            result.details = {
                "laion_loaded": laion_loaded,
                "siglip_loaded": siglip_loaded
            }
            
            logger.info(
                f"Quality: LAION_aes={result.raw_aesthetic:.1f}/10 ({result.aesthetic_score:.2f}), "
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
        
        laion_loaded = self._ensure_laion_loaded()
        siglip_loaded = self._ensure_siglip_loaded()
        
        if not laion_loaded and not siglip_loaded:
            result.final_score = 0.75
            return result, None
        
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
            
            img = Image.open(filepath)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # LAION aesthetic score
            if laion_loaded:
                result.aesthetic_score, result.raw_aesthetic = self._score_aesthetic_laion(img)
            else:
                result.aesthetic_score = 0.5
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
            
            logger.info(
                f"Quality: LAION={result.raw_aesthetic:.1f}/10, "
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
        
        laion_loaded = self._ensure_laion_loaded()
        siglip_loaded = self._ensure_siglip_loaded()
        
        if not laion_loaded and not siglip_loaded:
            result.final_score = 0.75
            return result, None
        
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
            
            img = Image.open(filepath)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # LAION aesthetic score (primary signal)
            if laion_loaded:
                result.aesthetic_score, result.raw_aesthetic = self._score_aesthetic_laion(img)
            else:
                result.aesthetic_score = 0.5
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
            
            result.details = {"source": source, "weights": weights}
            
            logger.info(
                f"Quality [{source}]: LAION={result.raw_aesthetic:.1f}/10×{weights['aesthetic']}, "
                f"tech={result.technical_score:.2f}×{weights['technical']}, "
                f"wall={result.wallpaper_score:.2f}×{weights['wallpaper']} -> "
                f"FINAL={result.final_score:.3f}"
            )
            
            return result, embedding
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            result.final_score = 0.75
            return result, None

