#!/usr/bin/env python3
"""
Wallpaper Metadata Extractor - Training-Free ML-Based Extraction

Leverages existing embedding models to extract rich metadata:
- SigLIP 2: Zero-shot classification for categories, moods, styles
- DINOv3: Attention-based focal point detection
- Pixel analysis: Colors, brightness, contrast, symmetry

No training required - all methods use text prompts or classical CV.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

logger = logging.getLogger("wallpaper_curator")


# =============================================================================
# VOCABULARY DEFINITIONS
# =============================================================================

# =============================================================================
# VOCABULARY DEFINITIONS (Contextual Prompts for SigLIP)
# =============================================================================

CATEGORY_VOCABULARY = {
    # Natural & Landscapes
    "nature": "a photo of a natural landscape with mountains, forests, oceans, or rivers",
    "space": "a photo of outer space showing galaxies, stars, planets, or nebulae",
    "animals": "a photo featuring animals, wildlife, birds, or creatures in their habitat",
    "underwater": "an underwater photo of the ocean depths, marine life, or coral reefs",
    "aerial": "an aerial view photo taken from a drone or bird's eye perspective",
    
    # Urban & Architecture
    "urban": "a photo of a city skyline, urban street, or metropolitan architecture",
    "architecture": "a photo focusing on modern building design, structures, or geometry",
    "automotive": "a photo of cars, motorcycles, or vehicles in a cinematic style",
    
    # Art & Design
    "abstract": "an abstract image with geometric patterns, shapes, or non-representational designs",
    "minimalist": "a minimalist image with simple composition and negative space",
    "art": "a piece of digital art, illustration, painting, or concept art",
    "anime": "an anime or manga style illustration with japanese animation aesthetics",
    "vintage": "an image with a vintage, retro, or old-fashioned aesthetic",
    
    # Technology & Gaming
    "technology": "a futuristic image with cyberpunk, neon, or sci-fi technology themes",
    "gaming": "a video game screenshot or gaming-related digital artwork",
    
    # Themes & Moods
    "dark": "a dark, low-key image suitable for AMOLED screens with black backgrounds",
    "colorful": "a vibrant, highly saturated image with bright rainbow colors",
    
    # Seasonal & Weather
    "seasonal": "a seasonal image showing spring, summer, autumn leaves, or winter snow",
    "weather": "a photo featuring weather elements like rain, storms, lightning, or clouds",
}

MOOD_VOCABULARY = {
    "calm": "calm peaceful serene wallpaper scene",
    "dramatic": "dramatic intense moody wallpaper with strong contrast",
    "mysterious": "mysterious dark enigmatic wallpaper atmosphere",
    "energetic": "vibrant energetic dynamic colorful wallpaper",
    "romantic": "romantic soft dreamy wallpaper with warm tones",
    "melancholic": "melancholic sad lonely wallpaper with muted colors",
    "inspiring": "inspiring uplifting hopeful wallpaper scene",
    "cozy": "cozy warm comfortable wallpaper atmosphere",
    "epic": "epic grand majestic wallpaper landscape",
    "surreal": "surreal dreamlike fantastical wallpaper scene",
}

STYLE_VOCABULARY = {
    "nord": "cool blue gray nordic wallpaper palette",
    "gruvbox": "retro orange brown warm wallpaper palette",
    "dracula": "dark purple vampire themed wallpaper",
    "monokai": "dark wallpaper with vibrant accent colors",
    "solarized": "warm yellow blue balanced wallpaper palette",
    "catppuccin": "pastel soft colors wallpaper palette",
    "cyberpunk": "cyberpunk neon lights futuristic city wallpaper",
    "vaporwave": "vaporwave retro pink purple aesthetic wallpaper",
    "minimalist": "minimalist simple clean wallpaper with negative space",
    "vintage": "vintage retro wallpaper with film grain and faded colors",
    "natural": "natural organic earthy green brown wallpaper",
}

COMPOSITION_VOCABULARY = {
    "rule_of_thirds": "composition following rule of thirds with subject off-center",
    "centered": "centered composition with subject in the middle",
    "symmetrical": "symmetrical balanced mirrored composition",
    "leading_lines": "composition with leading lines guiding the eye",
    "diagonal": "dynamic diagonal composition with angular elements",
    "minimal": "minimal negative space simple composition",
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ExtractedMetadata:
    """Container for all extracted metadata."""
    # Category classification
    primary_category: str = ""
    subcategories: List[str] = field(default_factory=list)
    category_confidence: float = 0.0
    
    # Mood and style
    mood_tags: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    
    # Composition
    composition_type: str = ""
    symmetry_score: float = 0.0
    focal_point: Tuple[float, float] = (0.5, 0.5)
    focal_point_method: str = "gradient"  # "dinov3" or "gradient"
    
    # Colors
    color_palette: List[str] = field(default_factory=list)
    dominant_hue: int = 0
    color_diversity: float = 0.0
    
    # Brightness and contrast
    brightness: int = 128
    contrast: float = 50.0
    is_dark_mode_friendly: bool = False


# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================

class MetadataExtractor:
    """
    Training-free metadata extraction using embedding models.
    
    Reuses already-loaded models from EmbeddingExtractor and MLQualityScorer.
    """
    
    def __init__(
        self,
        siglip_model=None,
        siglip_processor=None,
        dinov3_model=None,
        dinov3_processor=None,
        device: str = "cpu"
    ):
        """
        Initialize metadata extractor with pre-loaded models.
        
        Args:
            siglip_model: Pre-loaded SigLIP model
            siglip_processor: Pre-loaded SigLIP processor
            dinov3_model: Pre-loaded DINOv3 model
            dinov3_processor: Pre-loaded DINOv3 processor
            device: Device to run on (cpu/cuda/mps)
        """
        self.siglip_model = siglip_model
        self.siglip_processor = siglip_processor
        self.dinov3_model = dinov3_model
        self.dinov3_processor = dinov3_processor
        self.device = device
        
        # SigLIP parameters for probability calculation
        self.logit_scale = None
        self.logit_bias = None
        
        # Precomputed text embeddings (lazy loading)
        self._category_embeddings = None
        self._mood_embeddings = None
        self._style_embeddings = None
        self._composition_embeddings = None
        self._text_embeddings_ready = False
        
        # Extract params and precompute text embeddings if SigLIP available now
        self._ensure_text_embeddings()
    
    def set_siglip_model(self, siglip_model, siglip_processor, device: str = None):
        """
        Set or update SigLIP model after initialization.
        
        This is useful when SigLIP is lazy-loaded elsewhere and becomes available later.
        """
        self.siglip_model = siglip_model
        self.siglip_processor = siglip_processor
        if device:
            self.device = device
        
        # Reset and precompute text embeddings
        self._text_embeddings_ready = False
        self._ensure_text_embeddings()
    
    def _extract_siglip_params(self):
        """Extract logit_scale and logit_bias from SigLIP model."""
        if self.siglip_model is None:
            return

        try:
            # HuggingFace SiglipModel stores these as parameters
            if hasattr(self.siglip_model, "logit_scale"):
                self.logit_scale = self.siglip_model.logit_scale.item()
                if hasattr(self.siglip_model, "logit_bias"):
                    # logit_bias is usually a vector in training, but scalar scalar for inference?
                    # Check shape. If vector, maybe take mean or 0? 
                    # SigLIP paper uses a scalar bias per class, but for zero-shot we often rely on the dot product.
                    # HuggingFace implem often has logit_bias as scalar or None.
                    bias = self.siglip_model.logit_bias
                    if hasattr(bias, "item"):
                        self.logit_bias = bias.item()
                    else:
                        self.logit_bias = 0.0 # Default fallback
            
            logger.info(f"Extracted SigLIP params: scale={self.logit_scale}, bias={self.logit_bias}")
        except Exception as e:
            logger.warning(f"Failed to extract SigLIP params: {e}. Using defaults.")
            self.logit_scale = np.log(10) # Default approximate scale
            self.logit_bias = -10 # Default bias
    
    def _ensure_text_embeddings(self) -> bool:
        """
        Ensure text embeddings are precomputed. Called lazily when needed.
        
        Returns:
            True if text embeddings are ready, False otherwise.
        """
        if self._text_embeddings_ready:
            return True
        
        if self.siglip_model is None or self.siglip_processor is None:
            logger.debug("Cannot precompute text embeddings: SigLIP model not available")
            return False
            
        # Extract model params first
        self._extract_siglip_params()
        
        logger.info("Precomputing text embeddings for zero-shot classification...")
        self._category_embeddings = self._precompute_text_embeddings(CATEGORY_VOCABULARY)
        self._mood_embeddings = self._precompute_text_embeddings(MOOD_VOCABULARY)
        self._style_embeddings = self._precompute_text_embeddings(STYLE_VOCABULARY)
        self._composition_embeddings = self._precompute_text_embeddings(COMPOSITION_VOCABULARY)
        
        # Check if any succeeded
        if self._category_embeddings is not None:
            self._text_embeddings_ready = True
            logger.info("Text embeddings precomputed successfully")
            return True
        else:
            logger.warning("Failed to precompute text embeddings")
            return False
    
    @classmethod
    def from_embedding_extractor(cls, embedding_extractor, quality_scorer=None) -> "MetadataExtractor":
        """
        Create MetadataExtractor from existing EmbeddingExtractor instance.
        
        Args:
            embedding_extractor: EmbeddingExtractor with loaded models
            quality_scorer: Optional MLQualityScorer for SigLIP access
        """
        # Get SigLIP from quality scorer if available (it's already loaded there)
        siglip_model = None
        siglip_processor = None
        
        if quality_scorer is not None:
            siglip_model = getattr(quality_scorer, '_siglip_model', None)
            siglip_processor = getattr(quality_scorer, '_siglip_processor', None)
        
        # Get DINOv3 from embedding extractor
        dinov3_model = getattr(embedding_extractor, '_dinov3_model', None)
        dinov3_processor = getattr(embedding_extractor, '_dinov3_processor', None)
        
        return cls(
            siglip_model=siglip_model,
            siglip_processor=siglip_processor,
            dinov3_model=dinov3_model,
            dinov3_processor=dinov3_processor,
            device=embedding_extractor.device
        )
    
    # =========================================================================
    # SIGLIP ZERO-SHOT CLASSIFICATION
    # =========================================================================
    
    def _precompute_text_embeddings(self, vocabulary: dict) -> Optional[np.ndarray]:
        """Precompute text embeddings for a vocabulary."""
        if self.siglip_model is None or self.siglip_processor is None:
            return None
        
        try:
            import torch
            import torch.nn.functional as F
            
            texts = list(vocabulary.values())
            text_inputs = self.siglip_processor(
                text=texts,
                padding=True,
                return_tensors="pt"
            )
            
            if self.device == "cuda":
                text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
            elif self.device == "mps":
                text_inputs = {k: v.to("mps") for k, v in text_inputs.items()}
            
            with torch.no_grad():
                text_features = self.siglip_model.get_text_features(**text_inputs)
                text_features = F.normalize(text_features, dim=-1)
            
            return text_features.cpu().numpy()
        except Exception as e:
            logger.warning(f"Failed to precompute text embeddings: {e}")
            return None
    
    def _classify_against_vocabulary(
        self,
        image_embedding: np.ndarray,
        vocabulary: dict,
        precomputed_embeddings: Optional[np.ndarray],
        top_k: int = 2,
        threshold: float = 0.5,
        method: str = "multi_label"  # "multi_label" or "single_label"
    ) -> List[Tuple[str, float]]:
        """
        Classify image against a vocabulary.
        
        Args:
            image_embedding: Image embedding vector
            vocabulary: Dict of {label: prompt}
            precomputed_embeddings: Normalized text embeddings
            top_k: Number of results to return
            threshold: Probability threshold (for multi_label)
            method: "multi_label" (independent sigmoid) or "single_label" (softmax)
            
        Returns:
            List of (label, score) tuples
        """
        if precomputed_embeddings is None:
            logger.debug("Classification skipped: precomputed_embeddings is None")
            return []
        
        try:
            # Ensure consistent dtype
            image_embedding = image_embedding.astype(np.float32)
            
            # CRITICAL: Normalize image embedding to unit length
            # This ensures dot product with normalized text embeddings = true cosine similarity
            # Without this, unnormalized embeddings produce artificially low values
            norm = np.linalg.norm(image_embedding)
            if norm > 1e-8:
                image_embedding = image_embedding / norm
            else:
                logger.warning("Image embedding has near-zero norm, cannot normalize")
                return []
            
            # Compute raw logits (cosine similarity since both normalized)
            # Range [-1, 1], typically [0.15, 0.30] for matches
            logits = np.dot(precomputed_embeddings, image_embedding)
            
            # Log raw statistics to debug low probabilities
            if logger.isEnabledFor(logging.DEBUG):
                l_min, l_max, l_mean = np.min(logits), np.max(logits), np.mean(logits)
                logger.debug(f"Raw logits (cos sim): min={l_min:.4f}, max={l_max:.4f}, mean={l_mean:.4f}")

            vocab_keys = list(vocabulary.keys())
            
            if method == "single_label":
                # For categories/composition: Pick the BEST match regardless of absolute score.
                # Use Softmax.
                # Scale logits by learned scale (or default ~100) to sharpen distribution
                scale = np.exp(self.logit_scale) if self.logit_scale is not None else 100.0
                scaled_logits = logits * scale
                
                # Softmax
                exp_logits = np.exp(scaled_logits - np.max(scaled_logits)) # Stable softmax
                probs = exp_logits / np.sum(exp_logits)
                
                # Sort all
                results = []
                for i, prob in enumerate(probs):
                    results.append((vocab_keys[i], float(prob)))
                results.sort(key=lambda x: x[1], reverse=True)
                
                # Check confidence of top result against a minimal reasonable threshold?
                # For softmax, if one is much higher than others, it's confident.
                return results[:top_k]
                
            else:
                # For Moods/Styles: Use RAW cosine similarity directly
                # The sigmoid with bias -16 is too aggressive for subjective aesthetics
                # Wallpaper mood/style classification typically has cosine similarities of 0.03-0.10
                # so we use the raw values with a low threshold instead of sigmoid
                
                # Use raw cosine similarities as scores (they're already in 0-1 range for normalized vectors)
                probs = logits
                
                # Use a very low threshold since these are raw cosine values, not probabilities
                # Typical values are 0.03-0.08 for subjective aesthetic matches
                raw_threshold = 0.04  # This captures the top matches
                
                # Log score statistics
                if len(probs) > 0:
                    min_score = float(np.min(probs))
                    max_score = float(np.max(probs))
                    mean_score = float(np.mean(probs))
                    logger.info(f"Raw cosine scores: min={min_score:.4f}, max={max_score:.4f}, mean={mean_score:.4f}")
                
                results = []
                for i, score in enumerate(probs):
                    if score >= raw_threshold:
                        results.append((vocab_keys[i], float(score)))
                
                results.sort(key=lambda x: x[1], reverse=True)
                
                if not results:
                    # If nothing passes threshold, just take the top 2 anyway
                    top_indices = np.argsort(probs)[-top_k:][::-1]
                    results = [(vocab_keys[i], float(probs[i])) for i in top_indices]
                    logger.info(f"No results passed threshold {raw_threshold:.3f}, returning top {top_k}: {results}")
                
                return results[:top_k]
            
        except Exception as e:
            logger.warning(
                f"Classification failed: {e}, "
                f"embedding shape: {image_embedding.shape}, dtype: {image_embedding.dtype}"
            )
            return []
    
    def classify_category(
        self,
        image_embedding: np.ndarray,
        top_k: int = 2
    ) -> Tuple[str, List[str], float]:
        """
        Classify image into categories using zero-shot classification.
        
        Returns:
            Tuple of (primary_category, subcategories, confidence)
        """
        if self._category_embeddings is None:
            self._category_embeddings = self._precompute_text_embeddings(CATEGORY_VOCABULARY)
        
        # Use single_label logic (Softmax) to ensure we always get a category
        results = self._classify_against_vocabulary(
            image_embedding,
            CATEGORY_VOCABULARY,
            self._category_embeddings,
            top_k=top_k,
            threshold=0.0, # Ignored for single_label
            method="single_label"
        )
        
        if not results:
            return "general", [], 0.0
        
        primary = results[0][0]
        confidence = results[0][1]
        
        # Subcategories are just the next best ones, if valid confidence
        subcats = [r[0] for r in results[1:] if r[1] > 0.1]
        
        return primary, subcats, confidence
    
    def classify_mood(self, image_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        """Classify image mood using zero-shot classification."""
        if self._mood_embeddings is None:
            self._mood_embeddings = self._precompute_text_embeddings(MOOD_VOCABULARY)
        
        results = self._classify_against_vocabulary(
            image_embedding,
            MOOD_VOCABULARY,
            self._mood_embeddings,
            top_k=top_k,
            threshold=0.05,  # Lowered for subjective wallpaper aesthetics
            method="multi_label"
        )
        
        return [r[0] for r in results]
    
    def classify_style(self, image_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        """Classify image style using zero-shot classification."""
        if self._style_embeddings is None:
            self._style_embeddings = self._precompute_text_embeddings(STYLE_VOCABULARY)
        
        results = self._classify_against_vocabulary(
            image_embedding,
            STYLE_VOCABULARY,
            self._style_embeddings,
            top_k=top_k,
            threshold=0.05,  # Lowered for subjective wallpaper aesthetics
            method="multi_label"
        )
        
        return [r[0] for r in results]
    
    def classify_composition(self, image_embedding: np.ndarray) -> str:
        """Classify composition type using zero-shot classification."""
        if self._composition_embeddings is None:
            self._composition_embeddings = self._precompute_text_embeddings(COMPOSITION_VOCABULARY)
        
        # Use single_label (Softmax)
        results = self._classify_against_vocabulary(
            image_embedding,
            COMPOSITION_VOCABULARY,
            self._composition_embeddings,
            top_k=1,
            threshold=0.0,
            method="single_label"
        )
        
        return results[0][0] if results else "balanced"
    
    # =========================================================================
    # PIXEL-BASED ANALYSIS
    # =========================================================================
    
    def extract_colors(
        self,
        image: Image.Image,
        n_colors: int = 5
    ) -> Tuple[List[str], int, float]:
        """
        Extract dominant colors using K-means in LAB space.
        
        Returns:
            Tuple of (hex_colors, dominant_hue, color_diversity)
        """
        # Convert to numpy and resize for efficiency
        img_small = image.resize((100, 100), Image.Resampling.LANCZOS)
        rgb = np.array(img_small)
        
        # Convert to LAB for perceptually uniform clustering
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        
        # Reshape and cluster
        pixels = lab.reshape(-1, 3).astype(np.float32)
        
        try:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
        except Exception:
            return ["#808080"], 0, 0.0
        
        # Convert cluster centers to hex
        hex_colors = []
        for center in kmeans.cluster_centers_:
            lab_pixel = np.uint8([[[center[0], center[1], center[2]]]])
            bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
            b, g, r = bgr_pixel[0][0]
            hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
        
        # Get dominant hue from largest cluster
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_idx = labels[np.argmax(counts)]
        dominant_lab = kmeans.cluster_centers_[dominant_idx]
        
        a, b = dominant_lab[1] - 128, dominant_lab[2] - 128
        hue = int(np.arctan2(b, a) * 180 / np.pi) % 360
        
        # Color diversity: average distance between cluster centers
        center_distances = []
        for i, c1 in enumerate(kmeans.cluster_centers_):
            for c2 in kmeans.cluster_centers_[i+1:]:
                center_distances.append(np.linalg.norm(c1 - c2))
        diversity = np.mean(center_distances) / 100 if center_distances else 0
        
        return hex_colors, hue, min(diversity, 1.0)
    
    def compute_brightness_contrast(
        self,
        image: Image.Image
    ) -> Tuple[int, float, bool]:
        """
        Compute brightness and contrast from grayscale image.
        
        Returns:
            Tuple of (brightness 0-255, contrast, is_dark_mode_friendly)
        """
        # Convert to grayscale
        gray = np.array(image.convert("L"))
        
        brightness = int(gray.mean())
        contrast = float(gray.std())
        
        # Dark mode friendly: low brightness, decent contrast
        is_dark_friendly = brightness < 60 and contrast > 20
        
        return brightness, contrast, is_dark_friendly
    
    def compute_symmetry(self, image: Image.Image) -> float:
        """
        Compute symmetry score (0-1).
        
        Uses pixel-wise MSE on the flipped image, but with Gaussian Blur 
        to be robust to high-frequency noise and textures.
        """
        # Resize to small square to focus on macro symmetry and ignore details
        # 64x64 is small enough to ignore minor textures but keep shapes
        img_small = image.resize((64, 64), Image.Resampling.LANCZOS)
        gray = np.array(img_small.convert("L"))
        
        # Apply Gaussian Blur to remove high-frequency noise
        # This makes the conceptual symmetry check much more robust
        blurred = cv2.GaussianBlur(gray, (5, 5), 2.0)
        
        # Calculate MSE between image and its horizontal flip
        flipped = np.fliplr(blurred)
        mse = np.mean((blurred - flipped) ** 2)
        
        # Normalize to 0-1 score (inverted MSE)
        # Using a sensitivity factor. 
        # MSE of ~500 (pixel diff ~22) should give score ~0.5
        sensitivity = 1000.0 
        score = np.exp(-mse / sensitivity)
        
        return float(max(0.0, min(1.0, score)))
    
    # =========================================================================
    # DINOV3 FOCAL POINT
    # =========================================================================
    
    def find_focal_point_dino(self, image: Image.Image) -> Tuple[float, float]:
        """
        Find focal point using DINOv3 attention maps.
        
        Returns:
            Tuple of (x, y) normalized coordinates (0-1)
        """
        if self.dinov3_model is None or self.dinov3_processor is None:
            return self._find_focal_point_gradient(image)
        
        try:
            import torch
            
            # Process image
            inputs = self.dinov3_processor(images=image, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.dinov3_model(**inputs, output_attentions=True)
                
                if outputs.attentions is not None and len(outputs.attentions) > 0:
                    # Get attention from last layer
                    last_attn = outputs.attentions[-1]  # [batch, heads, seq, seq]
                    
                    # Average across heads, get CLS attention to patches
                    cls_attn = last_attn[:, :, 0, 1:].mean(dim=1)  # [batch, num_patches]
                    
                    # Reshape to spatial grid
                    num_patches = cls_attn.shape[1]
                    h = w = int(num_patches ** 0.5)
                    
                    if h * w == num_patches:
                        attn_map = cls_attn.view(1, h, w).cpu().numpy()[0]
                        
                        # Find centroid of high-attention region
                        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                        total = attn_map.sum() + 1e-6
                        
                        cx = (x_coords * attn_map).sum() / total / w
                        cy = (y_coords * attn_map).sum() / total / h
                        
                        return (float(cx), float(cy))
            
            # Fallback to gradient method
            return self._find_focal_point_gradient(image)
            
        except Exception as e:
            logger.warning(f"DINOv3 focal point failed: {e}")
            return self._find_focal_point_gradient(image)
    
    def _find_focal_point_gradient(self, image: Image.Image) -> Tuple[float, float]:
        """Fallback focal point detection using gradient magnitude."""
        gray = np.array(image.resize((200, 200), Image.Resampling.LANCZOS).convert("L"))
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        h, w = gray.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        total = magnitude.sum() + 1e-6
        cx = (x_coords * magnitude).sum() / total / w
        cy = (y_coords * magnitude).sum() / total / h
        
        return (float(cx), float(cy))
    
    # =========================================================================
    # MAIN EXTRACTION
    # =========================================================================
    
    def extract_all(
        self,
        image: Image.Image,
        siglip_embedding: Optional[np.ndarray] = None,
        quality_score: float = 0.0
    ) -> ExtractedMetadata:
        """
        Extract all metadata from an image.
        
        Args:
            image: PIL Image to process
            siglip_embedding: Optional pre-computed SigLIP embedding
            quality_score: Quality score (1-10 scale) for optimization decisions
            
        Returns:
            ExtractedMetadata with all fields populated
        """
        metadata = ExtractedMetadata()
        
        # Pixel-based analysis (no ML)
        try:
            metadata.color_palette, metadata.dominant_hue, metadata.color_diversity = \
                self.extract_colors(image)
        except Exception as e:
            logger.warning(f"Color extraction failed: {e}")
        
        try:
            metadata.brightness, metadata.contrast, metadata.is_dark_mode_friendly = \
                self.compute_brightness_contrast(image)
        except Exception as e:
            logger.warning(f"Brightness/contrast failed: {e}")
        
        try:
            metadata.symmetry_score = self.compute_symmetry(image)
        except Exception as e:
            logger.warning(f"Symmetry computation failed: {e}")
        
        # SigLIP-based classification (if embedding available)
        if siglip_embedding is not None:
            logger.info(f"SigLIP embedding available: shape={siglip_embedding.shape}, "
                       f"dtype={siglip_embedding.dtype}, norm={np.linalg.norm(siglip_embedding):.4f}")
            try:
                metadata.primary_category, metadata.subcategories, metadata.category_confidence = \
                    self.classify_category(siglip_embedding)
                logger.info(f"Category result: {metadata.primary_category} (conf={metadata.category_confidence:.3f})")
            except Exception as e:
                logger.warning(f"Category classification failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            try:
                metadata.mood_tags = self.classify_mood(siglip_embedding)
                logger.info(f"Mood result: {metadata.mood_tags}")
            except Exception as e:
                logger.warning(f"Mood classification failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            try:
                metadata.style_tags = self.classify_style(siglip_embedding)
                logger.info(f"Style result: {metadata.style_tags}")
            except Exception as e:
                logger.warning(f"Style classification failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            try:
                metadata.composition_type = self.classify_composition(siglip_embedding)
            except Exception as e:
                logger.warning(f"Composition classification failed: {e}")
        else:
            logger.warning("SigLIP embedding is None - skipping ML classification")
        
        # DINOv3 focal point (quality-based optimization)
        # Premium tier (6.5+): Use DINOv3 for best accuracy
        # Standard tier (<6.5): Use gradient fallback for speed
        try:
            if quality_score >= 6.5 and self.dinov3_model is not None:
                metadata.focal_point = self.find_focal_point_dino(image)
                metadata.focal_point_method = "dinov3"
            else:
                # Fast gradient fallback for standard tier
                metadata.focal_point = self._find_focal_point_gradient(image)
                metadata.focal_point_method = "gradient"
        except Exception as e:
            logger.warning(f"Focal point detection failed: {e}")
        
        return metadata
