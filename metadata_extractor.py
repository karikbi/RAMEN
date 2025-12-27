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

CATEGORY_VOCABULARY = {
    # Natural & Landscapes
    "nature": "a landscape photo of mountains, forests, oceans, lakes, rivers, or natural scenery",
    "space": "a photo of galaxies, nebulae, stars, planets, aurora, or cosmic phenomena",
    "animals": "photos of animals, birds, wildlife, pets, or creatures in nature",
    "underwater": "underwater photography, ocean depths, marine life, or aquatic scenes",
    "aerial": "aerial views, drone photography, or bird's eye view landscapes",
    
    # Urban & Architecture
    "urban": "a photo of city skylines, streets, urban landscapes, or metropolitan scenes",
    "architecture": "architectural photography, buildings, structures, or modern design",
    "automotive": "cars, motorcycles, vehicles, racing, or automotive photography",
    
    # Art & Design
    "abstract": "abstract art, geometric patterns, gradients, shapes, or non-representational designs",
    "minimalist": "minimalist design, simple clean aesthetics, negative space, or minimal compositions",
    "art": "digital art, illustration, painting, fantasy art, or artistic concept art",
    "anime": "anime art style, manga illustrations, or Japanese animation aesthetics",
    "vintage": "vintage aesthetics, retro style, old-fashioned, or nostalgic imagery",
    
    # Technology & Gaming
    "technology": "cyberpunk, futuristic, sci-fi, neon, or technology-themed imagery",
    "gaming": "video game screenshots, gaming aesthetics, or game-related artwork",
    
    # Themes & Moods
    "dark": "dark themes, black backgrounds, AMOLED-friendly, or low-key photography",
    "colorful": "bright vibrant colors, vivid imagery, rainbow colors, or highly saturated photos",
    
    # Seasonal & Weather
    "seasonal": "seasonal imagery, spring blossoms, autumn leaves, winter snow, or summer scenes",
    "weather": "weather phenomena, rain, storms, clouds, lightning, or atmospheric conditions",
}

MOOD_VOCABULARY = {
    "calm": "a calm peaceful serene tranquil wallpaper",
    "dramatic": "a dramatic intense powerful striking wallpaper",
    "mysterious": "a mysterious dark moody atmospheric wallpaper",
    "energetic": "a vibrant energetic dynamic lively wallpaper",
    "romantic": "a romantic warm soft dreamy wallpaper",
    "melancholic": "a melancholic sad nostalgic lonely wallpaper",
    "inspiring": "an inspiring uplifting motivational hopeful wallpaper",
    "cozy": "a cozy warm comfortable inviting wallpaper",
    "epic": "an epic grand majestic awe-inspiring wallpaper",
    "surreal": "a surreal dreamlike otherworldly fantastical wallpaper",
}

STYLE_VOCABULARY = {
    "nord": "a nord color scheme blue gray dark wallpaper",
    "gruvbox": "a gruvbox color scheme orange brown retro wallpaper",
    "dracula": "a dracula theme purple dark gothic wallpaper",
    "monokai": "a monokai dark theme with vibrant accent colors",
    "solarized": "a solarized color scheme warm yellow blue wallpaper",
    "catppuccin": "a catppuccin pastel color palette soft wallpaper",
    "cyberpunk": "a cyberpunk neon futuristic sci-fi wallpaper",
    "vaporwave": "a vaporwave aesthetic pink purple retro 80s wallpaper",
    "minimalist": "a minimalist simple clean sparse wallpaper",
    "vintage": "a vintage retro old-fashioned nostalgic wallpaper",
    "natural": "a natural organic earthy green brown wallpaper",
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
        
        # Precomputed text embeddings (eager loading for performance)
        # Precompute on init to avoid first-call delay
        self._category_embeddings = None
        self._mood_embeddings = None
        self._style_embeddings = None
        self._composition_embeddings = None
        
        # Precompute text embeddings if SigLIP available
        if self.siglip_model is not None and self.siglip_processor is not None:
            logger.info("Precomputing text embeddings for zero-shot classification...")
            self._category_embeddings = self._precompute_text_embeddings(CATEGORY_VOCABULARY)
            self._mood_embeddings = self._precompute_text_embeddings(MOOD_VOCABULARY)
            self._style_embeddings = self._precompute_text_embeddings(STYLE_VOCABULARY)
            self._composition_embeddings = self._precompute_text_embeddings(COMPOSITION_VOCABULARY)
            logger.info("Text embeddings precomputed (eliminates first-call delay)")
    
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
        threshold: float = 0.20
    ) -> List[Tuple[str, float]]:
        """Classify image against a vocabulary using cosine similarity."""
        if precomputed_embeddings is None:
            return []
        
        try:
            # Compute cosine similarities
            similarities = np.dot(precomputed_embeddings, image_embedding)
            
            # Get top-k above threshold
            vocab_keys = list(vocabulary.keys())
            results = []
            
            for i, score in enumerate(similarities):
                if score >= threshold:
                    results.append((vocab_keys[i], float(score)))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
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
        
        results = self._classify_against_vocabulary(
            image_embedding,
            CATEGORY_VOCABULARY,
            self._category_embeddings,
            top_k=top_k,
            threshold=0.20
        )
        
        if not results:
            return "general", [], 0.0
        
        primary = results[0][0]
        confidence = results[0][1]
        subcats = [r[0] for r in results[1:]]
        
        return primary, subcats, confidence
    
    def classify_mood(self, image_embedding: np.ndarray, top_k: int = 2) -> List[str]:
        """Classify image mood using zero-shot classification."""
        if self._mood_embeddings is None:
            self._mood_embeddings = self._precompute_text_embeddings(MOOD_VOCABULARY)
        
        results = self._classify_against_vocabulary(
            image_embedding,
            MOOD_VOCABULARY,
            self._mood_embeddings,
            top_k=top_k,
            threshold=0.22
        )
        
        return [r[0] for r in results]
    
    def classify_style(self, image_embedding: np.ndarray, top_k: int = 2) -> List[str]:
        """Classify image style using zero-shot classification."""
        if self._style_embeddings is None:
            self._style_embeddings = self._precompute_text_embeddings(STYLE_VOCABULARY)
        
        results = self._classify_against_vocabulary(
            image_embedding,
            STYLE_VOCABULARY,
            self._style_embeddings,
            top_k=top_k,
            threshold=0.22
        )
        
        return [r[0] for r in results]
    
    def classify_composition(self, image_embedding: np.ndarray) -> str:
        """Classify composition type using zero-shot classification."""
        if self._composition_embeddings is None:
            self._composition_embeddings = self._precompute_text_embeddings(COMPOSITION_VOCABULARY)
        
        results = self._classify_against_vocabulary(
            image_embedding,
            COMPOSITION_VOCABULARY,
            self._composition_embeddings,
            top_k=1,
            threshold=0.15
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
        Compute horizontal symmetry score by comparing with flipped image.
        
        Returns:
            Symmetry score 0-1 (1 = perfectly symmetrical)
        """
        # Convert to grayscale and resize for efficiency
        img_small = image.resize((200, 200), Image.Resampling.LANCZOS)
        gray = np.array(img_small.convert("L"), dtype=np.float32)
        
        # Compare with horizontally flipped version
        flipped = np.fliplr(gray)
        
        # Compute normalized difference
        diff = np.abs(gray - flipped).mean()
        symmetry = 1.0 - (diff / 255.0)
        
        return float(symmetry)
    
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
            try:
                metadata.primary_category, metadata.subcategories, metadata.category_confidence = \
                    self.classify_category(siglip_embedding)
            except Exception as e:
                logger.warning(f"Category classification failed: {e}")
            
            try:
                metadata.mood_tags = self.classify_mood(siglip_embedding)
            except Exception as e:
                logger.warning(f"Mood classification failed: {e}")
            
            try:
                metadata.style_tags = self.classify_style(siglip_embedding)
            except Exception as e:
                logger.warning(f"Style classification failed: {e}")
            
            try:
                metadata.composition_type = self.classify_composition(siglip_embedding)
            except Exception as e:
                logger.warning(f"Composition classification failed: {e}")
        
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
