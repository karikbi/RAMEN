#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Part 2: Embedding Extraction

Extracts embeddings from 4 models: MobileNetV3, EfficientNetV2, SigLIP, and DINOv2.
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ============================================================================
# WARNING SUPPRESSIONS
# ============================================================================

# Suppress numpy deprecated warnings from keras (np.object, np.bool, etc.)
warnings.filterwarnings("ignore", message=".*`np.object` is a deprecated alias.*")
warnings.filterwarnings("ignore", message=".*`np.bool` is a deprecated alias.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")

# Suppress xFormers warnings from DINOv2
warnings.filterwarnings("ignore", message="xFormers is not available")

# Suppress slow image processor warning from transformers
warnings.filterwarnings("ignore", message=".*slow image processor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*is unset.*")

# Increase PIL pixel limit to prevent DecompressionBomb warnings for large wallpapers
Image.MAX_IMAGE_PIXELS = 200_000_000  # 200 megapixels (up from ~89MP default)

logger = logging.getLogger("wallpaper_curator")

# Model cache directory
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)

# Set cache directories before importing ML libraries
# Use HF_HOME instead of deprecated TRANSFORMERS_CACHE
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")
os.environ["TORCH_HOME"] = str(MODELS_DIR / "torch")


@dataclass
class EmbeddingSet:
    """Container for all 4 model embeddings."""
    mobilenet_v3: Optional[np.ndarray] = None  # 576-dim, int8
    efficientnet_v2: Optional[np.ndarray] = None  # 1280-dim, int8
    siglip: Optional[np.ndarray] = None  # 1152-dim, float16
    dinov2: Optional[np.ndarray] = None  # 1024-dim, int8
    
    mobilenet_v3_dim: int = 576
    efficientnet_v2_dim: int = 1280
    siglip_dim: int = 1152
    dinov2_dim: int = 1024
    
    def is_complete(self) -> bool:
        """Check if all embeddings are present."""
        return all([
            self.mobilenet_v3 is not None,
            self.efficientnet_v2 is not None,
            self.siglip is not None,
            self.dinov2 is not None
        ])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mobilenet_v3": self.mobilenet_v3.tolist() if self.mobilenet_v3 is not None else None,
            "efficientnet_v2": self.efficientnet_v2.tolist() if self.efficientnet_v2 is not None else None,
            "siglip": self.siglip.tolist() if self.siglip is not None else None,
            "dinov2": self.dinov2.tolist() if self.dinov2 is not None else None,
        }


class EmbeddingExtractor:
    """Extracts embeddings from multiple neural network models."""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize embedding extractor.
        
        Args:
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        """
        self.device = self._detect_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Lazy-loaded models
        self._mobilenet = None
        self._efficientnet = None
        self._siglip_model = None
        self._siglip_processor = None
        self._dinov2 = None
        self._dinov2_transform = None
    
    def _detect_device(self, device: str) -> str:
        """Detect available device."""
        if device != "auto":
            return device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    def _quantize_int8(self, embedding: np.ndarray) -> np.ndarray:
        """Quantize float32 embedding to int8."""
        return np.clip(embedding * 127, -128, 127).astype(np.int8)
    
    def _load_image(self, filepath, size: tuple) -> Image.Image:
        """Load and resize image."""
        img = Image.open(filepath)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.resize(size, Image.Resampling.LANCZOS)
    
    # =========================================================================
    # MOBILENET V3 (576-dim)
    # =========================================================================
    
    def _load_mobilenet(self):
        """Load MobileNetV3 model."""
        if self._mobilenet is None:
            try:
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')
                
                self._mobilenet = tf.keras.applications.MobileNetV3Small(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )
                logger.info("Loaded MobileNetV3-Small")
            except Exception as e:
                logger.error(f"Failed to load MobileNetV3: {e}")
        return self._mobilenet
    
    def extract_mobilenet(self, filepath) -> Optional[np.ndarray]:
        """Extract MobileNetV3 embedding (576-dim, int8)."""
        model = self._load_mobilenet()
        if model is None:
            return None
        
        try:
            import tensorflow as tf
            
            img = self._load_image(filepath, (224, 224))
            img_array = np.array(img, dtype=np.float32)
            img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            embedding = model.predict(img_array, verbose=0)[0]
            return self._quantize_int8(embedding)
        except Exception as e:
            logger.error(f"MobileNetV3 extraction failed: {e}")
            return None
    
    # =========================================================================
    # EFFICIENTNET V2 (1280-dim)
    # =========================================================================
    
    def _load_efficientnet(self):
        """Load EfficientNetV2 model."""
        if self._efficientnet is None:
            try:
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')
                
                self._efficientnet = tf.keras.applications.EfficientNetV2L(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )
                logger.info("Loaded EfficientNetV2-Large")
            except Exception as e:
                logger.error(f"Failed to load EfficientNetV2: {e}")
        return self._efficientnet
    
    def extract_efficientnet(self, filepath) -> Optional[np.ndarray]:
        """Extract EfficientNetV2 embedding (1280-dim, int8)."""
        model = self._load_efficientnet()
        if model is None:
            return None
        
        try:
            import tensorflow as tf
            
            img = self._load_image(filepath, (480, 480))
            img_array = np.array(img, dtype=np.float32)
            img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            embedding = model.predict(img_array, verbose=0)[0]
            return self._quantize_int8(embedding)
        except Exception as e:
            logger.error(f"EfficientNetV2 extraction failed: {e}")
            return None
    
    # =========================================================================
    # SIGLIP (1152-dim)
    # =========================================================================
    
    def _load_siglip(self):
        """Load SigLIP model from HuggingFace."""
        if self._siglip_model is None:
            try:
                from transformers import AutoProcessor, AutoModel
                import torch
                
                model_name = "google/siglip-large-patch16-384"
                self._siglip_processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
                self._siglip_model = AutoModel.from_pretrained(model_name)
                
                if self.device == "cuda":
                    self._siglip_model = self._siglip_model.cuda()
                elif self.device == "mps":
                    self._siglip_model = self._siglip_model.to("mps")
                
                self._siglip_model.eval()
                logger.info("Loaded SigLIP-Large")
            except Exception as e:
                logger.error(f"Failed to load SigLIP: {e}")
        return self._siglip_model, self._siglip_processor
    
    def extract_siglip(self, filepath) -> Optional[np.ndarray]:
        """Extract SigLIP embedding (1152-dim, float16)."""
        model, processor = self._load_siglip()
        if model is None or processor is None:
            return None
        
        try:
            import torch
            
            img = self._load_image(filepath, (384, 384))
            inputs = processor(images=img, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            
            embedding = outputs.cpu().numpy()[0]
            return embedding.astype(np.float16)
        except Exception as e:
            logger.error(f"SigLIP extraction failed: {e}")
            return None
    
    # =========================================================================
    # DINOV2 (1024-dim)
    # =========================================================================
    
    def _load_dinov2(self):
        """Load DINOv2 model from torch hub."""
        if self._dinov2 is None:
            try:
                import torch
                import torchvision.transforms as T
                
                self._dinov2 = torch.hub.load(
                    'facebookresearch/dinov2',
                    'dinov2_vitl14',
                    pretrained=True
                )
                
                if self.device == "cuda":
                    self._dinov2 = self._dinov2.cuda()
                elif self.device == "mps":
                    self._dinov2 = self._dinov2.to("mps")
                
                self._dinov2.eval()
                
                # DINOv2 transform
                self._dinov2_transform = T.Compose([
                    T.Resize((518, 518)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                logger.info("Loaded DINOv2-Large")
            except Exception as e:
                logger.error(f"Failed to load DINOv2: {e}")
        return self._dinov2, self._dinov2_transform
    
    def extract_dinov2(self, filepath) -> Optional[np.ndarray]:
        """Extract DINOv2 embedding (1024-dim, int8)."""
        model, transform = self._load_dinov2()
        if model is None or transform is None:
            return None
        
        try:
            import torch
            
            img = Image.open(filepath)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            img_tensor = transform(img).unsqueeze(0)
            
            if self.device == "cuda":
                img_tensor = img_tensor.cuda()
            elif self.device == "mps":
                img_tensor = img_tensor.to("mps")
            
            with torch.no_grad():
                embedding = model(img_tensor)
            
            embedding = embedding.cpu().numpy()[0]
            return self._quantize_int8(embedding)
        except Exception as e:
            logger.error(f"DINOv2 extraction failed: {e}")
            return None
    
    # =========================================================================
    # MAIN EXTRACTION FUNCTION
    # =========================================================================
    
    def extract_all_embeddings(self, filepath) -> EmbeddingSet:
        """
        Extract embeddings from all 4 models.
        
        Args:
            filepath: Path to the image file.
        
        Returns:
            EmbeddingSet with all available embeddings.
        """
        embeddings = EmbeddingSet()
        
        logger.debug(f"Extracting embeddings for {filepath}")
        
        # MobileNetV3 (576-dim)
        embeddings.mobilenet_v3 = self.extract_mobilenet(filepath)
        
        # EfficientNetV2 (1280-dim)
        embeddings.efficientnet_v2 = self.extract_efficientnet(filepath)
        
        # SigLIP (1152-dim)
        embeddings.siglip = self.extract_siglip(filepath)
        
        # DINOv2 (1024-dim)
        embeddings.dinov2 = self.extract_dinov2(filepath)
        
        return embeddings
    
    def extract_all_with_recovery(
        self,
        filepath,
        wallpaper_id: str,
        progress_tracker=None,
        memory_monitor=None
    ) -> EmbeddingSet:
        """
        Extract embeddings with progress tracking for crash recovery.
        
        This method:
        - Checks which models have already been processed
        - Saves progress after each model completes
        - Monitors memory and triggers GC if needed
        - Continues with remaining models on resume
        
        Args:
            filepath: Path to the image file.
            wallpaper_id: Unique ID for progress tracking.
            progress_tracker: Optional EmbeddingProgressTracker instance.
            memory_monitor: Optional MemoryMonitor instance.
        
        Returns:
            EmbeddingSet with all available embeddings.
        """
        embeddings = EmbeddingSet()
        
        # Model extraction map
        model_extractors = {
            "mobilenet": ("mobilenet_v3", self.extract_mobilenet),
            "efficientnet": ("efficientnet_v2", self.extract_efficientnet),
            "siglip": ("siglip", self.extract_siglip),
            "dinov2": ("dinov2", self.extract_dinov2),
        }
        
        # Determine which models to process
        if progress_tracker:
            pending_models = progress_tracker.get_pending_models(wallpaper_id)
            
            # Load any previously saved embeddings
            saved = progress_tracker.get_saved_embeddings(wallpaper_id)
            for model, (attr, _) in model_extractors.items():
                if model in saved:
                    setattr(embeddings, attr, saved[model])
        else:
            pending_models = list(model_extractors.keys())
        
        logger.debug(f"Extracting embeddings for {wallpaper_id}, pending: {pending_models}")
        
        for model in pending_models:
            attr, extractor = model_extractors[model]
            
            # Check memory before each model
            if memory_monitor and memory_monitor.should_gc():
                memory_monitor.force_gc()
            
            try:
                embedding = extractor(filepath)
                setattr(embeddings, attr, embedding)
                
                # Mark complete and save progress
                if progress_tracker:
                    # Convert numpy array to list for JSON serialization
                    embedding_data = embedding.tolist() if embedding is not None else None
                    progress_tracker.mark_complete(wallpaper_id, model, embedding_data)
                
            except Exception as e:
                logger.warning(f"Model {model} failed for {wallpaper_id}: {e}")
                # Graceful degradation: continue with other models
                continue
        
        return embeddings
    
    def release_models(self) -> None:
        """
        Release loaded models to free memory.
        
        Useful when processing in batches to manage memory.
        """
        import gc
        
        self._mobilenet = None
        self._efficientnet = None
        self._siglip_model = None
        self._siglip_processor = None
        self._dinov2 = None
        self._dinov2_transform = None
        
        gc.collect()
        
        # Try to clear GPU memory if using PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("Released embedding models from memory")

