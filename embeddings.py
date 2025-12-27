#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Part 2: Embedding Extraction

Extracts embeddings from 4 models:
- MobileNetV4-Small (960D) - primary embedding model
- EfficientNetV2-Large (1280D) - visual similarity
- SigLIP 2 Large (1152D) - semantic understanding
- DINOv3-Large (1024D) - scene composition
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict

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

# HuggingFace token for gated model access (required for CI/CD)
# Set via: export HF_TOKEN=hf_xxx... or in GitHub Actions secrets
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


@dataclass
class EmbeddingSet:
    """Container for all model embeddings (V2 stack with backward compatibility)."""
    # MobileNetV4-Small: 960-dim primary embedding
    mobilenet_v4: Optional[np.ndarray] = None  # 960-dim, int8
    efficientnet_v2: Optional[np.ndarray] = None  # 1280-dim, int8
    siglip: Optional[np.ndarray] = None  # 1152-dim, float16 (now SigLIP 2)
    dinov2: Optional[np.ndarray] = None  # 1024-dim, int8 (legacy alias)
    dinov3: Optional[np.ndarray] = None  # 1024-dim, int8 (NEW)
    
    mobilenet_v4_dim: int = 960
    efficientnet_v2_dim: int = 1280
    siglip_dim: int = 1152
    dinov3_dim: int = 1024
    dinov2_dim: int = 1024  # legacy alias
    
    def is_complete(self) -> bool:
        """Check if all embeddings are present (V2 stack)."""
        return all([
            self.mobilenet_v4 is not None,
            self.efficientnet_v2 is not None,
            self.siglip is not None,
            self.dinov3 is not None
        ])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            # V2 stack (primary)
            "mobilenet_v4": self.mobilenet_v4.tolist() if self.mobilenet_v4 is not None else None,

            "efficientnet_v2": self.efficientnet_v2.tolist() if self.efficientnet_v2 is not None else None,
            "siglip": self.siglip.tolist() if self.siglip is not None else None,
            "dinov3": self.dinov3.tolist() if self.dinov3 is not None else None,
            # Legacy aliases
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
        self._mobilenet_v4 = None  # MobileNetV4-Small (timm)
        self._mobilenet_v4_projection_960 = None  # 1280->960 projection

        self._mobilenet = None  # Legacy MobileNetV3 (TensorFlow)
        self._efficientnet = None
        self._siglip_model = None
        self._siglip_processor = None
        self._dinov3_model = None  # DINOv3 (HuggingFace)
        self._dinov3_processor = None
        self._dinov2 = None  # Legacy DINOv2
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
    
    def _ensure_image(self, input_data: Union[str, Path, Image.Image], size: tuple = None) -> Image.Image:
        """
        Ensure input is a PIL Image and optionally resize.
        
        Args:
            input_data: Filepath or PIL Image
            size: Optional (width, height) to resize to
            
        Returns:
            PIL Image (RGB)
        """
        if isinstance(input_data, (str, Path)):
            img = Image.open(input_data)
        else:
            img = input_data
            
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        if size:
            # Always copy if resizing to avoid modifying original if it was passed in
            if img.size != size:
                img = img.resize(size, Image.Resampling.LANCZOS)
            elif isinstance(input_data, Image.Image):
                # meaningful copy to prevent side effects if strictly needed, 
                # but usually fine. For safety in pipeline:
                pass
                
        return img

    def _load_image(self, filepath, size: tuple) -> Image.Image:
        """Legacy helper - use _ensure_image instead."""
        return self._ensure_image(filepath, size)
    
    # =========================================================================
    # MOBILENET V4 (960-dim)
    # =========================================================================
    
    def _load_mobilenet_v4(self):
        """Load MobileNetV4-Small model from timm."""
        if self._mobilenet_v4 is None:
            try:
                import timm
                import torch
                
                model_name = 'mobilenetv4_conv_small.e2400_r224_in1k'
                self._mobilenet_v4 = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=0  # Remove classifier for embeddings
                )
                
                if self.device == "cuda":
                    self._mobilenet_v4 = self._mobilenet_v4.cuda()
                elif self.device == "mps":
                    self._mobilenet_v4 = self._mobilenet_v4.to("mps")
                
                self._mobilenet_v4.eval()
                
                # CRITICAL: Detect actual model output dimensions
                # MobileNetV4-Conv-Small SHOULD output 960-dim, but we've observed 1280-dim
                # Possible causes: wrong variant, timm version issue, or checkpoint mismatch
                with torch.no_grad():
                    test_input = torch.randn(1, 3, 224, 224)
                    if self.device == "cuda":
                        test_input = test_input.cuda()
                    elif self.device == "mps":
                        test_input = test_input.to("mps")
                    test_output = self._mobilenet_v4(test_input)
                    actual_dim = test_output.shape[1]
                    logger.info(f"MobileNetV4 actual output dimension: {actual_dim}")
                
                # Adaptive projection layer creation based on actual dimensions
                if actual_dim == 960:
                    # EXPECTED: MobileNetV4-Conv-Small outputs 960-dim
                    logger.info("✓ MobileNetV4 dimensions correct (960)")
                    self._mobilenet_v4_projection_960 = None  # No projection needed
                    self._mobilenet_v4_actual_dim = 960
                    logger.info("Loaded MobileNetV4-Small (960D, no projection needed)")
                    
                elif actual_dim == 1280:
                    # UNEXPECTED: Getting 1280-dim (possibly wrong variant or timm issue)
                    logger.warning(
                        f"⚠️  MobileNetV4 dimension mismatch! "
                        f"Expected 960-dim for Conv-Small, got {actual_dim}-dim. "
                        f"Possible causes: (1) Wrong model variant loaded, "
                        f"(2) timm version issue, (3) Checkpoint mismatch. "
                        f"Creating adaptive 1280->960 projection."
                    )
                    # Single-stage projection: 1280->960
                    self._mobilenet_v4_projection_960 = torch.nn.Linear(1280, 960)
                    
                    with torch.no_grad():
                        self._mobilenet_v4_projection_960.weight.data = torch.eye(960, 1280)
                        self._mobilenet_v4_projection_960.bias.data.zero_()
                    
                    if self.device == "cuda":
                        self._mobilenet_v4_projection_960 = self._mobilenet_v4_projection_960.cuda()
                    elif self.device == "mps":
                        self._mobilenet_v4_projection_960 = self._mobilenet_v4_projection_960.to("mps")
                    
                    self._mobilenet_v4_actual_dim = 1280
                    logger.info("Loaded MobileNetV4-Small (1280D -> 960D projection)")
                    
                else:
                    # CRITICAL ERROR: Unexpected dimension
                    raise ValueError(
                        f"MobileNetV4 output dimension {actual_dim} is neither 960 nor 1280! "
                        f"Model: {model_name}. This requires manual investigation."
                    )
            except Exception as e:
                logger.error(f"Failed to load MobileNetV4: {e}")
        return self._mobilenet_v4, self._mobilenet_v4_projection_960, getattr(self, '_mobilenet_v4_actual_dim', 1280)
    
    def extract_mobilenet_v4(self, input_data: Union[str, Path, Image.Image]) -> Optional[np.ndarray]:
        """
        Extract MobileNetV4 embeddings (960-dim).
        
        Returns:
            960-dim int8 embedding or None if extraction fails
        """
        model, projection_960, actual_dim = self._load_mobilenet_v4()
        if model is None:
            return None
        
        try:
            import torch
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            
            # Get model-specific transforms
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)
            
            img = self._ensure_image(input_data)
            img_tensor = transform(img).unsqueeze(0)
            
            if self.device == "cuda":
                img_tensor = img_tensor.cuda()
            elif self.device == "mps":
                img_tensor = img_tensor.to("mps")
            
            with torch.no_grad():
                embedding_raw = model(img_tensor)
                
                # Adaptive projection based on actual dimensions
                if actual_dim == 960:
                    # No projection needed
                    embedding_960 = embedding_raw
                elif actual_dim == 1280:
                    # Single-stage 1280->960 projection
                    embedding_960 = projection_960(embedding_raw)
                else:
                    logger.error(f"Unexpected dimension {actual_dim} during extraction")
                    return None
            
            emb_960 = embedding_960.cpu().numpy()[0]
            
            return self._quantize_int8(emb_960)
        except Exception as e:
            logger.error(f"MobileNetV4 extraction failed: {e}")
            logger.debug(f"MobileNetV4 extraction error details:", exc_info=True)
            return None
    
    # =========================================================================
    # MOBILENET V3 (576-dim) - LEGACY FALLBACK
    # =========================================================================
    
    def _load_mobilenet(self):
        """Load MobileNetV3 model (legacy fallback)."""
        if self._mobilenet is None:
            try:
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')
                
                self._mobilenet = tf.keras.applications.MobileNetV3Small(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg'
                )
                logger.info("Loaded MobileNetV3-Small (legacy fallback)")
            except Exception as e:
                logger.error(f"Failed to load MobileNetV3: {e}")
        return self._mobilenet
    
    def extract_mobilenet(self, input_data: Union[str, Path, Image.Image]) -> Optional[np.ndarray]:
        """Extract MobileNetV3 embedding (576-dim, int8) - legacy fallback."""
        model = self._load_mobilenet()
        if model is None:
            return None
        
        try:
            import tensorflow as tf
            
            img = self._ensure_image(input_data, (224, 224))
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
    
    def extract_efficientnet(self, input_data: Union[str, Path, Image.Image]) -> Optional[np.ndarray]:
        """Extract EfficientNetV2 embedding (1280-dim, int8)."""
        model = self._load_efficientnet()
        if model is None:
            return None
        
        try:
            import tensorflow as tf
            
            img = self._ensure_image(input_data, (480, 480))
            img_array = np.array(img, dtype=np.float32)
            img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            embedding = model.predict(img_array, verbose=0)[0]
            return self._quantize_int8(embedding)
        except Exception as e:
            logger.error(f"EfficientNetV2 extraction failed: {e}")
            return None
    
    # =========================================================================
    # SIGLIP 2 (1152-dim) - Upgraded from SigLIP v1
    # =========================================================================
    
    def _load_siglip(self):
        """Load SigLIP 2 model from HuggingFace."""
        if self._siglip_model is None:
            try:
                from transformers import AutoProcessor, AutoModel
                import torch
                
                # SigLIP 2: Better localization, multilingual, officially stable
                model_name = "google/siglip2-large-patch16-384"
                self._siglip_processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
                self._siglip_model = AutoModel.from_pretrained(model_name)
                
                if self.device == "cuda":
                    self._siglip_model = self._siglip_model.cuda()
                elif self.device == "mps":
                    self._siglip_model = self._siglip_model.to("mps")
                
                self._siglip_model.eval()
                logger.info("Loaded SigLIP 2 Large")
            except Exception as e:
                logger.error(f"Failed to load SigLIP 2: {e}")
        return self._siglip_model, self._siglip_processor
    
    def extract_siglip(self, input_data: Union[str, Path, Image.Image]) -> Optional[np.ndarray]:
        """Extract SigLIP embedding (1152-dim, float16)."""
        model, processor = self._load_siglip()
        if model is None or processor is None:
            return None
        
        try:
            import torch
            
            img = self._ensure_image(input_data, (384, 384))
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
    
    def calculate_siglip_quality_score(self, filepath) -> tuple[Optional[np.ndarray], float]:
        """
        Extract SigLIP embedding AND calculate quality score using text similarity.
        
        Uses softmax over positive vs negative prompt similarities to compute
        a probability-like quality score. More mathematically sound than raw
        similarity differences.
        
        Returns:
            Tuple of (embedding, quality_score). 
            - embedding: 1152-dim float16 vector
            - quality_score: 0.0-1.0, where higher = better quality
        """
        model, processor = self._load_siglip()
        if model is None or processor is None:
            logger.warning("SigLIP model not loaded, using default score 0.75")
            return None, 0.75  # Default pass score if model fails to load
        
        try:
            import torch
            import torch.nn.functional as F
            
            img = self._load_image(filepath, (384, 384))
            
            # Single comparison prompt pair - simpler and more reliable
            positive_prompt = "a high quality beautiful wallpaper photograph"
            negative_prompt = "a low quality blurry bad image"
            
            # Process image
            image_inputs = processor(images=img, return_tensors="pt")
            
            # Process both prompts
            text_inputs = processor(
                text=[positive_prompt, negative_prompt], 
                padding=True, 
                return_tensors="pt"
            )
            
            # Move to device
            if self.device == "cuda":
                image_inputs = {k: v.cuda() for k, v in image_inputs.items()}
                text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
            elif self.device == "mps":
                image_inputs = {k: v.to("mps") for k, v in image_inputs.items()}
                text_inputs = {k: v.to("mps") for k, v in text_inputs.items()}
            
            with torch.no_grad():
                # Get features
                image_features = model.get_image_features(**image_inputs)
                text_features = model.get_text_features(**text_inputs)
                
                # Normalize for cosine similarity
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Compute similarities: [positive_sim, negative_sim]
                similarities = torch.matmul(image_features, text_features.T)[0]
                
                pos_sim = similarities[0].item()
                neg_sim = similarities[1].item()
                
                # Use softmax with temperature to convert to probabilities
                # Temperature controls sensitivity - lower = more discriminative
                temperature = 0.1
                logits = torch.tensor([pos_sim, neg_sim]) / temperature
                probs = F.softmax(logits, dim=0)
                
                # Quality score = probability of matching positive prompt
                quality_score = probs[0].item()
                
                # Log for debugging
                logger.info(
                    f"Quality: pos_sim={pos_sim:.3f}, neg_sim={neg_sim:.3f}, "
                    f"score={quality_score:.3f}"
                )
                
                # Get embedding
                embedding = image_features.cpu().numpy()[0].astype(np.float16)
            
            return embedding, quality_score
            
        except Exception as e:
            logger.error(f"SigLIP quality scoring failed: {e}")
            # Return a default passing score on error - don't reject due to model issues
            return None, 0.75
    


    # =========================================================================
    # DINOV3 (1024-dim) - Upgraded from DINOv2
    # =========================================================================

    def _load_dinov3(self):
        """Load DINOv3 model from HuggingFace (requires HF_TOKEN for gated access)."""
        if self._dinov3_model is None:
            try:
                from transformers import AutoModel, AutoImageProcessor
                import torch
                
                # DINOv3: +6 mIoU composition, same 1024D dimensions
                # Gated model - requires HF_TOKEN (from env) after accepting terms
                model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
                
                # Use token for gated model access in CI/CD
                token = HF_TOKEN
                if not token:
                    logger.warning("HF_TOKEN not set - DINOv3 may fail if terms not accepted locally")
                
                self._dinov3_processor = AutoImageProcessor.from_pretrained(model_name, token=token)
                self._dinov3_model = AutoModel.from_pretrained(model_name, token=token)
                
                if self.device == "cuda":
                    self._dinov3_model = self._dinov3_model.cuda()
                elif self.device == "mps":
                    self._dinov3_model = self._dinov3_model.to("mps")
                
                self._dinov3_model.eval()
                logger.info("Loaded DINOv3-Large")
            except Exception as e:
                logger.error(f"Failed to load DINOv3: {e}")
                logger.info("Fallback: Will use DINOv2 if DINOv3 unavailable")
        return self._dinov3_model, self._dinov3_processor
    
    def extract_dinov3(self, input_data: Union[str, Path, Image.Image]) -> Optional[np.ndarray]:
        """Extract DINOv3 embedding (1024-dim, int8)."""
        model, processor = self._load_dinov3()
        if model is None or processor is None:
            return None
        
        try:
            import torch
            
            img = self._ensure_image(input_data)
            inputs = processor(images=img, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Get CLS token embedding (pooler output or last hidden state)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output
                else:
                    embedding = outputs.last_hidden_state[:, 0]
            
            embedding = embedding.cpu().numpy()[0]
            return self._quantize_int8(embedding)
        except Exception as e:
            logger.error(f"DINOv3 extraction failed: {e}")
            return None
    
    # =========================================================================
    # DINOV2 (1024-dim) - LEGACY FALLBACK
    # =========================================================================

    
    def _load_dinov2(self):
        """Load DINOv2 model from torch hub (legacy fallback)."""
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
                
                logger.info("Loaded DINOv2-Large (legacy fallback)")
            except Exception as e:
                logger.error(f"Failed to load DINOv2: {e}")
        return self._dinov2, self._dinov2_transform
    
    def extract_dinov2(self, input_data: Union[str, Path, Image.Image]) -> Optional[np.ndarray]:
        """Extract DINOv2 embedding (1024-dim, int8) - legacy fallback."""
        model, transform = self._load_dinov2()
        if model is None or transform is None:
            return None
        
        try:
            import torch
            
            # DINOv2 uses its own transform which includes resize
            img = self._ensure_image(input_data)
            
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
        Extract embeddings from all models (V2 stack).
        
        Args:
            filepath: Path to the image file.
        
        Returns:
            EmbeddingSet with all available embeddings.
        """
        embeddings = EmbeddingSet()
        
        logger.debug(f"Extracting embeddings for {filepath}")
        
        # MobileNetV4 (960-dim)
        embeddings.mobilenet_v4 = self.extract_mobilenet_v4(filepath)
        
        # EfficientNetV2 (1280-dim)
        embeddings.efficientnet_v2 = self.extract_efficientnet(filepath)
        
        # SigLIP 2 (1152-dim)
        embeddings.siglip = self.extract_siglip(filepath)
        
        # DINOv3 (1024-dim) + legacy alias
        embeddings.dinov3 = self.extract_dinov3(filepath)
        embeddings.dinov2 = embeddings.dinov3  # Legacy alias
        
        return embeddings
    
    def extract_optimized(
        self, 
        filepath: Union[str, Path], 
        skip_siglip: bool = False
    ) -> EmbeddingSet:
        """
        Optimized extraction that loads the image once for all models.
        
        Args:
            filepath: Path to the image file.
            skip_siglip: If True, skip SigLIP extraction (e.g. if already computed).
            
        Returns:
            EmbeddingSet with requested embeddings.
        """
        embeddings = EmbeddingSet()
        
        try:
            # Load and decode image ONCE
            with Image.open(filepath) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    
                # Force load into memory to allow closing file
                img.load()
                
                # MobileNetV4 (960-dim)
                embeddings.mobilenet_v4 = self.extract_mobilenet_v4(img)
                
                # EfficientNetV2 (1280-dim)
                embeddings.efficientnet_v2 = self.extract_efficientnet(img)
                
                # SigLIP 2 (1152-dim)
                if not skip_siglip:
                    embeddings.siglip = self.extract_siglip(img)
                
                # DINOv3 (1024-dim) + legacy alias
                embeddings.dinov3 = self.extract_dinov3(img)
                embeddings.dinov2 = embeddings.dinov3  # Legacy alias
                
        except Exception as e:
            logger.error(f"Optimized batch extraction failed for {filepath}: {e}")
            
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
        
        # Model extraction map - V2 stack
        model_extractors = {
            "mobilenet_v4": ("mobilenet_v4", self.extract_mobilenet_v4),
            "efficientnet": ("efficientnet_v2", self.extract_efficientnet),
            "siglip": ("siglip", self.extract_siglip),
            "dinov3": ("dinov3", self.extract_dinov3),
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
        
        # V2 stack
        self._mobilenet_v4 = None
        self._mobilenet_v4_projection_960 = None

        self._dinov3_model = None
        self._dinov3_processor = None
        
        # Legacy models
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

