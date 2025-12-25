#!/usr/bin/env python3
"""
Download ML Models for RAMEN Pipeline

Pre-downloads all ML models used in the quality scoring pipeline:
- TensorFlow: MobileNetV3Small, EfficientNetV2L
- PyTorch/HuggingFace: SigLIP, DINOv2

This script is used by the GitHub Actions workflow to cache models
and avoid re-downloading them on every run.
"""

import os
import sys


def main():
    """Download all required ML models."""
    print("📥 Downloading ML models for RAMEN pipeline...")
    print("=" * 60)
    
    # Create cache directories
    os.makedirs('./models', exist_ok=True)
    os.makedirs(os.path.expanduser('~/.cache/huggingface'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.cache/torch'), exist_ok=True)
    
    try:
        # =====================================================================
        # TensorFlow Models
        # =====================================================================
        print("\n1️⃣  Loading TensorFlow models...")
        print("   - MobileNetV3Small")
        print("   - EfficientNetV2L")
        
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV3Small, EfficientNetV2L
        
        # Suppress TF warnings
        tf.get_logger().setLevel('ERROR')
        
        # Download MobileNetV3Small
        MobileNetV3Small(weights='imagenet', include_top=False, pooling='avg')
        print("   ✅ MobileNetV3Small downloaded")
        
        # Download EfficientNetV2L
        EfficientNetV2L(weights='imagenet', include_top=False, pooling='avg')
        print("   ✅ EfficientNetV2L downloaded")
        
        # =====================================================================
        # PyTorch/HuggingFace Models
        # =====================================================================
        print("\n2️⃣  Loading PyTorch/HuggingFace models...")
        print("   - SigLIP (google/siglip-large-patch16-256)")
        
        from transformers import AutoModel, AutoProcessor
        
        # Download SigLIP
        AutoProcessor.from_pretrained('google/siglip-large-patch16-256')
        print("   ✅ SigLIP processor downloaded")
        
        AutoModel.from_pretrained('google/siglip-large-patch16-256')
        print("   ✅ SigLIP model downloaded")
        
        # =====================================================================
        # DINOv2 Model
        # =====================================================================
        print("\n3️⃣  Loading DINOv2 model...")
        print("   - DINOv2 ViT-L/14")
        
        import torch
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        print("   ✅ DINOv2 downloaded")
        
        # =====================================================================
        # Success
        # =====================================================================
        print("\n" + "=" * 60)
        print("✅ All models downloaded successfully!")
        print("=" * 60)
        return 0
        
    except ImportError as e:
        print(f"\n❌ Error: Missing required package - {e}", file=sys.stderr)
        print("Please ensure all dependencies from requirements.txt are installed.", file=sys.stderr)
        return 1
        
    except Exception as e:
        print(f"\n❌ Error downloading models: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
