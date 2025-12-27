#!/usr/bin/env python3
"""
Test script to verify MobileNetV4 and ML category classification fixes.
"""

import sys
import logging
from pathlib import Path
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

def test_mobilenet_v4():
    """Test MobileNetV4 embedding extraction."""
    logger.info("=" * 60)
    logger.info("TEST 1: MobileNetV4 Embedding Extraction")
    logger.info("=" * 60)
    
    try:
        from embeddings import EmbeddingExtractor
        
        # Create test image
        test_img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        test_path = Path('/tmp/test_wallpaper.jpg')
        test_img.save(test_path)
        logger.info(f"Created test image: {test_path}")
        
        # Initialize extractor
        extractor = EmbeddingExtractor(device='cpu')
        logger.info("Initialized EmbeddingExtractor")
        
        # Test extraction
        emb_960, emb_576 = extractor.extract_mobilenet_v4(test_path)
        
        if emb_960 is not None and emb_576 is not None:
            logger.info(f"✓ MobileNetV4 960-dim shape: {emb_960.shape}")
            logger.info(f"✓ MobileNetV4 576-dim shape: {emb_576.shape}")
            logger.info(f"✓ MobileNetV4 960-dim dtype: {emb_960.dtype}")
            logger.info(f"✓ MobileNetV4 576-dim dtype: {emb_576.dtype}")
            
            # Verify shapes
            assert emb_960.shape == (960,), f"Expected (960,), got {emb_960.shape}"
            assert emb_576.shape == (576,), f"Expected (576,), got {emb_576.shape}"
            assert emb_960.dtype == np.int8, f"Expected int8, got {emb_960.dtype}"
            assert emb_576.dtype == np.int8, f"Expected int8, got {emb_576.dtype}"
            
            logger.info("✅ TEST 1 PASSED: MobileNetV4 extraction successful!")
            return True
        else:
            logger.error("✗ TEST 1 FAILED: MobileNetV4 returned None")
            return False
            
    except Exception as e:
        logger.error(f"✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ml_category_classification():
    """Test ML category classification with dtype fix."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: ML Category Classification")
    logger.info("=" * 60)
    
    try:
        from metadata_generator import CategoryClassifier
        from ml_quality_scorer import MLQualityScorer
        from embeddings import EmbeddingExtractor
        
        # Initialize components
        extractor = EmbeddingExtractor(device='cpu')
        scorer = MLQualityScorer(embedding_extractor=extractor)
        
        logger.info("Loading SigLIP model...")
        scorer._ensure_siglip_loaded()
        
        classifier = CategoryClassifier(
            siglip_model=scorer._siglip_model,
            siglip_processor=scorer._siglip_processor,
            device='cpu'
        )
        logger.info("Initialized CategoryClassifier")
        
        # Create dummy float16 embedding (like real SigLIP output)
        dummy_embedding = np.random.randn(1152).astype(np.float16)
        logger.info(f"Created dummy embedding: shape={dummy_embedding.shape}, dtype={dummy_embedding.dtype}")
        
        # Test classification
        results = classifier.classify(dummy_embedding, top_k=2, threshold=0.20)
        
        if results:
            logger.info(f"✓ Classification results: {results}")
            logger.info(f"✓ Top category: {results[0][0]} (confidence: {results[0][1]:.3f})")
            logger.info("✅ TEST 2 PASSED: ML classification successful!")
            return True
        else:
            logger.warning("⚠ TEST 2 WARNING: Classification returned empty results (may be below threshold)")
            logger.info("✅ TEST 2 PASSED: No dtype errors (empty results acceptable)")
            return True
            
    except Exception as e:
        logger.error(f"✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("Starting fix verification tests...")
    logger.info("")
    
    test1_passed = test_mobilenet_v4()
    test2_passed = test_ml_category_classification()
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Test 1 (MobileNetV4): {'✅ PASSED' if test1_passed else '✗ FAILED'}")
    logger.info(f"Test 2 (ML Classification): {'✅ PASSED' if test2_passed else '✗ FAILED'}")
    logger.info("")
    
    if test1_passed and test2_passed:
        logger.info("🎉 ALL TESTS PASSED! Fixes are working correctly.")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
