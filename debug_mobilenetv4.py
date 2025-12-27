#!/usr/bin/env python3
"""
Debug script to check actual MobileNetV4 model output dimensions.
This will help us understand if we're getting 960 or 1280 dimensions.
"""

import sys

def check_model_architecture():
    """Check the actual model architecture and output dimensions."""
    try:
        import timm
        import torch
        
        print("=" * 70)
        print("MOBILENETV4 ARCHITECTURE DEBUG")
        print("=" * 70)
        
        # Check timm version
        print(f"\ntimm version: {timm.__version__}")
        
        # List all MobileNetV4 models
        print("\nAvailable MobileNetV4 models:")
        all_models = timm.list_models('mobilenetv4*')
        for model_name in all_models:
            print(f"  - {model_name}")
        
        # Load the specific model we're using
        model_name = 'mobilenetv4_conv_small.e2400_r224_in1k'
        print(f"\nLoading model: {model_name}")
        
        model = timm.create_model(
            model_name,
            pretrained=False,  # Don't download weights for this test
            num_classes=0  # Remove classifier
        )
        model.eval()
        
        # Test with dummy input
        test_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"\nModel output shape: {output.shape}")
        print(f"Output dimensions: {output.shape[1]}")
        
        # Check model structure
        print(f"\nModel structure (last few layers):")
        print(model)
        
        # Try to find the feature dimension
        if hasattr(model, 'num_features'):
            print(f"\nmodel.num_features: {model.num_features}")
        
        if hasattr(model, 'feature_info'):
            print(f"\nFeature info: {model.feature_info}")
        
        # Conclusion
        print("\n" + "=" * 70)
        if output.shape[1] == 960:
            print("✅ CORRECT: Model outputs 960 dimensions as expected")
            print("   Action: Revert projection layers to 960→576")
        elif output.shape[1] == 1280:
            print("⚠️  UNEXPECTED: Model outputs 1280 dimensions")
            print("   Possible causes:")
            print("   1. Wrong model variant loaded")
            print("   2. timm version issue")
            print("   3. Model definition changed")
            print("   Action: Keep current 1280→960→576 projection")
        else:
            print(f"❌ UNKNOWN: Model outputs {output.shape[1]} dimensions")
            print("   Action: Manual investigation required")
        print("=" * 70)
        
        return output.shape[1]
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("\nPlease install timm: pip install timm>=1.0.0")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    actual_dim = check_model_architecture()
    sys.exit(0 if actual_dim else 1)
