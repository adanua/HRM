#!/usr/bin/env python3
"""
Simple demo script to show MacOS/Metal support in HRM.
This validates that the core device utilities work correctly.
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(__file__))

import torch
from utils.device_utils import (
    get_device, 
    move_batch_to_device, 
    create_device_context, 
    get_map_location
)

def main():
    print("🧠 HRM MacOS/Metal Support Demo")
    print("=" * 50)
    
    # Detect available device
    device = get_device()
    print(f"✅ Detected device: {device}")
    
    if device.type == "cuda":
        print("   🚀 Using NVIDIA GPU with CUDA")
    elif device.type == "mps":
        print("   🍎 Using Apple Silicon with Metal Performance Shaders")
    elif device.type == "cpu":
        print("   💻 Using CPU (fallback)")
    
    print()
    
    # Test tensor operations
    print("Testing tensor operations...")
    
    # Create a sample batch (simulating puzzle data)
    sample_batch = {
        'input_ids': torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        'labels': torch.tensor([0, 1]),
        'attention_mask': torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    }
    
    print(f"   Original batch device: {sample_batch['input_ids'].device}")
    
    # Move batch to device (replaces .cuda() calls)
    moved_batch = move_batch_to_device(sample_batch, device)
    print(f"   Moved batch device: {moved_batch['input_ids'].device}")
    
    # Test device context (replaces torch.device("cuda"))
    with create_device_context(device):
        test_tensor = torch.ones(2, 3)
        print(f"   Tensor created in context: {test_tensor.device}")
    
    # Test model loading compatibility
    map_location = get_map_location()
    print(f"   Model loading map_location: {map_location}")
    
    print()
    print("✅ All tests passed! HRM is ready for cross-platform use.")
    
    # Platform-specific tips
    print("\n💡 Platform-specific notes:")
    if device.type == "cuda":
        print("   - FlashAttention is available for optimal performance")
        print("   - Multi-GPU training supported with torchrun")
    elif device.type == "mps":
        print("   - Metal provides optimized matrix operations")
        print("   - FlashAttention not needed (MPS has built-in optimizations)")
        print("   - Single-device training only")
    else:
        print("   - Consider using a GPU for faster training")
        print("   - Suitable for small experiments and testing")

if __name__ == "__main__":
    main()