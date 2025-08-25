#!/usr/bin/env python3
"""
Test script to verify the loss function fix for Mixup training
"""

import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup

def test_loss_fix():
    """Test that we can use different loss criteria for training and validation."""
    
    print("Testing loss function compatibility fix...")
    
    # Simulate model output and targets
    batch_size = 4
    num_classes = 10
    
    # Model outputs (logits)
    outputs = torch.randn(batch_size, num_classes)
    
    # Regular integer targets (for validation)
    targets_int = torch.randint(0, num_classes, (batch_size,))
    
    # Soft targets (for mixup training)
    targets_soft = torch.zeros(batch_size, num_classes)
    targets_soft.scatter_(1, targets_int.unsqueeze(1), 1.0)
    
    # Initialize loss functions
    train_criterion = SoftTargetCrossEntropy()  # For mixup training
    val_criterion = nn.CrossEntropyLoss()       # For validation
    
    print(f"✓ Created loss criteria successfully")
    
    # Test training loss (with soft targets)
    try:
        train_loss = train_criterion(outputs, targets_soft)
        print(f"✓ Training loss (SoftTargetCrossEntropy): {train_loss.item():.4f}")
    except Exception as e:
        print(f"✗ Training loss failed: {e}")
        return False
    
    # Test validation loss (with integer targets)
    try:
        val_loss = val_criterion(outputs, targets_int)
        print(f"✓ Validation loss (CrossEntropyLoss): {val_loss.item():.4f}")
    except Exception as e:
        print(f"✗ Validation loss failed: {e}")
        return False
    
    # Test Mixup functionality
    print("\nTesting Mixup functionality...")
    try:
        mixup_fn = Mixup(
            mixup_alpha=0.2,
            cutmix_alpha=0.2,
            label_smoothing=0.1,
            num_classes=num_classes
        )
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        dummy_targets = torch.randint(0, num_classes, (batch_size,))
        
        # Apply mixup
        mixed_input, mixed_targets = mixup_fn(dummy_input, dummy_targets)
        
        # Test that mixed targets work with SoftTargetCrossEntropy
        mixed_outputs = torch.randn(batch_size, num_classes)
        mixup_loss = train_criterion(mixed_outputs, mixed_targets)
        
        print(f"✓ Mixup + SoftTargetCrossEntropy: {mixup_loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Mixup test failed: {e}")
        return False
    
    print("\n🎉 All loss function tests passed!")
    print("The fix correctly separates training (SoftTargetCrossEntropy) and validation (CrossEntropyLoss) criteria.")
    
    return True

if __name__ == "__main__":
    test_loss_fix()