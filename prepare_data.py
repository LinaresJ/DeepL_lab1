#!/usr/bin/env python3
"""
Prepare aircraft dataset for ImageFolder format with 80/10/10 split
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_dataset():
    """Convert CSV-based dataset to ImageFolder structure with proper splits."""
    
    # Create data directories
    base_dir = Path("data")
    for split in ["train", "val", "test"]:
        (base_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Read all data
    train_df = pd.read_csv("aviones/train.csv")
    val_df = pd.read_csv("aviones/val.csv") 
    test_df = pd.read_csv("aviones/test.csv")
    
    # Combine all data for custom 80/10/10 split
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print(f"Total samples: {len(all_df)}")
    
    # Get unique classes
    classes = sorted(all_df['Classes'].unique())
    print(f"Found {len(classes)} classes")
    
    # Perform stratified 80/10/10 split
    # First split: 80% train, 20% temp
    X = all_df['filename'].values
    y = all_df['Classes'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Second split: 10% val, 10% test from temp (50/50 split of 20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(all_df)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(all_df)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(all_df)*100:.1f}%)")
    
    # Create class directories
    for split in ["train", "val", "test"]:
        for cls in classes:
            (base_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Copy images to appropriate directories
    def copy_split_images(filenames, labels, split_name):
        copied = 0
        missing = 0
        
        for filename, label in zip(filenames, labels):
            src = Path("aviones/images") / filename
            dst = base_dir / split_name / label / filename
            
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1
        
        print(f"{split_name}: Copied {copied} images, {missing} missing")
        return copied
    
    print("\nCopying images...")
    train_count = copy_split_images(X_train, y_train, "train")
    val_count = copy_split_images(X_val, y_val, "val")
    test_count = copy_split_images(X_test, y_test, "test")
    
    total = train_count + val_count + test_count
    print(f"\nTotal images copied: {total}")
    
    # Verify structure
    print("\nDataset structure:")
    for split in ["train", "val", "test"]:
        split_path = base_dir / split
        n_classes = len(list(split_path.iterdir()))
        n_images = sum(len(list((split_path / cls).glob("*.jpg"))) for cls in split_path.iterdir() if cls.is_dir())
        print(f"  {split}: {n_classes} classes, {n_images} images")
    
    # Verify class distribution
    print("\nClass distribution in train set:")
    train_path = base_dir / "train"
    class_counts = {}
    for cls_dir in sorted(train_path.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*.jpg")))
            class_counts[cls_dir.name] = count
    
    # Show first 10 classes
    for i, (cls, count) in enumerate(sorted(class_counts.items())[:10]):
        print(f"  {cls}: {count} images")
    print("  ...")

if __name__ == "__main__":
    prepare_dataset()