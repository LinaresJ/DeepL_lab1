#!/usr/bin/env python3
"""
PyTorch Image Classifier Training Script for Mac M4 with MPS
Optimized for Mac mini M4 with MPS GPU acceleration
Supports 8 model variants with 100-class ImageFolder datasets
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

import timm
from timm.data import Mixup, create_transform
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')


# ===================== Device & Memory Management =====================

def get_device(force_cpu: bool = False) -> torch.device:
    """Get optimal device for Mac M4 with MPS support."""
    if force_cpu:
        return torch.device("cpu")
    
    if torch.backends.mps.is_available():
        print("✓ MPS device detected - using Apple Silicon GPU acceleration")
        return torch.device("mps")
    else:
        print("⚠️ MPS not available - falling back to CPU")
        return torch.device("cpu")


def clear_memory(device: torch.device):
    """Clear memory cache for MPS/CUDA devices."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def get_memory_usage(device: torch.device) -> str:
    """Get current memory usage string."""
    if device.type == "mps":
        # MPS doesn't have direct memory queries yet
        return "MPS memory monitoring not available"
    elif device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.2f}/{reserved:.2f}GB"
    return "CPU mode"


# ===================== Model Configurations =====================

MODEL_CONFIGS = {
    "r18_base": {
        "timm_name": "resnet18",
        "input_size": 224,
        "batch_size": 32,  # M4 optimized
        "use_custom_head": False,
    },
    "r18_plus": {
        "timm_name": "resnet18",
        "input_size": 224,
        "batch_size": 32,
        "use_custom_head": True,
        "use_ema": True,
        "two_phase": True,
    },
    "r34_base": {
        "timm_name": "resnet34",
        "input_size": 224,
        "batch_size": 24,  # Reduced for M4
        "use_custom_head": False,
    },
    "r34_plus": {
        "timm_name": "resnet34", 
        "input_size": 224,
        "batch_size": 24,
        "use_custom_head": True,
        "use_ema": True,
        "two_phase": True,
    },
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "input_size": 224,
        "batch_size": 32,
        "use_custom_head": False,
    },
    "efficientnet_b1": {
        "timm_name": "efficientnet_b1",
        "input_size": 240,
        "batch_size": 24,
        "use_custom_head": False,
    },
    "efficientnet_b2": {
        "timm_name": "efficientnet_b2",
        "input_size": 260,
        "batch_size": 16,  # Reduced for larger input
        "use_custom_head": False,
    },
    "densenet121": {
        "timm_name": "densenet121",
        "input_size": 224,
        "batch_size": 24,
        "use_custom_head": False,
    },
}


# ===================== Custom Model Heads =====================

class EnhancedResNetHead(nn.Module):
    """Enhanced head for ResNet plus variants."""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ===================== Model Builder =====================

def build_model(
    variant: str,
    num_classes: int,
    use_se: bool = False,
    pretrained: bool = True
) -> Tuple[nn.Module, List[Dict]]:
    """Build model with variant-specific configurations."""
    config = MODEL_CONFIGS[variant]
    model_name = config["timm_name"]
    
    # Handle SE-ResNet variants
    if use_se and "resnet" in model_name:
        model_name = model_name.replace("resnet", "seresnet")
    
    # Create base model
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes if not config.get("use_custom_head") else 0
    )
    
    # Apply custom head for plus variants
    if config.get("use_custom_head"):
        in_features = model.num_features
        model.fc = EnhancedResNetHead(in_features, num_classes)
    
    # Create parameter groups for discriminative learning rates
    param_groups = []
    
    if config.get("two_phase"):
        # Group parameters for discriminative LRs
        backbone_early = []
        backbone_late = []
        head_params = []
        
        for name, param in model.named_parameters():
            if "fc" in name or "head" in name or "classifier" in name:
                head_params.append(param)
            elif "layer1" in name or "layer2" in name:
                backbone_early.append(param)
            else:
                backbone_late.append(param)
        
        param_groups = [
            {"params": backbone_early, "lr_scale": 0.5},  # 1e-4 when base lr is 2e-4
            {"params": backbone_late, "lr_scale": 1.0},   # 2e-4
            {"params": head_params, "lr_scale": 1.5},     # 3e-4
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr_scale": 1.0}]
    
    return model, param_groups


# ===================== Data Transforms =====================

def build_transforms(
    input_size: int,
    is_training: bool,
    use_augmentation: bool = True
) -> transforms.Compose:
    """Build transforms with M4-optimized augmentations."""
    
    if is_training and use_augmentation:
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            auto_augment="rand-m9-mstd0.5-inc1",  # RandAugment
            interpolation="bicubic",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    else:
        transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    return transform


# ===================== Dataset & DataLoader =====================

def prepare_datasets(
    data_root: str,
    val_split: Optional[float],
    variant: str,
    use_augmentation: bool = True
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """Prepare train and validation datasets."""
    data_path = Path(data_root)
    train_path = data_path / "train"
    val_path = data_path / "val"
    
    config = MODEL_CONFIGS[variant]
    input_size = config["input_size"]
    
    # Build transforms
    train_transform = build_transforms(input_size, True, use_augmentation)
    val_transform = build_transforms(input_size, False, False)
    
    # Check for existing val folder
    if val_path.exists():
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    else:
        # Create validation split from training data
        full_dataset = datasets.ImageFolder(train_path)
        
        if val_split:
            val_size = int(len(full_dataset) * val_split)
            train_size = len(full_dataset) - val_size
            train_indices, val_indices = random_split(
                range(len(full_dataset)), [train_size, val_size]
            )
            
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
            
            # Apply transforms
            train_dataset.dataset.transform = train_transform
            val_dataset.dataset.transform = val_transform
        else:
            raise ValueError("No validation folder found and --val-split not specified")
    
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    batch_size: int,
    num_workers: int = 4,
    balance_sampler: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Create optimized dataloaders for M4."""
    
    train_sampler = None
    if balance_sampler:
        # Calculate class weights for balanced sampling
        targets = train_dataset.targets if hasattr(train_dataset, 'targets') else \
                 [train_dataset.dataset.targets[i] for i in train_dataset.indices]
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[t] for t in targets]
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,  # MPS doesn't use pin_memory
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


# ===================== SAM Optimizer Wrapper =====================

class SAM:
    """Sharpness Aware Minimization wrapper."""
    def __init__(self, base_optimizer, rho=0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho
        
    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.base_optimizer.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.requires_grad_(False)
                p.add_(p.grad * scale)
                p.requires_grad_(True)
                
    @torch.no_grad()
    def second_step(self):
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.requires_grad_(False)
                p.sub_(p.grad * (self.rho / (self._grad_norm() + 1e-12)))
                p.requires_grad_(True)
        self.base_optimizer.step()
        
    def _grad_norm(self):
        grad_norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.base_optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return grad_norm
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
        
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups
    
    def state_dict(self):
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


# ===================== Training Functions =====================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: any,
    device: torch.device,
    scaler: GradScaler,
    mixup_fn: Optional[Mixup],
    accumulation_steps: int,
    use_sam: bool,
    ema_model: Optional[ModelEmaV2] = None
) -> Tuple[float, float, float]:
    """Train for one epoch with M4 optimizations."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply mixup
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
        
        # Forward pass with autocast
        if device.type == "mps":
            # MPS autocast support
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        elif device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        if device.type == "mps":
            loss.backward()
        else:
            scaler.scale(loss).backward()
        
        # Optimizer step with accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_sam:
                # SAM first step
                optimizer.first_step()
                optimizer.zero_grad()
                
                # Second forward-backward
                if device.type == "mps":
                    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets) / accumulation_steps
                    loss.backward()
                elif device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets) / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / accumulation_steps
                    loss.backward()
                
                # SAM second step
                optimizer.second_step()
            else:
                if device.type == "mps":
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
            
            optimizer.zero_grad()
            
            # Update EMA
            if ema_model is not None:
                ema_model.update(model)
        
        # Calculate accuracy (handle mixup targets)
        running_loss += loss.item() * accumulation_steps
        
        if mixup_fn is None:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Clear memory after epoch
    clear_memory(device)
    
    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy, 0.0  # Return 0 for top5 as placeholder


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    calc_top5: bool = True
) -> Tuple[float, float, float]:
    """Validate model performance."""
    model.eval()
    
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_top1 += predicted.eq(targets).sum().item()
            
            # Top-5 accuracy
            if calc_top5:
                _, top5_pred = outputs.topk(5, 1, True, True)
                correct_top5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()
    
    avg_loss = running_loss / len(loader)
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total if calc_top5 else 0.0
    
    return avg_loss, top1_acc, top5_acc


# ===================== Evaluation & Visualization =====================

def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: Path
):
    """Comprehensive model evaluation with visualizations."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    # Classification report
    report = classification_report(
        all_targets, all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    # Save per-class metrics
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(output_dir / 'per_class_metrics.csv')
    
    # Model summary
    summary = {
        "accuracy": report['accuracy'],
        "macro_avg": report['macro avg'],
        "weighted_avg": report['weighted avg'],
        "total_samples": len(all_targets)
    }
    
    with open(output_dir / 'model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Evaluation complete. Results saved to {output_dir}")
    print(f"   Top-1 Accuracy: {report['accuracy']*100:.2f}%")


# ===================== Checkpoint Management =====================

def save_checkpoint(
    model: nn.Module,
    optimizer: any,
    scheduler: any,
    epoch: int,
    best_acc: float,
    args: argparse.Namespace,
    filepath: Path,
    ema_model: Optional[ModelEmaV2] = None
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'args': vars(args)
    }
    
    if ema_model is not None:
        checkpoint['ema_state_dict'] = ema_model.state_dict()
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: any = None,
    scheduler: any = None,
    ema_model: Optional[ModelEmaV2] = None
) -> Tuple[int, float]:
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if ema_model is not None and 'ema_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_acc']


# ===================== Main Training Function =====================

def main():
    parser = argparse.ArgumentParser(description='PyTorch Image Classifier for Mac M4')
    
    # Model
    parser.add_argument('--variant', type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model variant to train')
    parser.add_argument('--use-se', action='store_true',
                        help='Use SE-ResNet variants for ResNet models')
    
    # Data
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root with train/ and val/ folders')
    parser.add_argument('--val-split', type=float, default=None,
                        help='Validation split ratio if no val/ folder exists')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--balance-sampler', action='store_true',
                        help='Use weighted sampler for imbalanced classes')
    
    # Training
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides model default)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Number of warmup epochs')
    parser.add_argument('--patience', type=int, default=12,
                        help='Early stopping patience')
    
    # Augmentation
    parser.add_argument('--no-augs', action='store_true',
                        help='Disable data augmentations')
    parser.add_argument('--mixup', type=float, default=0.2,
                        help='Mixup alpha')
    parser.add_argument('--cutmix', type=float, default=0.2,
                        help='CutMix alpha')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    
    # Advanced
    parser.add_argument('--ema', action='store_true',
                        help='Use exponential moving average')
    parser.add_argument('--sam', action='store_true',
                        help='Use SAM optimizer')
    parser.add_argument('--freeze-epochs', type=int, default=5,
                        help='Number of epochs to freeze backbone (plus variants)')
    parser.add_argument('--final-resolution', type=int, default=288,
                        help='Final resolution for high-res fine-tuning')
    
    # System
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage')
    parser.add_argument('--memory-monitor', action='store_true',
                        help='Monitor memory usage')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--outdir', type=str, default='./runs',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = get_device(args.force_cpu)
    
    # Setup output directory
    output_dir = Path(args.outdir) / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Get model configuration
    config = MODEL_CONFIGS[args.variant]
    batch_size = args.batch_size or config["batch_size"]
    
    # Auto-enable features for plus variants
    if "plus" in args.variant:
        args.ema = True
    
    print(f"\n🚀 Training {args.variant} on {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Input size: {config['input_size']}x{config['input_size']}")
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(
        args.data_root,
        args.val_split,
        args.variant,
        not args.no_augs
    )
    
    num_classes = len(train_dataset.classes if hasattr(train_dataset, 'classes') 
                     else train_dataset.dataset.classes)
    class_names = (train_dataset.classes if hasattr(train_dataset, 'classes')
                  else train_dataset.dataset.classes)
    
    print(f"   Classes: {num_classes}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size,
        args.num_workers,
        args.balance_sampler
    )
    
    # Calculate gradient accumulation steps
    effective_batch_size = 128
    accumulation_steps = max(1, effective_batch_size // batch_size)
    print(f"   Gradient accumulation: {accumulation_steps} steps")
    
    # Build model
    model, param_groups = build_model(
        args.variant,
        num_classes,
        args.use_se
    )
    model = model.to(device)
    
    # Setup optimizer
    base_optimizer = AdamW(
        [
            {**group, "lr": args.lr * group.pop("lr_scale")}
            for group in param_groups
        ],
        weight_decay=args.weight_decay
    )
    
    optimizer = SAM(base_optimizer, rho=0.05) if args.sam else base_optimizer
    
    # Setup scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    scheduler = CosineAnnealingLR(
        optimizer.base_optimizer if args.sam else optimizer,
        T_max=total_steps - warmup_steps
    )
    
    # Setup loss function
    if not args.no_augs and (args.mixup > 0 or args.cutmix > 0):
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            label_smoothing=args.label_smoothing,
            num_classes=num_classes
        )
        criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        # Fix label_smoothing parameter
        if hasattr(nn.CrossEntropyLoss, 'label_smoothing'):
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    
    # Setup EMA
    ema_model = None
    if args.ema:
        ema_model = ModelEmaV2(model, decay=0.999)
        print("   EMA: Enabled")
    
    # Setup AMP scaler
    scaler = GradScaler() if device.type == "cuda" else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0
    
    if args.resume:
        print(f"\n📂 Resuming from {args.resume}")
        start_epoch, best_acc = load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            scheduler,
            ema_model
        )
    
    # Training metrics log
    metrics_log = []
    patience_counter = 0
    
    # Two-phase training for plus variants
    freeze_backbone = config.get("two_phase", False) and start_epoch < args.freeze_epochs
    
    print(f"\n🏋️ Starting training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        # Handle two-phase training
        if freeze_backbone and epoch == args.freeze_epochs:
            print(f"\n🔓 Unfreezing backbone at epoch {epoch}")
            freeze_backbone = False
            for param in model.parameters():
                param.requires_grad = True
        
        if freeze_backbone:
            # Freeze backbone, only train head
            for name, param in model.named_parameters():
                if "fc" not in name and "head" not in name and "classifier" not in name:
                    param.requires_grad = False
        
        # Handle high-res fine-tuning for plus variants
        if config.get("two_phase") and epoch == args.epochs - 25:
            print(f"\n🔍 Switching to high-res {args.final_resolution}x{args.final_resolution}")
            # Recreate datasets with new resolution
            train_dataset, val_dataset = prepare_datasets(
                args.data_root,
                args.val_split,
                args.variant,
                not args.no_augs
            )
            train_loader, val_loader = create_dataloaders(
                train_dataset,
                val_dataset,
                batch_size // 2,  # Reduce batch size for higher resolution
                args.num_workers,
                args.balance_sampler
            )
        
        # Adjust learning rate with warmup
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in (optimizer.base_optimizer if args.sam else optimizer).param_groups:
                param_group['lr'] = warmup_lr * param_group.get('lr_scale', 1.0)
        
        # Train epoch
        train_loss, train_acc, _ = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, mixup_fn, accumulation_steps, args.sam, ema_model
        )
        
        # Validate
        val_model = ema_model.module if ema_model else model
        val_loss, val_top1, val_top5 = validate(
            val_model, val_loader, criterion, device
        )
        
        # Step scheduler after warmup
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        # Log metrics
        current_lr = (optimizer.base_optimizer if args.sam else optimizer).param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"Top1: {val_top1:.2f}% "
              f"Top5: {val_top5:.2f}% "
              f"LR: {current_lr:.6f}")
        
        if args.memory_monitor:
            print(f"   Memory: {get_memory_usage(device)}")
        
        metrics_log.append({
            'epoch': epoch + 1,
            'lr': current_lr,
            'weight_decay': args.weight_decay,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'top1': val_top1,
            'top5': val_top5
        })
        
        # Save best model
        if val_top1 > best_acc:
            best_acc = val_top1
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_acc,
                args, output_dir / 'best.pt', ema_model
            )
            print(f"   ✓ New best: {best_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_acc,
            args, output_dir / 'last.pt', ema_model
        )
    
    # Save metrics log
    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv(output_dir / 'metrics_log.tsv', sep='\t', index=False)
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete! Best accuracy: {best_acc:.2f}%")
    
    # Final evaluation
    print("\n📊 Running final evaluation...")
    
    # Load best model
    best_model_path = output_dir / 'best.pt'
    if best_model_path.exists():
        load_checkpoint(best_model_path, model)
        model = model.to(device)
    
    evaluate_model(
        model,
        val_loader,
        device,
        class_names,
        output_dir
    )
    
    print(f"\n🎉 All results saved to {output_dir}")


if __name__ == "__main__":
    main()