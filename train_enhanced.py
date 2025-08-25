#!/usr/bin/env python3
"""
Enhanced PyTorch Image Classifier Training Script for Mac M4 with MPS
With comprehensive progress tracking and visualization
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
from collections import deque

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

# Progress tracking imports
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.text import Text

warnings.filterwarnings('ignore')

# Initialize Rich console
console = Console()


# ===================== Progress Tracking Classes =====================

class TrainingTracker:
    """Comprehensive training progress tracker."""
    
    def __init__(self, total_epochs: int, train_batches: int, val_batches: int):
        self.total_epochs = total_epochs
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.start_time = time.time()
        self.epoch_start_time = None
        
        # Metrics history
        self.train_losses = deque(maxlen=100)
        self.val_losses = []
        self.val_accs = []
        self.lrs = []
        
        # Best metrics
        self.best_acc = 0
        self.best_epoch = 0
        
        # Time tracking
        self.epoch_times = deque(maxlen=5)
        
    def start_epoch(self, epoch: int):
        """Start timing an epoch."""
        self.epoch_start_time = time.time()
        self.current_epoch = epoch
        
    def end_epoch(self, val_acc: float):
        """End epoch and update statistics."""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = self.current_epoch
    
    def get_eta(self) -> str:
        """Calculate ETA based on recent epoch times."""
        if not self.epoch_times:
            return "Calculating..."
        
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - self.current_epoch - 1
        eta_seconds = avg_epoch_time * remaining_epochs
        
        return str(timedelta(seconds=int(eta_seconds)))
    
    def get_speed(self) -> str:
        """Get current training speed."""
        if not self.epoch_times:
            return "N/A"
        
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        samples_per_sec = (self.train_batches * 32) / avg_epoch_time  # Approximate
        
        return f"{samples_per_sec:.1f} img/s"
    
    def create_dashboard(self, epoch: int, train_loss: float, val_loss: float, 
                        val_acc: float, lr: float) -> Table:
        """Create a rich dashboard with training metrics."""
        table = Table(title=f"Training Progress - Epoch {epoch+1}/{self.total_epochs}")
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Current", style="magenta")
        table.add_column("Best", style="green")
        
        table.add_row("Train Loss", f"{train_loss:.4f}", "-")
        table.add_row("Val Loss", f"{val_loss:.4f}", f"{min(self.val_losses) if self.val_losses else 0:.4f}")
        table.add_row("Val Acc", f"{val_acc:.2f}%", f"{self.best_acc:.2f}% (E{self.best_epoch+1})")
        table.add_row("Learning Rate", f"{lr:.2e}", "-")
        table.add_row("Speed", self.get_speed(), "-")
        table.add_row("ETA", self.get_eta(), "-")
        
        elapsed = str(timedelta(seconds=int(time.time() - self.start_time)))
        table.add_row("Elapsed", elapsed, "-")
        
        return table


# ===================== Device & Memory Management =====================

def get_device(force_cpu: bool = False) -> torch.device:
    """Get optimal device for Mac M4 with MPS support."""
    if force_cpu:
        return torch.device("cpu")
    
    if torch.backends.mps.is_available():
        console.print("✓ MPS device detected - using Apple Silicon GPU acceleration", style="green")
        return torch.device("mps")
    else:
        console.print("⚠️ MPS not available - falling back to CPU", style="yellow")
        return torch.device("cpu")


def clear_memory(device: torch.device):
    """Clear memory cache for MPS/CUDA devices."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


# ===================== Model Configurations =====================

MODEL_CONFIGS = {
    "r18_base": {
        "timm_name": "resnet18",
        "input_size": 224,
        "batch_size": 32,
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
        "batch_size": 24,
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
        "batch_size": 16,
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
    
    if use_se and "resnet" in model_name:
        model_name = model_name.replace("resnet", "seresnet")
    
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes if not config.get("use_custom_head") else 0
    )
    
    if config.get("use_custom_head"):
        in_features = model.num_features
        model.fc = EnhancedResNetHead(in_features, num_classes)
    
    param_groups = []
    
    if config.get("two_phase"):
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
            {"params": backbone_early, "lr_scale": 0.5},
            {"params": backbone_late, "lr_scale": 1.0},
            {"params": head_params, "lr_scale": 1.5},
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
            auto_augment="rand-m9-mstd0.5-inc1",
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
    
    train_transform = build_transforms(input_size, True, use_augmentation)
    val_transform = build_transforms(input_size, False, False)
    
    if val_path.exists():
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    else:
        full_dataset = datasets.ImageFolder(train_path)
        
        if val_split:
            val_size = int(len(full_dataset) * val_split)
            train_size = len(full_dataset) - val_size
            train_indices, val_indices = random_split(
                range(len(full_dataset)), [train_size, val_size]
            )
            
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
            
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
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
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
    ema_model: Optional[ModelEmaV2] = None,
    epoch: int = 0,
    show_progress: bool = True
) -> Tuple[float, float, float]:
    """Train for one epoch with progress tracking."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    if show_progress:
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", 
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
                   colour='green')
    else:
        pbar = loader
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
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
        
        loss = loss / accumulation_steps
        
        if device.type == "mps":
            loss.backward()
        else:
            scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_sam:
                optimizer.first_step()
                optimizer.zero_grad()
                
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
                
                optimizer.second_step()
            else:
                if device.type == "mps":
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
            
            optimizer.zero_grad()
            
            if ema_model is not None:
                ema_model.update(model)
        
        running_loss += loss.item() * accumulation_steps
        
        if mixup_fn is None:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        if show_progress and batch_idx % 10 == 0:
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100.0 * correct / total if total > 0 else 0.0
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    clear_memory(device)
    
    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy, 0.0


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    calc_top5: bool = True,
    epoch: int = 0,
    show_progress: bool = True
) -> Tuple[float, float, float]:
    """Validate model with progress tracking."""
    model.eval()
    
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    # Progress bar
    if show_progress:
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]  ", 
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
                   colour='blue')
    else:
        pbar = loader
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_top1 += predicted.eq(targets).sum().item()
            
            if calc_top5:
                _, top5_pred = outputs.topk(5, 1, True, True)
                correct_top5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            # Update progress bar
            if show_progress and batch_idx % 10 == 0:
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100.0 * correct_top1 / total
                pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
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
    console.print("\n[bold cyan]Running final evaluation...[/bold cyan]")
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    report = classification_report(
        all_targets, all_preds,
        target_names=class_names[:len(np.unique(all_targets))],
        output_dict=True
    )
    
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(output_dir / 'per_class_metrics.csv')
    
    summary = {
        "accuracy": report['accuracy'],
        "macro_avg": report['macro avg'],
        "weighted_avg": report['weighted avg'],
        "total_samples": len(all_targets)
    }
    
    with open(output_dir / 'model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    console.print(f"[green]✓ Evaluation complete. Results saved to {output_dir}[/green]")
    console.print(f"[yellow]  Final Accuracy: {report['accuracy']*100:.2f}%[/yellow]")


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
    parser = argparse.ArgumentParser(description='Enhanced PyTorch Classifier for Mac M4')
    
    # Model
    parser.add_argument('--variant', type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model variant to train')
    parser.add_argument('--use-se', action='store_true',
                        help='Use SE-ResNet variants')
    
    # Data
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--val-split', type=float, default=None,
                        help='Validation split ratio')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data workers')
    parser.add_argument('--balance-sampler', action='store_true',
                        help='Use weighted sampler')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30,  # Reduced for faster experiments
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Augmentation
    parser.add_argument('--no-augs', action='store_true',
                        help='Disable augmentations')
    parser.add_argument('--mixup', type=float, default=0.2,
                        help='Mixup alpha')
    parser.add_argument('--cutmix', type=float, default=0.2,
                        help='CutMix alpha')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing')
    
    # Advanced
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA')
    parser.add_argument('--sam', action='store_true',
                        help='Use SAM')
    parser.add_argument('--freeze-epochs', type=int, default=5,
                        help='Freeze backbone epochs')
    parser.add_argument('--final-resolution', type=int, default=288,
                        help='Final resolution')
    
    # System
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--outdir', type=str, default='./runs',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume checkpoint')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bars')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup
    device = get_device(args.force_cpu)
    output_dir = Path(args.outdir) / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Model config
    config = MODEL_CONFIGS[args.variant]
    batch_size = args.batch_size or config["batch_size"]
    
    if "plus" in args.variant:
        args.ema = True
    
    # Print configuration
    console.print(Panel.fit(
        f"[bold cyan]Training {args.variant}[/bold cyan]\n"
        f"Device: {device}\n"
        f"Batch Size: {batch_size}\n"
        f"Input Size: {config['input_size']}x{config['input_size']}\n"
        f"Epochs: {args.epochs}",
        title="Training Configuration"
    ))
    
    # Prepare data
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
    
    console.print(f"Classes: {num_classes}")
    console.print(f"Train samples: {len(train_dataset)}")
    console.print(f"Val samples: {len(val_dataset)}")
    
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size,
        args.num_workers,
        args.balance_sampler
    )
    
    # Gradient accumulation
    effective_batch_size = 128
    accumulation_steps = max(1, effective_batch_size // batch_size)
    console.print(f"Gradient accumulation: {accumulation_steps} steps")
    
    # Build model
    model, param_groups = build_model(
        args.variant,
        num_classes,
        args.use_se
    )
    model = model.to(device)
    
    # Optimizer
    base_optimizer = AdamW(
        [
            {**group, "lr": args.lr * group.pop("lr_scale")}
            for group in param_groups
        ],
        weight_decay=args.weight_decay
    )
    
    optimizer = SAM(base_optimizer, rho=0.05) if args.sam else base_optimizer
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    scheduler = CosineAnnealingLR(
        optimizer.base_optimizer if args.sam else optimizer,
        T_max=total_steps - warmup_steps
    )
    
    # Loss
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
    
    # Always use CrossEntropyLoss for validation (no soft targets)
    val_criterion = nn.CrossEntropyLoss()
    
    # EMA
    ema_model = None
    if args.ema:
        ema_model = ModelEmaV2(model, decay=0.999)
        console.print("[green]EMA: Enabled[/green]")
    
    # AMP
    scaler = GradScaler() if device.type == "cuda" else None
    
    # Resume
    start_epoch = 0
    best_acc = 0
    
    if args.resume:
        console.print(f"[yellow]Resuming from {args.resume}[/yellow]")
        start_epoch, best_acc = load_checkpoint(
            Path(args.resume),
            model,
            optimizer,
            scheduler,
            ema_model
        )
    
    # Initialize tracker
    tracker = TrainingTracker(args.epochs, len(train_loader), len(val_loader))
    
    # Training
    metrics_log = []
    patience_counter = 0
    freeze_backbone = config.get("two_phase", False) and start_epoch < args.freeze_epochs
    
    console.print("\n[bold green]Starting Training[/bold green]")
    console.print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        tracker.start_epoch(epoch)
        
        # Two-phase training
        if freeze_backbone and epoch == args.freeze_epochs:
            console.print(f"[yellow]Unfreezing backbone at epoch {epoch}[/yellow]")
            freeze_backbone = False
            for param in model.parameters():
                param.requires_grad = True
        
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name and "head" not in name and "classifier" not in name:
                    param.requires_grad = False
        
        # Warmup
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
            for param_group in (optimizer.base_optimizer if args.sam else optimizer).param_groups:
                param_group['lr'] = warmup_lr * param_group.get('lr_scale', 1.0)
        
        # Train
        train_loss, train_acc, _ = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, mixup_fn, accumulation_steps, args.sam, ema_model,
            epoch, not args.no_progress
        )
        
        # Validate
        val_model = ema_model.module if ema_model else model
        val_loss, val_top1, val_top5 = validate(
            val_model, val_loader, val_criterion, device, True,
            epoch, not args.no_progress
        )
        
        # Step scheduler
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        # Update tracker
        tracker.end_epoch(val_top1)
        tracker.train_losses.append(train_loss)
        tracker.val_losses.append(val_loss)
        tracker.val_accs.append(val_top1)
        
        current_lr = (optimizer.base_optimizer if args.sam else optimizer).param_groups[0]['lr']
        tracker.lrs.append(current_lr)
        
        # Display dashboard
        dashboard = tracker.create_dashboard(epoch, train_loss, val_loss, val_top1, current_lr)
        console.print(dashboard)
        
        metrics_log.append({
            'epoch': epoch + 1,
            'lr': current_lr,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'top1': val_top1,
            'top5': val_top5
        })
        
        # Save best
        if val_top1 > best_acc:
            best_acc = val_top1
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_acc,
                args, output_dir / 'best.pt', ema_model
            )
            console.print(f"[green]✓ New best: {best_acc:.2f}%[/green]")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            console.print(f"[yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break
        
        # Save last
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_acc,
            args, output_dir / 'last.pt', ema_model
        )
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv(output_dir / 'metrics_log.tsv', sep='\t', index=False)
    
    console.print("\n" + "=" * 60)
    console.print(f"[bold green]✅ Training complete! Best accuracy: {best_acc:.2f}%[/bold green]")
    
    # Final evaluation
    best_model_path = output_dir / 'best.pt'
    if best_model_path.exists():
        load_checkpoint(best_model_path, model)
        model = model.to(device)
    
    evaluate_model(model, val_loader, device, class_names, output_dir)
    
    console.print(f"[cyan]Results saved to {output_dir}[/cyan]")


if __name__ == "__main__":
    main()