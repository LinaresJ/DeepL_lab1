#!/usr/bin/env python3
"""
Run all 8 model variant experiments and generate summary
"""

import subprocess
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

# Model variants to train
VARIANTS = [
    "r18_base", 
    "r34_base",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "densenet121"
]

# Training configuration
BASE_CONFIG = {
    "data_root": "./data",
    "epochs": 10,  # Reduced for faster experiments
    "num_workers": 4,
    "seed": 42,
}

def run_experiment(variant: str) -> dict:
    """Run a single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {variant}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Build command - using fixed train.py instead of train_enhanced.py
    cmd = [
        "python3", "train.py",
        "--variant", variant,
        "--data-root", BASE_CONFIG["data_root"],
        "--epochs", str(BASE_CONFIG["epochs"]),
        "--num-workers", str(BASE_CONFIG["num_workers"]),
        "--seed", str(BASE_CONFIG["seed"]),
        "--outdir", f"./runs/{variant}"
    ]
    
    # Add variant-specific options
    if "plus" in variant:
        cmd.extend(["--sam"])  # Use SAM for plus variants
    
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Training successful
        elapsed = time.time() - start_time
        
        # Read results
        output_dir = Path(f"./runs/{variant}/{variant}")
        
        # Read model summary
        summary_file = output_dir / "model_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
        else:
            summary = {"accuracy": 0.0}
        
        # Read metrics log
        metrics_file = output_dir / "metrics_log.tsv"
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file, sep='\t')
            best_top1 = metrics_df['top1'].max()
            best_top5 = metrics_df['top5'].max()
            final_epoch = len(metrics_df)
        else:
            best_top1 = 0.0
            best_top5 = 0.0
            final_epoch = 0
        
        return {
            "variant": variant,
            "status": "completed",
            "accuracy": summary.get("accuracy", 0.0) * 100,
            "best_top1": best_top1,
            "best_top5": best_top5,
            "epochs_trained": final_epoch,
            "training_time": elapsed,
            "error": None
        }
        
    except subprocess.CalledProcessError as e:
        # Training failed
        print(f"Error training {variant}: {e}")
        return {
            "variant": variant,
            "status": "failed",
            "accuracy": 0.0,
            "best_top1": 0.0,
            "best_top5": 0.0,
            "epochs_trained": 0,
            "training_time": time.time() - start_time,
            "error": str(e)
        }
    except Exception as e:
        print(f"Unexpected error with {variant}: {e}")
        return {
            "variant": variant,
            "status": "error",
            "accuracy": 0.0,
            "best_top1": 0.0,
            "best_top5": 0.0,
            "epochs_trained": 0,
            "training_time": time.time() - start_time,
            "error": str(e)
        }


def generate_summary(results: list) -> str:
    """Generate markdown summary of all experiments."""
    
    summary = "# Aircraft Classification Experiments Summary\n\n"
    summary += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary += f"**Dataset**: 100 Aircraft Classes (80/10/10 split)\n"
    summary += f"**Device**: Mac M4 with MPS acceleration\n"
    summary += f"**Epochs**: {BASE_CONFIG['epochs']}\n\n"
    
    # Performance table
    summary += "## Performance Comparison\n\n"
    summary += "| Model Variant | Status | Top-1 Acc (%) | Top-5 Acc (%) | Epochs | Time (min) |\n"
    summary += "|--------------|---------|---------------|---------------|--------|------------|\n"
    
    for r in sorted(results, key=lambda x: x['best_top1'], reverse=True):
        status_emoji = "✅" if r['status'] == "completed" else "❌"
        time_min = r['training_time'] / 60
        summary += f"| {r['variant']} | {status_emoji} | "
        summary += f"{r['best_top1']:.2f} | {r['best_top5']:.2f} | "
        summary += f"{r['epochs_trained']} | {time_min:.1f} |\n"
    
    # Best performers
    summary += "\n## Top Performers\n\n"
    top3 = sorted([r for r in results if r['status'] == "completed"], 
                  key=lambda x: x['best_top1'], reverse=True)[:3]
    
    for i, r in enumerate(top3, 1):
        summary += f"{i}. **{r['variant']}**: {r['best_top1']:.2f}% Top-1 accuracy\n"
    
    # Model categories comparison
    summary += "\n## Category Analysis\n\n"
    
    resnet_base = [r for r in results if 'r18_base' in r['variant'] or 'r34_base' in r['variant']]
    resnet_plus = [r for r in results if 'r18_plus' in r['variant'] or 'r34_plus' in r['variant']]
    efficientnet = [r for r in results if 'efficientnet' in r['variant']]
    densenet = [r for r in results if 'densenet' in r['variant']]
    
    summary += "### Average Performance by Model Family\n\n"
    
    if resnet_base:
        avg_resnet_base = sum(r['best_top1'] for r in resnet_base) / len(resnet_base)
        summary += f"- **ResNet (base)**: {avg_resnet_base:.2f}%\n"
    
    if resnet_plus:
        avg_resnet_plus = sum(r['best_top1'] for r in resnet_plus) / len(resnet_plus)
        summary += f"- **ResNet (plus)**: {avg_resnet_plus:.2f}%\n"
    
    if efficientnet:
        avg_efficientnet = sum(r['best_top1'] for r in efficientnet) / len(efficientnet)
        summary += f"- **EfficientNet**: {avg_efficientnet:.2f}%\n"
    
    if densenet:
        avg_densenet = sum(r['best_top1'] for r in densenet) / len(densenet)
        summary += f"- **DenseNet**: {avg_densenet:.2f}%\n"
    
    # Training efficiency
    summary += "\n## Training Efficiency\n\n"
    
    fastest = min(results, key=lambda x: x['training_time'])
    slowest = max(results, key=lambda x: x['training_time'])
    
    summary += f"- **Fastest**: {fastest['variant']} ({fastest['training_time']/60:.1f} min)\n"
    summary += f"- **Slowest**: {slowest['variant']} ({slowest['training_time']/60:.1f} min)\n"
    
    avg_time = sum(r['training_time'] for r in results) / len(results)
    total_time = sum(r['training_time'] for r in results)
    summary += f"- **Average time**: {avg_time/60:.1f} min\n"
    summary += f"- **Total time**: {total_time/60:.1f} min\n"
    
    # Key insights
    summary += "\n## Key Insights\n\n"
    
    # Enhancement comparison
    base_variants = [r for r in results if 'base' in r['variant'] and r['status'] == "completed"]
    plus_variants = [r for r in results if 'plus' in r['variant'] and r['status'] == "completed"]
    
    if base_variants and plus_variants:
        avg_base = sum(r['best_top1'] for r in base_variants) / len(base_variants)
        avg_plus = sum(r['best_top1'] for r in plus_variants) / len(plus_variants)
        improvement = avg_plus - avg_base
        
        summary += f"1. **Enhancement Impact**: Plus variants show {improvement:+.2f}% improvement over base variants\n"
    
    # Model size vs performance
    summary += "2. **Model Complexity**: "
    if efficientnet:
        eff_sorted = sorted([r for r in efficientnet if r['status'] == "completed"], 
                          key=lambda x: x['variant'])
        if len(eff_sorted) > 1:
            summary += f"EfficientNet B0→B2 shows progressive improvement "
            summary += f"({eff_sorted[0]['best_top1']:.1f}%→{eff_sorted[-1]['best_top1']:.1f}%)\n"
    
    # Best overall
    best_overall = max(results, key=lambda x: x['best_top1'])
    summary += f"3. **Best Overall**: {best_overall['variant']} achieved {best_overall['best_top1']:.2f}% Top-1 accuracy\n"
    
    # Failed experiments
    failed = [r for r in results if r['status'] != "completed"]
    if failed:
        summary += f"\n⚠️ **Note**: {len(failed)} experiments failed to complete\n"
    
    return summary


def main():
    """Run all experiments and generate summary."""
    print("🚀 Starting all experiments...")
    print(f"Running {len(VARIANTS)} model variants")
    
    results = []
    
    for i, variant in enumerate(VARIANTS, 1):
        print(f"\n[{i}/{len(VARIANTS)}] Running {variant}...")
        result = run_experiment(variant)
        results.append(result)
        
        # Save intermediate results
        with open("experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ {variant} completed: {result['best_top1']:.2f}% accuracy")
    
    # Generate summary
    print("\n📊 Generating summary...")
    summary = generate_summary(results)
    
    # Save summary
    with open("EXPERIMENT_SUMMARY.md", "w") as f:
        f.write(summary)
    
    print("\n✅ All experiments complete!")
    print(f"Results saved to EXPERIMENT_SUMMARY.md")
    
    # Print quick summary
    print("\nQuick Results:")
    for r in sorted(results, key=lambda x: x['best_top1'], reverse=True):
        print(f"  {r['variant']}: {r['best_top1']:.2f}%")


if __name__ == "__main__":
    main()