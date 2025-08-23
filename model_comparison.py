
"""
Model Comparison Script
-----------------------
Aggregates results from ./runs/<variant>/ and produces:
  - learning_curves_top1.png
  - learning_curves_val_loss.png
  - best_top1_bar.png
  - best_top5_bar.png (if available)
  - accuracy_vs_time_scatter.png (if experiment_results.json has timings)
  - family_averages_bar.png
  - per-class comparison bars for top-2 (if per_class_metrics.csv exist)
Saves outputs into ./comparison_outputs/
"""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

VARIANTS = [
    "r18_base","r18_plus","r34_base","r34_plus",
    "efficientnet_b0","efficientnet_b1","efficientnet_b2","densenet121"
]

RUNS = Path("./runs")
OUT = Path("./comparison_outputs")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = Path("experiment_results.json")

def find_metrics_path(variant):
    for p in [RUNS/variant/"metrics_log.tsv", RUNS/variant/variant/"metrics_log.tsv"]:
        if p.exists(): return p
    return None

def find_summary_path(variant):
    for p in [RUNS/variant/"model_summary.json", RUNS/variant/variant/"model_summary.json"]:
        if p.exists(): return p
    return None

def find_per_class_path(variant):
    for p in [RUNS/variant/"per_class_metrics.csv", RUNS/variant/variant/"per_class_metrics.csv"]:
        if p.exists(): return p
    return None

# Load times if present
times_by_variant = {}
if RESULTS_JSON.exists():
    with open(RESULTS_JSON) as f:
        data = json.load(f)
        if isinstance(data, list):
            for r in data:
                if isinstance(r, dict) and r.get("variant"):
                    times_by_variant[r["variant"]] = r.get("training_time", None)

# Collect metrics
per_epoch = {}
for v in VARIANTS:
    mp = find_metrics_path(v)
    if mp is not None:
        try:
            df = pd.read_csv(mp, sep="\t")
            per_epoch[v] = df
        except Exception as e:
            print(f"[WARN] Could not read metrics for {v}: {e}")

# 1) Top-1 learning curves
plt.figure(figsize=(10,6))
any_plot = False
for v, df in per_epoch.items():
    if "epoch" in df.columns and "top1" in df.columns:
        plt.plot(df["epoch"], df["top1"], label=v)
        any_plot = True
plt.title("Top-1 Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Top-1 Accuracy (%)")
if any_plot:
    plt.legend(loc="best")
plt.tight_layout()
plt.savefig(OUT/"learning_curves_top1.png", dpi=150)
plt.close()

# 2) Val loss curves
plt.figure(figsize=(10,6))
any_plot = False
for v, df in per_epoch.items():
    if "epoch" in df.columns and "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label=v)
        any_plot = True
plt.title("Validation Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
if any_plot:
    plt.legend(loc="best")
plt.tight_layout()
plt.savefig(OUT/"learning_curves_val_loss.png", dpi=150)
plt.close()

# 3) Best Top-1 & Top-5 bar charts
rows = []
for v, df in per_epoch.items():
    row = {"variant": v}
    row["best_top1"] = float(df["top1"].max()) if "top1" in df.columns else float("nan")
    row["best_top5"] = float(df["top5"].max()) if "top5" in df.columns else float("nan")
    rows.append(row)
perf = pd.DataFrame(rows).sort_values("best_top1", ascending=False)
perf.to_csv(OUT/"summary_performance.csv", index=False)

if not perf.empty and perf["best_top1"].notna().any():
    plt.figure(figsize=(10,6))
    plt.bar(perf["variant"], perf["best_top1"])
    plt.title("Best Top-1 Accuracy by Model")
    plt.xlabel("Model Variant")
    plt.ylabel("Best Top-1 Accuracy (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT/"best_top1_bar.png", dpi=150)
    plt.close()

if not perf.empty and perf["best_top5"].notna().any():
    plt.figure(figsize=(10,6))
    plt.bar(perf["variant"], perf["best_top5"])
    plt.title("Best Top-5 Accuracy by Model")
    plt.xlabel("Model Variant")
    plt.ylabel("Best Top-5 Accuracy (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT/"best_top5_bar.png", dpi=150)
    plt.close()

# 4) Accuracy vs time scatter
rows = []
for v, df in per_epoch.items():
    if "top1" in df.columns:
        best_top1 = float(df["top1"].max())
        t = times_by_variant.get(v, None)
        if t is not None:
            rows.append({"variant": v, "best_top1": best_top1, "minutes": t/60.0})
trade = pd.DataFrame(rows).sort_values("best_top1", ascending=False)
trade.to_csv(OUT/"accuracy_vs_time.csv", index=False)
if not trade.empty:
    plt.figure(figsize=(8,6))
    plt.scatter(trade["minutes"], trade["best_top1"])
    for _, r in trade.iterrows():
        plt.annotate(r["variant"], (r["minutes"], r["best_top1"]), xytext=(3,3), textcoords="offset points")
    plt.title("Best Top-1 vs Training Time (minutes)")
    plt.xlabel("Training Time (min)")
    plt.ylabel("Best Top-1 Accuracy (%)")
    plt.tight_layout()
    plt.savefig(OUT/"accuracy_vs_time_scatter.png", dpi=150)
    plt.close()

# 5) Family averages
def bucket(v):
    if v in ["r18_base","r34_base"]: return "ResNet (base)"
    if v in ["r18_plus","r34_plus"]: return "ResNet (plus)"
    if v.startswith("efficientnet"): return "EfficientNet"
    if v.startswith("densenet"): return "DenseNet"
    return "Other"

fam_rows = []
for v, df in per_epoch.items():
    if "top1" in df.columns:
        fam_rows.append({"family": bucket(v), "variant": v, "best_top1": float(df["top1"].max())})
fam = pd.DataFrame(fam_rows)
fam.to_csv(OUT/"family_raw.csv", index=False)

if not fam.empty:
    avg = fam.groupby("family", as_index=False)["best_top1"].mean().sort_values("best_top1", ascending=False)
    avg.to_csv(OUT/"family_avg.csv", index=False)
    plt.figure(figsize=(8,6))
    plt.bar(avg["family"], avg["best_top1"])
    plt.title("Average Best Top-1 by Model Family")
    plt.xlabel("Family")
    plt.ylabel("Average Best Top-1 (%)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT/"family_averages_bar.png", dpi=150)
    plt.close()

# 6) Per-class comparison (top-2)
# Find top-2 variants by best_top1
top2 = list(perf["variant"].head(2).values) if not perf.empty else []
pcs_loaded = 0
for v in top2:
    p = find_per_class_path(v)
    if p is None: 
        continue
    try:
        df = pd.read_csv(p)
        # Use recall as a proxy if accuracy not present
        if "accuracy" in df.columns:
            series = df.set_index(df.columns[0])["accuracy"]
        elif "recall" in df.columns:
            series = df.set_index(df.columns[0])["recall"]
        else:
            series = None
        if series is None or series.empty:
            continue
        plt.figure(figsize=(10,4))
        plt.bar(range(len(series.values)), series.values)
        plt.title(f"Per-Class Accuracy Proxy — {v}")
        plt.xlabel("Class (index order)")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(OUT/f"per_class_{v}.png", dpi=150)
        plt.close()
        pcs_loaded += 1
    except Exception as e:
        print(f"[WARN] Per-class metrics for {v}: {e}")

print("Done. Outputs in", OUT.resolve())
