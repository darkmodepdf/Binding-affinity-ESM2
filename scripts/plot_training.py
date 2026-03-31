import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

def main():
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument("--history", type=str, default=str(LOGS_DIR / "training_history.json"), help="Path to training history JSON")
    parser.add_argument("--output-dir", type=str, default=str(LOGS_DIR), help="Directory to save plots")
    args = parser.parse_args()

    history_path = Path(args.history)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        print(f"Error: {history_path} not found.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    if not history.get("train") or not history.get("val"):
        print("Error: training history is empty or incomplete.")
        return

    train_epochs = [e["epoch"] for e in history["train"]]
    val_epochs = [e["epoch"] for e in history["val"]]
    
    # ── 1. Overall Loss Curve ──
    plt.figure(figsize=(10, 6))
    train_loss = [e.get("regression_loss") or e.get("total_loss") for e in history["train"]]
    val_loss = [e.get("val/loss") for e in history["val"]]
    
    plt.plot(train_epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(val_epochs, val_loss, label="Validation Loss", marker="x")
    plt.title("Overall Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / "loss_overall.png", dpi=150)
    plt.close()
    
    # ── 2. Per-Head Validation Loss ──
    plt.figure(figsize=(12, 8))
    # find all keys that are per-head val losses
    all_keys = set()
    for e in history["val"]:
        all_keys.update(e.keys())
    
    head_keys = [k for k in all_keys if k.startswith("val/loss_")]
    if head_keys:
        for k in head_keys:
            head_val = [e.get(k, float("nan")) for e in history["val"]]
            head_name = k.replace("val/loss_", "")
            plt.plot(val_epochs, head_val, label=f"{head_name}", marker="s", markersize=4)
        
        plt.title("Per-Head Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / "loss_per_head.png", dpi=150)
        plt.close()

    # ── 3. Additional Metrics ──
    pearson = [e.get("val/overall/pearson_r") for e in history["val"]]
    if any(p is not None for p in pearson):
        plt.figure(figsize=(10, 6))
        plt.plot(val_epochs, pearson, label="Pearson R", color="green", marker="d")
        plt.title("Validation Pearson Correlation")
        plt.xlabel("Epoch")
        plt.ylabel("Pearson R")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(output_dir / "pearson_progression.png", dpi=150)
        plt.close()

    print(f"Saved plots to {output_dir}")

if __name__ == "__main__":
    main()
