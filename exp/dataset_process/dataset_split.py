import numpy as np
import os
import random
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split

# Fixed seed
fix_seed = 2024
random.seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset",      type=str, required=True, help="Dataset name")
parser.add_argument("--use_stratify", type=str, default="True", help="Whether to use stratify")
args = parser.parse_args()

infile       = os.path.join("./datasets", f"{args.dataset}.npz")
dataset_path = os.path.join("./datasets",  args.dataset)
os.makedirs(dataset_path, exist_ok=True)

assert os.path.exists(infile), f"{infile} does not exist!"

print("loading...", infile)
data = np.load(infile, allow_pickle=True)
X = data["X"]
y = data["y"]

num_classes = len(np.unique(y))
assert num_classes == y.max() + 1, "Labels are not continuous"

N = len(y)
print(f"Total samples: {N}, num_classes: {num_classes}")

# Show per-class counts so you can spot imbalance
counts = Counter(y.tolist())
print("Per-class sample counts:")
for cls in sorted(counts):
    print(f"  class {cls}: {counts[cls]} samples")

min_class_count = min(counts.values())

# ── Split sizes: 70 / 15 / 15, but never exceed N ──────────────────────────
test_size  = max(num_classes, int(round(0.15 * N)))
valid_size = max(num_classes, int(round(0.15 * N)))
if test_size + valid_size >= N:
    test_size  = num_classes
    valid_size = num_classes
train_size = N - test_size - valid_size
if train_size <= 0:
    raise ValueError("Dataset too small for requested split sizes.")

print(f"Split sizes → train: {train_size}, valid: {valid_size}, test: {test_size}")

# ── First split: (train) vs (valid + test) ─────────────────────────────────
# Stratify only when every class has >= 2 samples
use_strat = (args.use_stratify == "True") and (min_class_count >= 2)
print(f"Stratify first split: {use_strat}")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=(test_size + valid_size),
    random_state=fix_seed,
    stratify=y if use_strat else None,
)

# ── Second split: (valid) vs (test) ────────────────────────────────────────
# Re-check min_class_count on y_temp; some classes may have only 1 member
# in the temp pool after the first split even when overall counts look fine.
min_temp = min(Counter(y_temp.tolist()).values())
use_strat_temp = use_strat and (min_temp >= 2)
if use_strat and not use_strat_temp:
    print(f"[warn] Disabling stratify for valid/test split: "
          f"least-populated class in temp pool has only {min_temp} sample(s).")

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp,
    test_size=test_size,
    random_state=fix_seed,
    stratify=y_temp if use_strat_temp else None,
)

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Valid: X={X_valid.shape}, y={y_valid.shape}")
print(f"Test:  X={X_test.shape},  y={y_test.shape}")

np.savez(os.path.join(dataset_path, "train.npz"), X=X_train, y=y_train)
np.savez(os.path.join(dataset_path, "valid.npz"), X=X_valid, y=y_valid)
np.savez(os.path.join(dataset_path, "test.npz"),  X=X_test,  y=y_test)
print("Split saved successfully.")
