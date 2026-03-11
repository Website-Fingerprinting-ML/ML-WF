#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def pad_or_truncate_1d(x: np.ndarray, L: int, dtype) -> np.ndarray:
    x = np.asarray(x)
    n = int(x.shape[0])
    if n >= L:
        return x[:L].astype(dtype, copy=False)
    out = np.zeros((L,), dtype=dtype)
    out[:n] = x.astype(dtype, copy=False)
    return out


def infer_domain_and_class(stem: str) -> tuple[str, str]:
    # stem like: vpn_rdp_capture3  or nonvpn_skype-chat_capture12
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {stem}.npz")

    domain = parts[0]
    cls = parts[1]

    if domain not in {"vpn", "nonvpn"}:
        raise ValueError(f"Could not infer domain from {stem}.npz (got {domain})")

    return domain, cls


def load_1d(npz_path: Path, key: str) -> np.ndarray:
    d = np.load(npz_path, allow_pickle=False)
    if key not in d:
        raise KeyError(f"{npz_path} missing key '{key}'. Found keys: {list(d.keys())}")
    arr = d[key]
    if arr.ndim != 1:
        raise ValueError(f"{npz_path}:{key} must be 1D, got shape {arr.shape}")
    return arr


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build a combined time+size dataset (X, y) where y encodes both the "
            "traffic class (rdp/skype/youtube) and the VPN domain (vpn/nonvpn) as a "
            "single integer label: y = domain_id * num_classes + class_id. "
            "This produces 6 classes for 3 traffic types × 2 domains."
        )
    )
    ap.add_argument("--root", default=".", help="Root folder (your vpn_wf_datasets). Default=.")
    ap.add_argument("--out_name", required=True, help="Output filename stem (e.g., wf_vpn_L65536); .npz is appended")
    ap.add_argument("--out_dir", default="./datasets", help="Output directory for the .npz file. Default=./datasets")
    ap.add_argument("--L", type=int, default=65536, help="Per-channel length. X shape will be (N, 2*L). Default=65536")
    ap.add_argument("--time_key", default="times", help="Array key in time npz files (default: times)")
    ap.add_argument("--size_key", default="sizes", help="Array key in size npz files (default: sizes)")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="dtype for X")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    dir_time_vpn = root / "vpn_npz_time"
    dir_size_vpn = root / "vpn_npz_size"
    dir_time_non = root / "nonvpn_npz_time"
    dir_size_non = root / "nonvpn_npz_size"

    for p in [dir_time_vpn, dir_size_vpn, dir_time_non, dir_size_non]:
        if not p.exists():
            raise FileNotFoundError(f"Missing folder: {p}")

    dtype = np.float32 if args.dtype == "float32" else np.float64
    L = int(args.L)

    # Collect all time files, then find matching size file by name
    pairs: list[tuple[Path, Path]] = []

    def add_pairs(time_dir: Path, size_dir: Path):
        for tfile in sorted(time_dir.glob("*.npz")):
            sfile = size_dir / tfile.name
            if not sfile.exists():
                raise FileNotFoundError(f"Missing matching size npz for {tfile.name} in {size_dir}")
            pairs.append((tfile, sfile))

    add_pairs(dir_time_vpn, dir_size_vpn)
    add_pairs(dir_time_non, dir_size_non)

    if not pairs:
        raise RuntimeError("No .npz files found to build dataset.")

    # Infer per-sample labels
    labels_class = []
    labels_domain = []
    stems = []
    for tfile, _ in pairs:
        domain, cls = infer_domain_and_class(tfile.stem)
        labels_class.append(cls)
        labels_domain.append(domain)
        stems.append(tfile.stem)

    # Build combined label: y = domain_id * num_classes + class_id
    # Ordering: nonvpn=0, vpn=1  |  classes sorted alphabetically
    class_names = sorted(set(labels_class))   # e.g. ['rdp', 'skype-chat', 'youtube']
    domain_names = ["nonvpn", "vpn"]          # fixed order
    class_to_id = {c: i for i, c in enumerate(class_names)}
    domain_to_id = {d: i for i, d in enumerate(domain_names)}

    num_classes = len(class_names)            # e.g. 3
    # Combined labels 0..(2*num_classes - 1), e.g. 0..5 for 3 classes
    # nonvpn_rdp=0, nonvpn_skype=1, nonvpn_youtube=2, vpn_rdp=3, vpn_skype=4, vpn_youtube=5
    y = np.array(
        [domain_to_id[d] * num_classes + class_to_id[c]
         for d, c in zip(labels_domain, labels_class)],
        dtype=np.int64,
    )
    domain_arr = np.array([domain_to_id[d] for d in labels_domain], dtype=np.int64)

    combined_class_names = [f"{d}_{c}" for d in domain_names for c in class_names]

    # Build X: each row is [time_channel | size_channel], shape (N, 2*L)
    N = len(pairs)
    X = np.zeros((N, 2 * L), dtype=dtype)
    orig_len = np.zeros((N,), dtype=np.int64)
    out_paths = []

    for i, (tfile, sfile) in enumerate(pairs):
        t = load_1d(tfile, args.time_key)
        s = load_1d(sfile, args.size_key)

        if t.shape[0] != s.shape[0]:
            raise ValueError(f"Length mismatch for {tfile.name}: time={t.shape[0]} size={s.shape[0]}")

        orig_len[i] = int(t.shape[0])

        tL = pad_or_truncate_1d(t, L, dtype)
        sL = pad_or_truncate_1d(s, L, dtype)

        X[i, :L] = tL
        X[i, L:] = sL

        out_paths.append(str(tfile))

    # Save
    out_stem = args.out_name if not args.out_name.endswith(".npz") else args.out_name[:-4]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_stem}.npz"

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        domain=domain_arr,           # 0=nonvpn, 1=vpn  (kept for reference)
        classes=np.array(combined_class_names),   # 6 combined class names
        pure_classes=np.array(class_names),        # 3 traffic-type names
        domains=np.array(domain_names),
        paths=np.array(out_paths),
        L=np.int64(L),
        seq_len=np.int64(2 * L),
        orig_len=orig_len,
        stems=np.array(stems),
    )

    print(f"Wrote: {out_path}")
    print(f"X shape: {X.shape} dtype={X.dtype}  (use --seq_len {2 * L} in WFlib)")
    print(f"y shape: {y.shape}  num_combined_classes={2 * num_classes}")
    print(f"Combined classes: {combined_class_names}")
    for cname in combined_class_names:
        cid = combined_class_names.index(cname)
        print(f"  {cid}: {cname}  (n={(y == cid).sum()})")
    print(f"orig_len min/median/max: {orig_len.min()}/{int(np.median(orig_len))}/{orig_len.max()}")


if __name__ == "__main__":
    main()
