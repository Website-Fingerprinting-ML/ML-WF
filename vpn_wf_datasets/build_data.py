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
        description="Build one combined time+size dataset (X,y) saved under /pvcvolume/ML-WF/datasets."
    )
    ap.add_argument("--root", default=".", help="Root folder (your vpn_wf_datasets). Default=.")
    ap.add_argument("--out_name", required=True, help="Output filename stem (e.g., wf_time_size_L65536); .npz is appended if missing")
    ap.add_argument("--out_dir", default="/pvcvolume/ML-WF/datasets", help="Output directory. Default=/pvcvolume/ML-WF/datasets")
    ap.add_argument("--L", type=int, default=65536, help="Per-channel window length. X will have seq_len=2*L. Default=65536")
    ap.add_argument("--time_key", default="times", help="Key in time npz (default: times)")
    ap.add_argument("--size_key", default="sizes", help="Key in size npz (default: sizes)")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="dtype for X")
    # Sliding-window augmentation: generates multiple rows per capture file.
    # Each window of length L is shifted by `stride` packets from the previous.
    # With num_windows=10 and stride=L//4, a capture with 200k packets yields
    # up to 10 rows instead of 1, multiplying your effective dataset size.
    ap.add_argument("--num_windows", type=int, default=1,
                    help="Max sliding windows to extract per capture (default=1, i.e. no augmentation). "
                         "Set to e.g. 10 to get up to 10x more samples.")
    ap.add_argument("--stride", type=int, default=-1,
                    help="Stride in packets between windows (default=-1 → L//2, i.e. 50%% overlap)")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    # Your folders
    dir_time_vpn = root / "vpn_npz_time"
    dir_size_vpn = root / "vpn_npz_size"
    dir_time_non = root / "nonvpn_npz_time"
    dir_size_non = root / "nonvpn_npz_size"

    for p in [dir_time_vpn, dir_size_vpn, dir_time_non, dir_size_non]:
        if not p.exists():
            raise FileNotFoundError(f"Missing folder: {p}")

    dtype = np.float32 if args.dtype == "float32" else np.float64
    L = int(args.L)
    stride = args.stride if args.stride > 0 else max(1, L // 2)

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

    # Infer labels
    labels_class = []
    labels_domain = []
    for tfile, _ in pairs:
        domain, cls = infer_domain_and_class(tfile.stem)
        labels_class.append(cls)
        labels_domain.append(domain)

    class_names = sorted(set(labels_class))
    class_to_id = {c: i for i, c in enumerate(class_names)}
    domain_to_id = {"nonvpn": 0, "vpn": 1}  # fixed

    # Build X with sliding windows: each capture → up to num_windows rows
    rows_X:      list[np.ndarray] = []
    rows_y:      list[int]        = []
    rows_domain: list[int]        = []
    rows_orig:   list[int]        = []
    rows_paths:  list[str]        = []
    rows_stems:  list[str]        = []

    for (tfile, sfile), dom, cls in zip(pairs, labels_domain, labels_class):
        t = load_1d(tfile, args.time_key)
        s = load_1d(sfile, args.size_key)

        if t.shape[0] != s.shape[0]:
            raise ValueError(f"Length mismatch for {tfile.name}: time={t.shape[0]} size={s.shape[0]}")

        orig_n  = int(t.shape[0])
        y_val   = domain_to_id[dom] * len(class_names) + class_to_id[cls]
        dom_val = domain_to_id[dom]

        wins_added = 0
        for w in range(args.num_windows):
            start = w * stride
            if start >= orig_n:
                break  # no real data left; stop early

            t_win = t[start: start + L]
            s_win = s[start: start + L]

            row = np.zeros(2 * L, dtype=dtype)
            n = len(t_win)          # may be < L near end of capture
            row[:n]    = t_win.astype(dtype)
            row[L:L+n] = s_win.astype(dtype)

            rows_X.append(row)
            rows_y.append(y_val)
            rows_domain.append(dom_val)
            rows_orig.append(orig_n)
            rows_paths.append(str(tfile))
            rows_stems.append(f"{tfile.stem}_w{w}")
            wins_added += 1

        if wins_added == 0:
            # Capture is empty — still emit one zero-padded row so labels stay intact
            rows_X.append(np.zeros(2 * L, dtype=dtype))
            rows_y.append(y_val)
            rows_domain.append(dom_val)
            rows_orig.append(orig_n)
            rows_paths.append(str(tfile))
            rows_stems.append(f"{tfile.stem}_w0")

    X        = np.stack(rows_X)
    y        = np.array(rows_y,      dtype=np.int64)
    domain   = np.array(rows_domain, dtype=np.int64)
    orig_len = np.array(rows_orig,   dtype=np.int64)

    # Write output
    out_stem = args.out_name if not args.out_name.endswith(".npz") else args.out_name[:-4]
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_stem}.npz"

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        domain=domain,   # 0 nonvpn, 1 vpn  (kept for reference)
        classes=np.array(class_names),
        domains=np.array(["nonvpn", "vpn"]),
        paths=np.array(rows_paths),
        L=np.int64(L),
        seq_len=np.int64(2 * L),
        orig_len=orig_len,
        stems=np.array(rows_stems),
    )

    print(f"Wrote: {out_path}")
    print(f"X shape: {X.shape} dtype={X.dtype}  (use --seq_len {2*L} in WFlib)")
    print(f"y values: {sorted(set(rows_y))}  classes={class_names}")
    print(f"domain counts: nonvpn={(domain==0).sum()} vpn={(domain==1).sum()}")
    print(f"windows per capture — stride={stride} num_windows={args.num_windows}")
    print(f"orig_len min/median/max: {orig_len.min()}/{int(np.median(orig_len))}/{orig_len.max()}")


if __name__ == "__main__":
    main()