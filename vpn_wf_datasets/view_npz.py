import os
import numpy as np

TIME_FOLDER = "./nonvpn_npz_time"
SIZE_FOLDER = "./nonvpn_npz_size"


def load_array(path, key):
    try:
        d = np.load(path)
        if key in d:
            return d[key]
        else:
            return None
    except Exception:
        return None


def print_array_preview(arr, label):
    if arr is None:
        print(f"  {label}: MISSING")
        return

    print(f"  {label} shape: {arr.shape}")

    if arr.ndim == 1:
        print(f"  First row (1D array): {arr[:10]}")  # first 10 elements
    elif arr.ndim >= 2:
        print(f"  First row: {arr[0]}")
        print(f"  First column: {arr[:, 0]}")
    else:
        print(f"  Unsupported array dimensions: {arr.ndim}")


def main():
    if not os.path.isdir(TIME_FOLDER):
        print(f"[warn] TIME_FOLDER not found: {TIME_FOLDER}")
        return

    if not os.path.isdir(SIZE_FOLDER):
        print(f"[warn] SIZE_FOLDER not found: {SIZE_FOLDER}")
        return

    files = sorted(f for f in os.listdir(TIME_FOLDER) if f.endswith(".npz"))

    print("\n=== NPZ First Row/Column Preview ===\n")

    for f in files:
        time_path = os.path.join(TIME_FOLDER, f)
        size_path = os.path.join(SIZE_FOLDER, f)

        print(f"\n===== {f} =====")

        time_arr = load_array(time_path, "times")
        size_arr = load_array(size_path, "sizes") if os.path.exists(size_path) else None

        print_array_preview(time_arr, "TIME")
        print_array_preview(size_arr, "SIZE")

    print("\nDone.\n")


if __name__ == "__main__":
    main()