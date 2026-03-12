import os
import numpy as np

TIME_FOLDER = "./nonvpn_npz_time"
SIZE_FOLDER = "./nonvpn_npz_size"

def load_length(path, key):
    try:
        d = np.load(path)
        if key in d:
            return d[key].size
        else:
            return None
    except Exception:
        return None


def main():
    if not os.path.isdir(TIME_FOLDER):
        print(f"[warn] TIME_FOLDER not found: {TIME_FOLDER}")
        return

    if not os.path.isdir(SIZE_FOLDER):
        print(f"[warn] SIZE_FOLDER not found: {SIZE_FOLDER}")
        return

    files = sorted(f for f in os.listdir(TIME_FOLDER) if f.endswith(".npz"))

    print("\n=== NPZ Length Comparison ===\n")

    for f in files:
        time_path = os.path.join(TIME_FOLDER, f)
        size_path = os.path.join(SIZE_FOLDER, f)

        time_len = load_length(time_path, "times")
        size_len = load_length(size_path, "sizes") if os.path.exists(size_path) else None

        print(f"{f}")
        print(f"  time length : {time_len}")
        print(f"  size length : {size_len}")

        if time_len is not None and size_len is not None:
            if time_len == size_len:
                print("  ✓ MATCH\n")
            else:
                print("  ✗ MISMATCH\n")
        else:
            print("  ⚠ Missing data in one folder\n")

    print("Done.\n")


if __name__ == "__main__":
    main()