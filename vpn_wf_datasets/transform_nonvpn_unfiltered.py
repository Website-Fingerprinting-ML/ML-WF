import os
import subprocess
import numpy as np
import re

NONVPN_INPUT_FOLDER = "./extra_vpn"
TIME_OUTPUT_FOLDER = "./vpn_npz_time"
SIZE_OUTPUT_FOLDER = "./vpn_npz_size"

os.makedirs(TIME_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SIZE_OUTPUT_FOLDER, exist_ok=True)

# Only filter: TCP
TCP_FILTER = "!dns && !icmp"


def _run_text(cmd, label: str) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed ({label})\ncmd: {' '.join(cmd)}\n"
            f"stderr:\n{res.stderr}"
        )
    return res.stdout


def _pick_one_ip(field: str) -> str | None:
    m = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", field or "")
    return m.group(0) if m else None


def detect_client_ip_first_tcp(pcap_path: str) -> str | None:
    return "10.117.1.1"


def extract_signed_time_and_size(pcap_path: str, client_ip: str):
    cmd = [
        "tshark",
        "-r", pcap_path,
        "-Y", TCP_FILTER,
        "-T", "fields",
        "-E", "separator=\t",
        "-e", "frame.time_epoch",
        "-e", "frame.len",
        "-e", "ip.src",
        "-e", "ip.dst",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None

    times = []
    sizes = []
    t0 = None

    for line in proc.stdout:
        t_str, len_str, src, dst = (line.rstrip("\n").split("\t") + ["", "", "", ""])[:4]
        if not t_str or not len_str:
            continue

        try:
            t = float(t_str)
            ln = float(len_str)
        except ValueError:
            continue

        if src == client_ip:
            if t0 is None:
                t0 = t
            rel = t - t0
            times.append(+rel)
            sizes.append(+ln)
        elif dst == client_ip:
            if t0 is None:
                t0 = t
            rel = t - t0
            times.append(-rel)
            sizes.append(-ln)

    proc.wait()

    return np.asarray(times, dtype=np.float64), np.asarray(sizes, dtype=np.float64)


def process_pcap(pcap_path: str) -> None:
    fname = os.path.basename(pcap_path)
    base = os.path.splitext(fname)[0]

    client_ip = detect_client_ip_first_tcp(pcap_path)
    if not client_ip:
        print(f"[{fname}] Could not detect client IP — skipping.")
        return

    t_arr, s_arr = extract_signed_time_and_size(pcap_path, client_ip)
    if t_arr.size == 0:
        print(f"[{fname}] No packets matched client IP — skipping.")
        return

    time_out = os.path.join(TIME_OUTPUT_FOLDER, base + ".npz")
    size_out = os.path.join(SIZE_OUTPUT_FOLDER, base + ".npz")

    np.savez_compressed(time_out, times=t_arr)
    np.savez_compressed(size_out, sizes=s_arr)

    print(f"[{fname}] client={client_ip} saved {t_arr.size} packets")


def main():
    if not os.path.isdir(NONVPN_INPUT_FOLDER):
        print(f"[warn] NONVPN_INPUT_FOLDER not found")
        return

    pcaps = sorted(
        f for f in os.listdir(NONVPN_INPUT_FOLDER)
        if f.endswith(".pcap") and "voip" not in f.lower()
    )

    for f in pcaps:
        p = os.path.join(NONVPN_INPUT_FOLDER, f)
        try:
            process_pcap(p)
        except Exception as e:
            print(f"[error] {f}: {e}")

    print("Done processing NONVPN (TCP only, VoIP excluded).")


if __name__ == "__main__":
    main()
