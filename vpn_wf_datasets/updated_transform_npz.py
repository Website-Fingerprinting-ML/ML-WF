import os
import subprocess
import numpy as np
from collections import defaultdict

INPUT_FOLDER = "./extra_vpn_udp"
OUTPUT_FOLDER = "./vpn_npz_size"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def detect_client_ip(pcap_file):
    """
    Detect client IP using:
    1. First TCP SYN (no ACK) if available
    2. Otherwise fallback to packet size heuristic
    """
    cmd = [
        "tshark", "-r", pcap_file,
        "-T", "fields",
        "-E", "separator=\t",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "tcp.flags.syn",
        "-e", "tcp.flags.ack",
        "-e", "frame.len",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    size_stats = defaultdict(lambda: {"bytes_sent": 0, "count": 0})

    for line in proc.stdout:
        parts = (line.rstrip("\n").split("\t") + [""] * 5)[:5]
        src, dst, syn, ack, length = parts

        if not src or not dst:
            continue

        # 1. Detect TCP SYN without ACK (connection initiator)
        if syn == "1" and ack == "0":
            proc.kill()
            return src

        # 2. Collect size statistics (fallback)
        try:
            length = int(length)
        except ValueError:
            continue

        size_stats[src]["bytes_sent"] += length
        size_stats[src]["count"] += 1

    proc.wait()

    avg_sizes = {}
    for ip, stats in size_stats.items():
        if stats["count"] > 0:
            avg_sizes[ip] = stats["bytes_sent"] / stats["count"]

    if not avg_sizes:
        return None

    return min(avg_sizes, key=avg_sizes.get)


def convert_pcap_to_npz(pcap_path, output_path):
    client_ip = detect_client_ip(pcap_path)

    if client_ip is None:
        print(f"Could not determine client for {pcap_path}")
        return

    print(f"[{os.path.basename(pcap_path)}] Detected client IP: {client_ip}")

    cmd = [
        "tshark", "-r", pcap_path,
        "-T", "fields",
        "-E", "separator=\t",
        "-e", "frame.len",
        "-e", "ip.src",
        "-e", "ip.dst",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

    vals = []

    for line in proc.stdout:
        parts = (line.rstrip("\n").split("\t") + ["", "", ""])[:3]
        length_str, src, dst = parts

        if not length_str:
            continue

        try:
            length = float(length_str)
        except ValueError:
            continue

        if src == client_ip:
            vals.append(+length)
        elif dst == client_ip:
            vals.append(-length)

    proc.wait()

    if not vals:
        print(f"No packets matched detected client in {pcap_path}")
        return

    arr = np.asarray(vals, dtype=np.float64)
    np.savez_compressed(output_path, sizes=arr)

    print(f"Saved {arr.size} packets → {output_path}")


for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".pcap"):
        input_path = os.path.join(INPUT_FOLDER, file)

        base_name = os.path.splitext(file)[0]
        output_file = base_name + ".npz"
        output_path = os.path.join(OUTPUT_FOLDER, output_file)

        convert_pcap_to_npz(input_path, output_path)

print("Done processing all files.")
