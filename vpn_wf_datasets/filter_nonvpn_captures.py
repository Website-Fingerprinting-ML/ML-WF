import os
import subprocess

input_folder = "./nonvpn_UNfiltered"
output_folder = "./nonvpn_captures_filtered"

os.makedirs(output_folder, exist_ok=True)

# TCP-only, remove retransmissions / dup-acks / common analysis "badness"
DISPLAY_FILTER = (
    "tcp"
    " and not tcp.analysis.retransmission"
    " and not tcp.analysis.fast_retransmission"
    " and not tcp.analysis.spurious_retransmission"
    " and not tcp.analysis.duplicate_ack"
    " and not tcp.analysis.lost_segment"
    " and not tcp.analysis.out_of_order"
    " and not tcp.analysis.previous_segment_not_captured"
)

for file in os.listdir(input_folder):
    if not file.endswith(".pcap"):
        continue

    input_path = os.path.join(input_folder, file)
    output_path = os.path.join(output_folder, file)

    command = [
        "tshark",
        "-r", input_path,
        "-Y", DISPLAY_FILTER,
        "-F", "pcap",
        "-w", output_path
    ]

    res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if res.returncode != 0:
        print(f"[error] {file}: tshark failed\n{res.stderr}")
        continue

    print(f"Filtered {file} → {output_path}")

print("Done filtering all files.")
