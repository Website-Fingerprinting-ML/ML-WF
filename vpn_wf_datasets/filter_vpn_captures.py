import os
import subprocess

input_folder = "./extra_vpn"
output_folder = "./extra_vpn_udp"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".pcap"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        command = [
            "tshark",
            "-r", input_path,
            "-Y", "!dns && !icmp",
            "-F", "pcap",
            "-w", output_path
        ]
        
        subprocess.run(command)
        print(f"Filtered {file} → {output_path}")

print("Done filtering all files.")
