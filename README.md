# Website-Fingerprinting-&-Mockingbird-Defense-Library

<p align="center">
<img src=".\figures\wflib.jpg" height = "180" alt="" align=center />
<br><br>
</p>

The following is based on the instructions provided by the developers of the WFLIB:
"WFlib is a Pytorch-based open-source library for website fingerprinting attacks, intended for research purposes only.

Website fingerprinting is a type of network attack in which an adversary attempts to deduce which website a user is visiting based on encrypted traffic patterns, even without directly seeing the content of the traffic.

We provide a neat code base to evaluate 11 advanced DL-based WF attacks on multiple datasets. This library is derived from our ACM CCS 2024 paper. If you find this repo useful, please cite our paper."

The main scope of our project was to use a few of the attacker models to experiment with defended datasets, more specifically tested out with the Mockingbird Defense. We hope for future work, we extend a defensive model library on top of the pre-exisisting attacker model library.

```bibtex
@inproceedings{deng2024wflib,
  title={Robust and Reliable Early-Stage Website Fingerprinting Attacks via Spatial-Temporal Distribution Analysis},
  author={Deng, Xinhao and Li, Qi and Xu, Ke},
  booktitle={Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security},
  year={2024}
}
```

Contributions via pull requests are welcome and appreciated.

## WFlib Overview

The code library includes 11 DL-based website fingerprinting attacks.

| Attacks | Conference  | Paper | Code |
|----------|----------|----------|----------|
| AWF | NDSS 2018 | [Automated Website Fingerprinting through Deep Learning](https://arxiv.org/pdf/1708.06376) | [DLWF](https://github.com/DistriNet/DLWF) |
| DF | CCS 2018 | [Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3243734.3243768) | [df](https://github.com/deep-fingerprinting/df) |
| Tik-Tok | PETS 2019 | [Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks](https://petsymposium.org/popets/2020/popets-2020-0043.pdf) | [Tik_Tok](https://github.com/msrocean/Tik_Tok) |
| Var-CNN | PETS 2019 | [Var-CNN: A Data-Efficient Website Fingerprinting Attack Based on Deep Learning](https://arxiv.org/pdf/1802.10215) | [Var-CNN](https://github.com/sanjit-bhat/Var-CNN) |
| TF | CCS 2019 | [Triplet Fingerprinting: More Practical and Portable Website Fingerprinting with N-shot Learning](https://dl.acm.org/doi/pdf/10.1145/3319535.3354217) | [tf](https://github.com/triplet-fingerprinting/tf) |
| BAPM | ACSAC 2021 | [BAPM: Block Attention Profiling Model for Multi-tab Website Fingerprinting Attacks on Tor](https://dl.acm.org/doi/pdf/10.1145/3485832.3485891) | None |
| ARES | S&P 2023 | [Robust Multi-tab Website Fingerprinting Attacks in the Wild](https://arxiv.org/pdf/2501.12622) | [Multitab-WF-Datasets](https://github.com/Xinhao-Deng/Multitab-WF-Datasets) |
| RF | Security 2023 | [Subverting Website Fingerprinting Defenses with Robust Traffic Representation](https://www.usenix.org/system/files/sec23fall-prepub-621_shen-meng.pdf) | [RF](https://github.com/robust-fingerprinting/RF) |
| NetCLR | CCS 2023 | [Realistic Website Fingerprinting By Augmenting Network Trace](https://arxiv.org/pdf/2309.10147) | [Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces](https://github.com/SPIN-UMass/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces) |
| TMWF | CCS 2023 | [Transformer-based Model for Multi-tab Website Fingerprinting Attack](https://dl.acm.org/doi/abs/10.1145/3576915.3623107) | [TMWF](https://github.com/jzx-bupt/TMWF) |
| Holmes | CCS 2024 | [Robust and Reliable Early-Stage Website Fingerprinting Attacks via Spatial-Temporal Distribution Analysis](https://arxiv.org/pdf/2407.00918) | [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library)|


We implemented all attacks using the same framework (Pytorch) and a consistent coding style, enabling researchers to evaluate and compare existing attacks easily.

## Usage

### Install 

```sh
git clone git@github.com:imakelaa/ML-WF.git
pip install --user .
```

**Note**

- Python 3.8 is required.

### Datasets

```sh
mkdir datasets
```
- Download VPN/Non-VPN datasets ([VNAT_release_1.zip (36 GB)](https://www.ll.mit.edu/r-d/datasets/vpnnonvpn-network-application-traffic-dataset-vnat)) and unzip into `./vpn_wf_datasets/unzip_OG/`

The raw dataset contains 162 pcap files across 10 traffic classes (rdp, skype-chat, youtube, netflix, rsync, scp, sftp, ssh, vimeo, voip) in both VPN and non-VPN variants. To produce the final `.npz` dataset used for training, the following preprocessing pipeline was applied from inside the `vpn_wf_datasets/` directory:

**Step 1 — Filter non-VPN captures** (removes TCP noise):
```sh
# Reads from nonvpn_UNfiltered/, writes to nonvpn_captures_filtered/
python filter_nonvpn_captures.py
```
Applies a tshark display filter: TCP-only, strips retransmissions, fast retransmissions, spurious retransmissions, duplicate ACKs, lost segments, out-of-order packets, and previous-segment-not-captured flags.

**Step 2 — Filter VPN captures** (removes protocol noise):
```sh
# Reads from extra_vpn/, writes to extra_vpn_udp/
python filter_vpn_captures.py
```
Removes DNS and ICMP packets from VPN traffic (VPN tunneling includes mixed protocol overhead).

**Step 3 — Convert filtered pcaps to NPZ** (per-packet timestamps and sizes):

Using a pcap-parsing tool (e.g. `dpkt` or `scapy`), extract per-packet arrival timestamps and byte sizes from each filtered pcap. Save each capture as a `.npz` file with keys `times` (1D float array) and `sizes` (1D float array) into the appropriate folder:
```
vpn_npz_time/       vpn_npz_size/
nonvpn_npz_time/    nonvpn_npz_size/
```
After filtering, 3 traffic classes with sufficient packet counts were retained: **rdp**, **skype-chat**, **youtube** — 3 captures each for VPN and non-VPN (18 capture files total).

**Step 4 — Build the combined dataset**:
```sh
# Run from vpn_wf_datasets/
python build_data.py --root . --out_name new_wf_L65536 --L 65536 --num_windows 10
```
Pairs matching `time` and `size` npz files, concatenates them into a single `2*L = 131072`-length feature vector per window, and assigns a numeric label encoding both domain (VPN=1, non-VPN=0) and class. Labels are assigned as `domain_id * num_classes + class_id` (e.g., non-VPN rdp=0, VPN rdp=3). Sliding-window augmentation produces multiple samples per capture file.

**Step 5 — Split into train/valid/test**:
```sh
python exp/dataset_process/dataset_split.py --dataset new_wf_L65536
```

The resulting dataset `new_wf_L65536.npz` has the following composition:

| Traffic Class | VPN Samples | Non-VPN Samples | Total Samples | Label (non-VPN / VPN) |
| --- | --- | --- | --- | --- |
| rdp | 25 | 20 | 45 | 3 / 0 |
| skype-chat | 3 | 3 | 6 | 4 / 1 |
| youtube | 17 | 19 | 36 | 5 / 2 |
| **Total** | **45** | **42** | **87** | — |

- Each sample has shape `(131072,)` — the first 65536 values are packet timestamps, the next 65536 are packet sizes (zero-padded to length `L=65536`).
- The `y` label encodes both whether the traffic is VPN-tunneled and the application type.
- `compare_size_time_npz.py` can be used to verify that the `times` and `sizes` arrays for each capture have matching lengths before running `build_data.py`.
- `view_npz.py` can be used to inspect the first few values of each npz capture file.

OR

- Download TOR datasets ([link](https://zenodo.org/records/13732130)) and place it in the folder `./datasets`

| Datasets | # of monitored websites | # of instances | Intro |
| --- | --- | --- | --- |
| CW.npz | 95 | 105730 | Closed-world dataset. [Details](https://dl.acm.org/doi/pdf/10.1145/3243734.3243768)|
| OW.npz |  95 | 146446 | Open-world dataset. [Details](https://dl.acm.org/doi/pdf/10.1145/3243734.3243768) |
| WTF-PAD.npz | 95 | 105730 | Dataset with WTF-PAD defense. [Details](https://arxiv.org/pdf/1512.00524) |
| Front.npz |  95 | 95000 | Dataset with Front defense. [Details](https://www.usenix.org/system/files/sec20-gong.pdf) |
| Walkie-Talkie.npz |  100 | 90000 | Dataset with Walkie-Talkie defense. [Details](https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-wang-tao.pdf) |


- The extracted dataset is in npz format and contains two values: X and y. X represents the cell sequence, with values being the direction (e.g., 1 or -1) multiplied by the timestamp. y corresponds to the labels. Note that the input of some datasets consists only of direction sequences.

- Divide the dataset into training, validation, and test sets.

```sh
# For single-tab datasets
python exp/dataset_process/dataset_split.py --dataset CW
# For multi-tab datasets
python exp/dataset_process/dataset_split.py --dataset Closed_2tab --use_stratify False
```

### Training \& Evaluation

We provide all experiment scripts for WF attacks in the folder `./scripts/`. For example, you can reproduce the DF attack on the CW dataset by executing the following command.

```sh
bash scripts/DF.sh
```

## Contact
If you have any questions or suggestions, feel free to contact:

- [Pakhi Sinha](https://imakelaa.github.io/website/) (pasinha@ucsc.edu)
- [Mahyar Vahabi](https://mvahabi.github.io/portfolio/) (mvahabi@ucsc.edu)

## Acknowledgements

We would like to thank all the authors of the referenced papers and the original developers of the attacker model library.
