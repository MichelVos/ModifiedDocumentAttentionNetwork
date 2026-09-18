"""
compare.py

Compare two PyTorch model checkpoints to determine whether they contain identical
parameters. This utility loads the weights from two checkpoint files, extracts
the encoder state dictionaries when present, and iterates through each tensor to
check for differences. If a parameter is missing or has a different value, the
script reports the mismatch and prints a final summary.

Typical use:
    - verify whether two training checkpoints are identical
    - confirm reproducibility across runs
    - debug checkpoint updates or training restarts

This script is a lightweight debugging utility for model inspection rather than
a training or evaluation component.
"""

import torch
from pathlib import Path

file1 = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_MIMpage50perc_L1_random/checkpoints/best_8.pt"
file2 = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_MIMpage50perc_L1_random/checkpoints/last_9.pt"

w1 = torch.load(file1, map_location="cpu", weights_only=False)
w2 = torch.load(file2, map_location="cpu", weights_only=False)

# If checkpoints contain extra structure
if "encoder_state_dict" in w1:
    w1 = w1["encoder_state_dict"]
if "encoder_state_dict" in w2:
    w2 = w2["encoder_state_dict"]

all_equal = True

for key in w1:
    if key not in w2:
        print(f"Missing key in file2: {key}")
        all_equal = False
        continue

    if not torch.equal(w1[key], w2[key]):
        print(f"Different tensor: {key}")
        all_equal = False

print("Fully identical:", all_equal)
