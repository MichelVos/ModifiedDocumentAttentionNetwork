import os
import numpy as np
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# -----------------------------
# Configuration
# -----------------------------

TAG = "READ_2016-valid_cer"
#TAG = "IAM-valid_cer"
#TAG = "RIMES-valid_cer"

RUNS =[
    { "path": "${HOME}/dev/python/DAN/outputs/READBAUTZEN_contrastive_pretrained/results", "description": "SimCLR Pretrain READ 2016", "label": "BautzenSimCLR", "seed1": "1" },
    #{ "path": "${HOME}/images/bautzen/bautzen/results", "description": "S1S1", "label": "Bautzen", "seed1": "1" },
    #{ "path": "${HOME}/images/bautzen/bautzen_read/results", "description": "S1S2", "label": "BautzenRead", "seed1": "1" },
    #{ "path": "${HOME}/images/bautzen/read/results", "description": "Random 1", "label": "Read", "seed1": "1" },
    { "path": "${HOME}/images/bautzen/random/results", "description": "Random 2", "label": "Random", "seed1": "0" }
]



# -----------------------------
# Helper
# -----------------------------

def load_scalar(logdir, tag):
    ea = EventAccumulator(logdir)
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        raise ValueError(
            f"Tag '{tag}' not found in {logdir}\n"
            f"Available tags: {ea.Tags()['scalars']}"
        )

    events = ea.Scalars(tag)

    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])

    return steps, values


# -----------------------------
# Load all runs
# -----------------------------

grouped = defaultdict(list)

for run in RUNS:
    try:
        steps, values = load_scalar(run["path"], TAG)
        grouped[run["seed1"]=="0"].append((steps, values))
    except Exception as e:
        print(f"Error loading {run['path']}: {e}")

# -----------------------------
# Process per seed1
# -----------------------------

for seed1, runs in sorted(grouped.items()):

    if len(runs) == 0:
        continue

    # use shortest length to align runs
    min_len = min(len(v) for _, v in runs)

    steps = runs[0][0][:min_len]

    curves = np.stack(
        [values[:min_len] for _, values in runs],
        axis=0
    )

    mean_curve = curves.mean(axis=0)

    auc = np.trapz(mean_curve, steps)

    best_idx = np.argmin(mean_curve)
    best_value = mean_curve[best_idx]
    best_epoch = steps[best_idx]

    print()
    print("=" * 60)
    print(f"seed1 = {seed1}")
    print("=" * 60)
    print(f"AUC           : {auc:.6f}")
    print(f"Lowest value  : {best_value:.6f}")
    print(f"Best epoch    : {best_epoch}")