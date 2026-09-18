"""
readTensorBoeard2.py

Inspect TensorBoard event files generated during training and validation for a
specific experiment run. This utility loads the event log, retrieves the
available metric tags, and prints scalar values for selected training metrics
such as loss or validation error. It is intended as a lightweight debugging aid
for monitoring experiment behavior and verifying that the expected metrics are
being recorded.

Typical use:
    - inspect available TensorBoard tags
    - read training or validation scalar values
    - debug experiment logs and metric naming conventions

This script is a diagnostic and analysis utility rather than a core model
training or evaluation component.
"""

from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_contrastive_line/results"
event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_contrastive_line_scale/results"
event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_contrastive_equal/results"
event_path = f"{Path.home()}/dev/python/DAN/outputs/IAM_contrastive_equal/results"
event_path = f"{Path.home()}/dev/python/DAN/outputs/IAM_contrastive_lineonly/results"

ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()  # Load the data


# List available tags (scalar, histogram, image, etc.)
print(ea.Tags())

# For example, get scalar data for 'train/loss'
scalars = ea.Scalars('train/loss_mean')

# Each entry is a namedtuple: (wall_time, step, value)
for s in scalars:  # show first 10
    print(f"Step {s.step}: {s.value}")

scalars = ea.Scalars('IAM-valid_cer')

# Each entry is a namedtuple: (wall_time, step, value)
for s in scalars:  # show first 10
    print(f"Step {s.step}: {s.value}")

scalars = ea.Scalars('IAM-valid_loss_ctc')

# Each entry is a namedtuple: (wall_time, step, value)
for s in scalars:  # show first 10
    print(f"Step {s.step}: {s.value}")

scalars = ea.Scalars('IAM-train_loss_ce')

# Each entry is a namedtuple: (wall_time, step, value)
for s in scalars:  # show first 10
    print(f"Step {s.step}: {s.value}")

scalars = ea.Scalars('IAM-valid_cer')

# Each entry is a namedtuple: (wall_time, step, value)
for s in scalars:  # show first 10
    print(f"Step {s.step}: {s.value}")