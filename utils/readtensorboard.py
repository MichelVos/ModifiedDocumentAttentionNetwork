"""
readtensorboard.py

Inspect TensorBoard event files generated during model training and validation.
This utility loads an event log, reloads the recorded scalars, and prints the
available metric tags together with their recorded values. It is intended for
quick debugging and experimental analysis, allowing users to inspect training
curves, validation metrics, and loss evolution without launching the full
TensorBoard interface.

Typical use:
    - inspect available metrics in a training run
    - read scalar values for loss and validation error
    - debug the evolution of experiment logs during training

This script is a diagnostic and analysis utility rather than a core training or
evaluation component.
"""

from tensorboard.backend.event_processing import event_accumulator

# Path to your event file or directory containing event files
event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_non_syn/results"
#event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_non_syn.fail.original.head.no.aug/results"
#event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_non_syn.with.augmentations.fail/results"
event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_non_syn.small/results"
event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_non_syn.repaired std this is the correct one/results"
event_path = f"{Path.home()}/share/ou/documents/afstuderen/testresultaten/metlogging/IAM.origineel/FCN_IAM_line_syn/results"
event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line/results"
event_path = f"{Path.home()}/dev/python/DAN/outputs/dan_IAM_page_realtext20251115/results"
event_path = f"{Path.home()}/dev/python/DAN/outputs/dan_IAM_page_realtext20251115.Fail multiline on nov 25th/results"
event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_non_syn.small/results" 
event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_non_syn.tiny/results" 
event_path = f"{Path.home()}/dev/python/DAN/outputs/dan_IAM_page_real_20251129/results"
event_path = f"{Path.home()}/share/ou/documents/afstuderen/testresultaten/metlogging/IAM.no_pretraining/dan_IAM_page/results" 
event_path = f"{Path.home()}/dev/python/DAN/outputs/dan_IAM_page_real_20251201/results"
event_path = f"{Path.home()}/dev/python/DAN/outputs/FCN_IAM_line_contrastive_line/results"
#event_path = "{Path.home()}/share/ou/documents/afstuderen/testresultaten/metlogging/IAM.origineel/dan_IAM_page/results"
ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()  # Load the data


# List available tags (scalar, histogram, image, etc.)
print(ea.Tags())


# For example, get scalar data for 'train/loss'
scalars = ea.Scalars('IAM-train_loss_ctc')

# Each entry is a namedtuple: (wall_time, step, value)
for s in scalars:  # show first 10
    print(f"Step {s.step}: {s.value}")

scalars = ea.Scalars('IAM-valid_cer')

# Each entry is a namedtuple: (wall_time, step, value)
for s in scalars:  # show first 10
    print(f"Step {s.step}: {s.value}")
'''

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
'''