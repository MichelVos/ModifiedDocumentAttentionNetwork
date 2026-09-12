from tensorboard.backend.event_processing import event_accumulator

event_path = "/home/michel/dev/python/DAN/outputs/FCN_IAM_line_contrastive_line/results"
event_path = "/home/michel/dev/python/DAN/outputs/FCN_IAM_line_contrastive_line_scale/results"
event_path = "/home/michel/dev/python/DAN/outputs/FCN_IAM_line_contrastive_equal/results"
event_path = "/home/michel/dev/python/DAN/outputs/IAM_contrastive_equal/results"
event_path = "/home/michel/dev/python/DAN/outputs/IAM_contrastive_lineonly/results"

ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()  # Load the data


# List available tags (scalar, histogram, image, etc.)
print(ea.Tags())

# For example, get scalar data for 'train/loss'
scalars = ea.Scalars('train/loss_mean')

# Each entry is a namedtuple: (wall_time, step, value)
for s in scalars:  # show first 10
    print(f"Step {s.step}: {s.value}")
'''
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
'''