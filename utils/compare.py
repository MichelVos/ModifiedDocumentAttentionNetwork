import torch

file1 = "/home/michel/dev/python/DAN/outputs/FCN_IAM_MIMpage50perc_L1_random/checkpoints/best_8.pt"
file2 = "/home/michel/dev/python/DAN/outputs/FCN_IAM_MIMpage50perc_L1_random/checkpoints/last_9.pt"

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
