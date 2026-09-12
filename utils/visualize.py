import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

sup_ckpt = torch.load("/home/michel/dev/python/DAN/outputs/FCN_IAM_line_non_syn/checkpoints/best.pt", 
    map_location="cpu",
    weights_only=False)
con_ckpt = torch.load("/home/michel/dev/python/DAN/outputs/IAM_contrastive_lineonly/best.pt", 
    map_location="cpu",
    weights_only=False)


#encoder_sd = state_dict["encoder_state_dict"]
#print(encoder_sd.keys())
'''

# example key – adjust to your model
weight = state_dict["encoder.cnn.conv1.weight"]  
# shape: [out_channels, in_channels, kH, kW]

# visualize first 16 filters
fig, axes = plt.subplots(4, 4, figsize=(6, 6))

for i, ax in enumerate(axes.flat):
    filt = weight[i]
    if filt.shape[0] == 3:
        filt = filt.mean(0)  # RGB → grayscale
    ax.imshow(filt.detach().numpy(), cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.show()
'''


#w = encoder_sd["init_blocks.0.conv1.weight"]
# shape: [out_channels, in_channels, kH, kW]

#n = min(16, w.shape[0])
#fig, axes = plt.subplots(4, 4, figsize=(6, 6))

#for i, ax in enumerate(axes.flat[:n]):
#    f = w[i]
#    if f.shape[0] > 1:
#        f = f.mean(0)  # grayscale
#    ax.imshow(f.detach().cpu(), cmap="gray")
#    ax.axis("off")

#plt.suptitle("init_blocks.0.conv1 filters")
#plt.tight_layout()
#plt.show()

def filter_stats(w):
    # w: [out, in, kH, kW]
    return {
        "mean_abs": w.abs().mean().item(),
        "l2": w.norm(p=2).item(),
        "max_abs": w.abs().max().item(),
    }

w_sup = sup_ckpt["encoder_state_dict"]["init_blocks.0.conv1.weight"]
w_con = con_ckpt["encoder_state_dict"]["init_blocks.0.conv1.weight"]
print("Supervised:", filter_stats(w_sup))
print("Contrastive:", filter_stats(w_con))