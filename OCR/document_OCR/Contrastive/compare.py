import torch

ssl = torch.load("/home/michel/tmp/IAM_contrastive/simclr_epoch101.pth", map_location="cpu")
#ssl = torch.load("/home/michel/dev/python/DAN/outputs/FCN_IAM_line_syn/checkpoints/last_216.pt", map_location="cpu", weights_only=False)
sup = torch.load("/home/michel/dev/python/DAN/outputs/FCN_IAM_line_syn/checkpoints/best_202.pt", map_location="cpu", weights_only=False)

# If they have a "state_dict"
ssl_sd = ssl["encoder_state_dict"] if "encoder_state_dict" in ssl else ssl
sup_sd = sup["encoder_state_dict"] if "encoder_state_dict" in sup else sup

common_keys = ssl_sd.keys() & sup_sd.keys()

for k in common_keys:
    w1 = ssl_sd[k].flatten()
    w2 = sup_sd[k].flatten()
    cos = torch.nn.functional.cosine_similarity(w1, w2, dim=0)
    l2  = torch.norm(w1 - w2).item()
    print(f"{k}: cosine={cos:.4f}, l2={l2:.4f}")
