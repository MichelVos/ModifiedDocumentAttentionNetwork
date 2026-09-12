import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class PageDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ])
        # pages normalized to 1239×1771
        self.transform = T.Compose([
            T.Resize((1771, 1239)),   # (H, W)
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        img = self.transform(img)
        return img, "page"
        

class LineDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ])
        # lines normalized to 32×1239
        self.transform = T.Compose([
            T.Resize((32, 1239)),    # (H, W)
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        img = self.transform(img)
        return img, "line"
