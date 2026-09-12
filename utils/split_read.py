import os
from PIL import Image
import numpy as np
import io
import cv2


training_set = 'valid'
input_dir = f"/home/michel/dev/python/formatted/RIMES_page/{training_set}"
output_dir = f"/home/michel/dev/python/formatted/RIMES_page_224/{training_set}"
patch_size = 224

os.makedirs(output_dir, exist_ok=True)

# create shard writer (10k samples per shard)


filecount = 0

for root, dirs, files in os.walk(input_dir):
    for fname in files:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(root, fname)
        print(f"Processing: {img_path}")

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        count = 0
        rawCount = 0
        for y in range(0, H - patch_size + 1, patch_size):
            for x in range(0, W - patch_size + 1, patch_size):
                rawCount += 1
                patch = img.crop((x, y, x + patch_size, y + patch_size))
                gray = np.array(patch.convert("L"))
                patch_np = np.array(patch.convert("L")) / 255.0

                ink_ratio = (patch_np < 0.75).mean()
                if ink_ratio < 0.05:
                    continue

                # gray = np.array(patch.convert("L")).astype(np.float32)

                # estimate background brightness
                bg = np.median(gray)

                # pixels significantly darker than background
                ink_mask = gray < (bg - 25)

                ink_ratio = ink_mask.mean()
                if ink_ratio < 0.05:
                    continue

                # percentage of VERY black pixels
                black_ratio = (patch_np < 0.1).mean()

                # reject mostly black patches
                if black_ratio > 0.40:
                    continue

                # detect half-black images
                row_black = (patch_np < 0.1).mean(axis=1)
                col_black = (patch_np < 0.1).mean(axis=0)

                # reject if many rows or columns are almost fully black
                if (row_black > 0.95).mean() > 0.25:
                    continue

                if (col_black > 0.95).mean() > 0.25:
                    continue

                # ----- EDGE DENSITY CHECK -----

                edges = cv2.Canny(gray, 50, 150)

                edge_ratio = (edges > 0).mean()

                # reject low-detail patches
                if edge_ratio < 0.01:
                    continue

                # encode patch to JPEG in memory
                buffer = io.BytesIO()
                patch.save(buffer, format="JPEG")
                img_bytes = buffer.getvalue()

                filename = f"{output_dir}/{filecount:07d}_{count:05d}.jpg"
                with open(filename, "wb") as f:
                    f.write(img_bytes)


                count += 1

        print(f"File {filecount}: {rawCount} raw patches, {count} accepted patches")

        filecount += 1
