import os
import glob
import random
from PIL import Image
import tifffile as tiff
import torch
from transformers import SamModel, SamProcessor
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# Ensure imagecodecs is installed for LZW-compressed TIFFs
try:
    import imagecodecs
except ImportError:
    print('imagecodecs not found. Installing...')
    os.system('pip install imagecodecs')
    import imagecodecs

# Paths
IMG_FOLDER = 'data/raw/user_images'  # New folder for PNG/JPG uploads/screenshots
MODEL_DIRS = [
    'models/water/checkpoints/SAM-water-hf',
    # Add other model directories here (e.g., vegetation)
]

# Dynamically import stack_preprocessing_variants and patch_first_conv from finetune_sam.py
finetune_path = os.path.join(os.path.dirname(__file__), 'models/water/finetune_sam.py') if not os.path.exists('finetune_sam.py') else 'finetune_sam.py'
spec = importlib.util.spec_from_file_location('finetune_sam', finetune_path)
finetune_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune_mod)
stack_preprocessing_variants = finetune_mod.stack_preprocessing_variants
patch_first_conv = finetune_mod.patch_first_conv

def show_images_grid_with_bbox(images, masks, title):
    import PIL.ImageDraw
    import cv2
    n = len(images)
    cols = 5
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    for i, (img, mask) in enumerate(zip(images, masks)):
        plt.subplot(rows, cols, i + 1)
        # Robustly squeeze and resize mask to match image size
        mask2d = None
        if isinstance(mask, np.ndarray):
            mask = np.squeeze(mask)
            if mask.ndim == 2:
                mask2d = mask
            elif mask.ndim == 3:
                mask2d = mask[0]
            elif mask.ndim == 1:
                mask2d = np.zeros(img.size[::-1], dtype='uint8')
            else:
                mask2d = np.zeros(img.size[::-1], dtype='uint8')
            if mask2d.shape != (img.size[1], img.size[0]):
                mask_img = Image.fromarray(mask2d.astype('uint8'))
                mask_resized = mask_img.resize(img.size, Image.NEAREST)
                mask2d = np.array(mask_resized)
        else:
            try:
                mask2d = np.array(mask)
                if mask2d.shape != (img.size[1], img.size[0]):
                    mask_img = Image.fromarray(mask2d.astype('uint8'))
                    mask_resized = mask_img.resize(img.size, Image.NEAREST)
                    mask2d = np.array(mask_resized)
            except Exception as e:
                mask2d = np.zeros(img.size[::-1], dtype='uint8')
        # --- Merge close regions using dilation, then find largest region ---
        img_annot = img.copy()
        if np.any(mask2d > 0):
            mask_bin = (mask2d > 0).astype(np.uint8)
            # Dilation to merge close blobs (kernel size can be tuned)
            kernel = np.ones((25, 25), np.uint8)  # Increase for more merging
            mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dilated, connectivity=8)
            # Ignore background (label 0), find largest component
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                x, y, w, h, area = stats[largest_label]
                if area > 100:  # Ignore tiny regions
                    draw = PIL.ImageDraw.Draw(img_annot)
                    draw.rectangle([(x, y), (x + w - 1, y + h - 1)], outline='blue', width=6)
                    label_y = max(y - 20, 0)
                    draw.text((x, label_y), 'water', fill='blue')
        plt.imshow(img_annot)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- Show 10 random images from Water Bodies Dataset with bounding boxes ---
WATER_BODIES_DIR = 'data/raw/water_bodies/Water Bodies Dataset/Images/'
water_image_files = [
    os.path.join(WATER_BODIES_DIR, fname)
    for fname in sorted(os.listdir(WATER_BODIES_DIR))
    if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
]
if len(water_image_files) < 10:
    print(f"Not enough images found in {WATER_BODIES_DIR} (found {len(water_image_files)}).")
    selected_water_images = water_image_files
else:
    selected_water_images = random.sample(water_image_files, 10)
water_imgs = []
water_mask_imgs = []
device = torch.device('cpu')
#''''cuda' if torch.cuda.is_available() else''' 

# --- Model loading with checkpoint resume option ---
RESUME_CHECKPOINT = None  # e.g., 'models/water/checkpoints/SAM-water-hf/checkpoint_2000' or None for default
if RESUME_CHECKPOINT and os.path.isdir(RESUME_CHECKPOINT):
    print(f"Loading model from checkpoint: {RESUME_CHECKPOINT}")
    model = SamModel.from_pretrained(RESUME_CHECKPOINT, ignore_mismatched_sizes=True)
else:
    model = SamModel.from_pretrained('models/water/checkpoints/SAM-water-hf', ignore_mismatched_sizes=True)
model = patch_first_conv(model, new_in_channels=15)
model.to(device)
model.eval()

for img_file in selected_water_images:
    pil_img = Image.open(img_file).convert('RGB')
    water_imgs.append(pil_img)
    stacked = stack_preprocessing_variants(img_file, target_size=(1024, 1024))
    image_tensor = torch.from_numpy(stacked.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        outputs = model(image_tensor)
        pred = outputs.pred_masks
        while pred.ndim > 4:
            if pred.shape[2] == 1:
                pred = pred.squeeze(2)
            elif pred.shape[2] == 3:
                pred = pred[:, :, 0, :, :]
            else:
                pred = pred.squeeze(2)
        if pred.ndim == 4:
            pass
        elif pred.ndim == 3:
            pred = pred.unsqueeze(0)
        elif pred.ndim == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        if pred.shape[1] > 1:
            pred = pred[:, 0:1, :, :]
        mask_pred = pred[0, 0].cpu().numpy()
        mask_pred_bin = (mask_pred > 0.5).astype(np.uint8) * 255
    water_mask_imgs.append(mask_pred_bin)
show_images_grid_with_bbox(water_imgs, water_mask_imgs, 'Water Dataset: Detected Water Regions')

# --- Show 10 random images from Satellite Dataset (NEW2-AerialImageDataset) with bounding boxes ---
SAT_IMG_DIR = 'data/raw/NEW2-AerialImageDataset/AerialImageDataset/test/images/'
sat_image_files = [
    os.path.join(SAT_IMG_DIR, fname)
    for fname in sorted(os.listdir(SAT_IMG_DIR))
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
]
if len(sat_image_files) < 10:
    print(f"Not enough images found in {SAT_IMG_DIR} (found {len(sat_image_files)}).")
    selected_sat_images = sat_image_files
else:
    selected_sat_images = random.sample(sat_image_files, 10)
sat_imgs = []
sat_mask_imgs = []
for img_file in selected_sat_images:
    pil_img = Image.open(img_file).convert('RGB')
    sat_imgs.append(pil_img)
    stacked = stack_preprocessing_variants(img_file, target_size=(1024, 1024))
    image_tensor = torch.from_numpy(stacked.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        outputs = model(image_tensor)
        pred = outputs.pred_masks
        while pred.ndim > 4:
            if pred.shape[2] == 1:
                pred = pred.squeeze(2)
            elif pred.shape[2] == 3:
                pred = pred[:, :, 0, :, :]
            else:
                pred = pred.squeeze(2)
        if pred.ndim == 4:
            pass
        elif pred.ndim == 3:
            pred = pred.unsqueeze(0)
        elif pred.ndim == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        if pred.shape[1] > 1:
            pred = pred[:, 0:1, :, :]
        mask_pred = pred[0, 0].cpu().numpy()
        mask_pred_bin = (mask_pred > 0.5).astype(np.uint8) * 255
    sat_mask_imgs.append(mask_pred_bin)
show_images_grid_with_bbox(sat_imgs, sat_mask_imgs, 'Satellite Dataset: Detected Water Regions')

print('All models processed.')
