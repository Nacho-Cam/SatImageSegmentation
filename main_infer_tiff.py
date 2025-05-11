import os
import glob
import random
from PIL import Image
import tifffile as tiff
import torch
from transformers import SamModel, SamProcessor
import numpy as np
import matplotlib.pyplot as plt

# Ensure imagecodecs is installed for LZW-compressed TIFFs
try:
    import imagecodecs
except ImportError:
    print('imagecodecs not found. Installing...')
    os.system('pip install imagecodecs')
    import imagecodecs

# Paths
# TIFF_IMAGE_PATH = 'data/raw/Sentinel2_9bandas_Madrid.tif'  # Example TIFF
IMG_FOLDER = 'data/raw/user_images'  # New folder for PNG/JPG uploads/screenshots
MODEL_DIRS = [
    'models/water/checkpoints/SAM-water-hf',
    # Add other model directories here (e.g., vegetation)
]

# --- Preprocessing function for inference images ---
def preprocess_infer_image(image, target_size=(256, 256)):
    # Resize and normalize as in training
    image = image.resize(target_size, Image.BILINEAR)
    image_np = np.array(image).astype(np.float32) / 255.0
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - imagenet_mean) / imagenet_std
    image = Image.fromarray(np.clip((image_np * 255), 0, 255).astype(np.uint8))
    return image

# Helper: process a batch of images (list of PIL.Image)
def process_with_sam_batch(pil_images, image_names):
    # Preprocess all images
    pil_images = [preprocess_infer_image(img) for img in pil_images]
    # Ensure all images are RGB
    pil_images = [img.convert('RGB') if img.mode != 'RGB' else img for img in pil_images]
    masks = []
    raw_masks = []
    mask_threshold = 0.3  # Lower threshold for more sensitivity
    for model_dir in MODEL_DIRS:
        print(f'Loading model from {model_dir}')
        model = SamModel.from_pretrained(model_dir)
        processor = SamProcessor.from_pretrained(model_dir)
        # Uncomment the following line to use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        # Batch process
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Input tensor device: {inputs['pixel_values'].device}")
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"Model {model_dir} output keys: {outputs.keys()}")
            if hasattr(outputs, 'pred_masks'):
                pred_masks = outputs.pred_masks.detach().cpu().numpy()
                for i in range(len(pil_images)):
                    mask = pred_masks[i]
                    if mask.ndim > 2:
                        mask = mask[0]
                    mask = np.squeeze(mask)
                    # Print mask stats before thresholding
                    print(f"[STATS] Image {i}: mask sum={np.sum(mask):.2f}, min={np.min(mask):.4f}, max={np.max(mask):.4f}, unique={np.unique(mask)}")
                    raw_masks.append(mask)
                    mask_bin = (mask > mask_threshold).astype('uint8') * 255
                    masks.append(mask_bin)
            else:
                print('No pred_masks in output, check model output structure.')
                masks.extend([np.zeros(pil_images[0].size[::-1], dtype='uint8')] * len(pil_images))
                raw_masks.extend([np.zeros(pil_images[0].size[::-1], dtype='float32')] * len(pil_images))
    return masks, raw_masks

def show_images_grid(images, title, is_mask=False):
    n = len(images)
    cols = 5
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if is_mask:
            if isinstance(img, np.ndarray) and img.ndim == 3:
                img = img[0]  # Take first channel if shape is (3, H, W)
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(img)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def show_images_grid_with_bbox(images, masks, title):
    import PIL.ImageDraw
    n = len(images)
    cols = 5
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    for i, (img, mask) in enumerate(zip(images, masks)):
        plt.subplot(rows, cols, i + 1)
        # Robustly squeeze and resize mask to match image size
        mask2d = None
        if isinstance(mask, np.ndarray):
            print(f"Image {i} mask unique values before squeeze: {np.unique(mask)}")
            mask = np.squeeze(mask)
            # If mask is not 2D, try to handle gracefully
            if mask.ndim == 2:
                mask2d = mask
            elif mask.ndim == 3:
                # Take first channel or slice
                mask2d = mask[0]
            elif mask.ndim == 1:
                # 1D mask, cannot reshape, skip
                print(f"Warning: mask {i} is 1D, skipping bounding box.")
                mask2d = np.zeros(img.size[::-1], dtype='uint8')
            else:
                print(f"Warning: mask {i} has unexpected shape {mask.shape}, skipping bounding box.")
                mask2d = np.zeros(img.size[::-1], dtype='uint8')
            # Ensure mask2d is 2D and same size as image
            if mask2d.shape != (img.size[1], img.size[0]):
                # Resize mask to image size
                mask_img = Image.fromarray(mask2d.astype('uint8'))
                mask_resized = mask_img.resize(img.size, Image.NEAREST)
                mask2d = np.array(mask_resized)
            print(f"Image {i} mask unique values after resize: {np.unique(mask2d)}")
        else:
            # Not a numpy array, try to convert
            try:
                mask2d = np.array(mask)
                if mask2d.shape != (img.size[1], img.size[0]):
                    mask_img = Image.fromarray(mask2d.astype('uint8'))
                    mask_resized = mask_img.resize(img.size, Image.NEAREST)
                    mask2d = np.array(mask_resized)
            except Exception as e:
                print(f"Warning: Could not process mask {i}: {e}")
                mask2d = np.zeros(img.size[::-1], dtype='uint8')
        ys, xs = np.where(mask2d > 0)
        img_annot = img.copy()
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            draw = PIL.ImageDraw.Draw(img_annot)
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline='blue', width=3)
            label_y = max(y_min - 20, 0)
            draw.text((x_min, label_y), 'water', fill='blue')
        else:
            print(f"No water detected in image {i}, no bounding box drawn.")
        plt.imshow(img_annot)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def show_images_grid_with_heatmap(images, raw_masks, title):
    n = len(images)
    cols = 5
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    for i, (img, raw_mask) in enumerate(zip(images, raw_masks)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        # Robustly squeeze and select channel if needed
        raw_mask = np.squeeze(raw_mask)
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[0]
        plt.imshow(raw_mask, cmap='jet', alpha=0.4, vmin=0, vmax=1)
        plt.axis('off')
    plt.suptitle(title + ' (Raw Mask Heatmap)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- Process 10 random images from EuroSAT dataset ---
# (DISABLED: EuroSAT dataset removed)
# EUROSAT_DIR = 'data/raw/EuroSAT/2750/'
# image_paths = []
# for root, dirs, files in os.walk(EUROSAT_DIR):
#     for file in files:
#         if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#             image_paths.append(os.path.join(root, file))
#
# if len(image_paths) < 10:
#     print(f"Not enough images found in {EUROSAT_DIR} (found {len(image_paths)}).")
#     selected_images = image_paths
# else:
#     selected_images = random.sample(image_paths, 10)
#
# original_imgs = []
# for img_file in selected_images:
#     pil_img = Image.open(img_file).convert('RGB')
#     original_imgs.append(pil_img)
#
# # Process images one at a time using GPU
# mask_imgs = []
# for img in original_imgs:
#     masks = process_with_sam_batch([img], ["dummy"])
#     mask_imgs.append(masks[0])
#
# # Show all original images in a single popup
# show_images_grid(original_imgs, 'Original Images')
# show_images_grid(mask_imgs, 'Processed Mask Images', is_mask=True)
# show_images_grid_with_bbox(original_imgs, mask_imgs, 'Detected Water Regions')

# --- Process 10 large images from Water Bodies Dataset ---
WATER_BODIES_DIR = 'data/raw/water_bodies/Water Bodies Dataset/Images/'
water_image_files = [
    os.path.join(WATER_BODIES_DIR, fname)
    for fname in sorted(os.listdir(WATER_BODIES_DIR))
    if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# Select 10 large images for inference
if len(water_image_files) < 10:
    print(f"Not enough images found in {WATER_BODIES_DIR} (found {len(water_image_files)}).")
    selected_water_images = water_image_files
else:
    selected_water_images = random.sample(water_image_files, 10)

water_imgs = []
for img_file in selected_water_images:
    pil_img = Image.open(img_file).convert('RGB')
    water_imgs.append(pil_img)

# Process images one at a time using GPU
water_mask_imgs = []
water_raw_masks = []
for img in water_imgs:
    masks, raw_masks = process_with_sam_batch([img], ["dummy"])
    water_mask_imgs.append(masks[0])
    water_raw_masks.append(raw_masks[0])

show_images_grid(water_imgs, 'Water Bodies Dataset: Original Images')
show_images_grid(water_mask_imgs, 'Water Bodies Dataset: Processed Mask Images', is_mask=True)
show_images_grid_with_bbox(water_imgs, water_mask_imgs, 'Water Bodies Dataset: Detected Water Regions')
show_images_grid_with_heatmap(water_imgs, water_raw_masks, 'Water Bodies Dataset')

# --- Process 10 large images from a generic folder in data/raw ---
LARGE_IMG_DIR = 'data/raw/large_water_images/'  # Place your large satellite images here
if not os.path.exists(LARGE_IMG_DIR):
    os.makedirs(LARGE_IMG_DIR)
    print(f"Created folder {LARGE_IMG_DIR}. Please add your large satellite images (JPG/PNG/TIFF) to this folder.")

large_image_files = [
    os.path.join(LARGE_IMG_DIR, fname)
    for fname in sorted(os.listdir(LARGE_IMG_DIR))
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
]

if len(large_image_files) < 1:
    print(f"No large images found in {LARGE_IMG_DIR}. Please add your satellite images.")
else:
    selected_large_images = random.sample(large_image_files, min(10, len(large_image_files)))
    large_imgs = []
    for img_file in selected_large_images:
        pil_img = Image.open(img_file).convert('RGB')
        large_imgs.append(pil_img)
    # Process images one at a time using GPU (no resizing, keep original size)
    def process_with_sam_batch_noscale(pil_images, image_names):
        # Use the same preprocessing as training/inference
        pil_images = [preprocess_infer_image(img) for img in pil_images]
        pil_images = [img.convert('RGB') if img.mode != 'RGB' else img for img in pil_images]
        masks = []
        raw_masks = []
        mask_threshold = 0.3  # Lower threshold for more sensitivity
        for model_dir in MODEL_DIRS:
            print(f'Loading model from {model_dir}')
            model = SamModel.from_pretrained(model_dir)
            processor = SamProcessor.from_pretrained(model_dir)
            # Uncomment the following line to use GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, 'pred_masks'):
                    pred_masks = outputs.pred_masks.detach().cpu().numpy()
                    for i in range(len(pil_images)):
                        mask = pred_masks[i]
                        if mask.ndim > 2:
                            mask = mask[0]
                        mask = np.squeeze(mask)
                        raw_masks.append(mask)
                        mask_bin = (mask > mask_threshold).astype('uint8') * 255
                        print(f"[TIFF DEBUG] Image {i} mask sum: {np.sum(mask_bin)}, unique: {np.unique(mask_bin)}")
                        masks.append(mask_bin)
                else:
                    masks.extend([np.zeros(pil_images[0].size[::-1], dtype='uint8')] * len(pil_images))
                    raw_masks.extend([np.zeros(pil_images[0].size[::-1], dtype='float32')] * len(pil_images))
        return masks, raw_masks
    large_mask_imgs = []
    large_raw_masks = []
    for img in large_imgs:
        masks, raw_masks = process_with_sam_batch_noscale([img], ["dummy"])
        large_mask_imgs.append(masks[0])
        large_raw_masks.append(raw_masks[0])
    show_images_grid(large_imgs, 'Large Satellite Images: Original')
    show_images_grid(large_mask_imgs, 'Large Satellite Images: Processed Mask', is_mask=True)
    show_images_grid_with_bbox(large_imgs, large_mask_imgs, 'Large Satellite Images: Detected Water Regions')
    show_images_grid_with_heatmap(large_imgs, large_raw_masks, 'Large Satellite Images')

# --- Process 10 images from NEW2-AerialImageDataset (train and test splits) ---
NEW2_TRAIN_IMG_DIR = 'data/raw/NEW2-AerialImageDataset/AerialImageDataset/train/images/'
NEW2_TEST_IMG_DIR = 'data/raw/NEW2-AerialImageDataset/AerialImageDataset/test/images/'
new2_train_image_files = [
    os.path.join(NEW2_TRAIN_IMG_DIR, fname)
    for fname in sorted(os.listdir(NEW2_TRAIN_IMG_DIR))
    if fname.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))
]
new2_test_image_files = [
    os.path.join(NEW2_TEST_IMG_DIR, fname)
    for fname in sorted(os.listdir(NEW2_TEST_IMG_DIR))
    if fname.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))
]
all_new2_image_files = new2_train_image_files + new2_test_image_files

if len(all_new2_image_files) < 1:
    print(f"No images found in NEW2-AerialImageDataset train/test folders.")
else:
    selected_new2_images = random.sample(all_new2_image_files, min(10, len(all_new2_image_files)))
    new2_imgs = []
    for img_file in selected_new2_images:
        pil_img = Image.open(img_file).convert('RGB')
        new2_imgs.append(pil_img)
    # Process images one at a time using GPU (no resizing, keep original size)
    def process_with_sam_batch_noscale(pil_images, image_names):
        # Use the same preprocessing as training/inference
        pil_images = [preprocess_infer_image(img) for img in pil_images]
        pil_images = [img.convert('RGB') if img.mode != 'RGB' else img for img in pil_images]
        masks = []
        raw_masks = []
        mask_threshold = 0.3  # Lower threshold for more sensitivity
        for model_dir in MODEL_DIRS:
            print(f'Loading model from {model_dir}')
            model = SamModel.from_pretrained(model_dir)
            processor = SamProcessor.from_pretrained(model_dir)
            # Uncomment the following line to use GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, 'pred_masks'):
                    pred_masks = outputs.pred_masks.detach().cpu().numpy()
                    for i in range(len(pil_images)):
                        mask = pred_masks[i]
                        if mask.ndim > 2:
                            mask = mask[0]
                        mask = np.squeeze(mask)
                        raw_masks.append(mask)
                        mask_bin = (mask > mask_threshold).astype('uint8') * 255
                        print(f"[TIFF DEBUG] Image {i} mask sum: {np.sum(mask_bin)}, unique: {np.unique(mask_bin)}")
                        masks.append(mask_bin)
                else:
                    masks.extend([np.zeros(pil_images[0].size[::-1], dtype='uint8')] * len(pil_images))
                    raw_masks.extend([np.zeros(pil_images[0].size[::-1], dtype='float32')] * len(pil_images))
        return masks, raw_masks
    new2_mask_imgs = []
    new2_raw_masks = []
    for img in new2_imgs:
        masks, raw_masks = process_with_sam_batch_noscale([img], ["dummy"])
        new2_mask_imgs.append(masks[0])
        new2_raw_masks.append(raw_masks[0])
    show_images_grid(new2_imgs, 'NEW2-AerialImageDataset (train+test): Original Images')
    show_images_grid(new2_mask_imgs, 'NEW2-AerialImageDataset (train+test): Processed Mask Images', is_mask=True)
    show_images_grid_with_bbox(new2_imgs, new2_mask_imgs, 'NEW2-AerialImageDataset (train+test): Detected Water Regions')
    show_images_grid_with_heatmap(new2_imgs, new2_raw_masks, 'NEW2-AerialImageDataset (train+test)')

    if len(new2_imgs) > 0:
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(new2_imgs[0])
        plt.title('NEW2 Example')
        plt.axis('off')
        # Try to load a training image
        try:
            from glob import glob
            train_img_path = glob('data/raw/water_bodies/Water Bodies Dataset/Images/*.jpg')[0]
            train_img = Image.open(train_img_path).convert('RGB')
            plt.subplot(1,2,2)
            plt.imshow(train_img)
            plt.title('Training Example')
            plt.axis('off')
            plt.suptitle('Visual Comparison: NEW2 vs Training')
            plt.show()
        except Exception as e:
            print(f"Could not load training image for comparison: {e}")

print('All models processed.')
# The original TIFF is never modified or lost.
