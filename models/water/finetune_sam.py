import os
import glob
from PIL import Image
import torch
from transformers import SamModel, SamProcessor
from tqdm import tqdm
import numpy as np
import random
import sys
import importlib.util
import cv2

RESUME_CHECKPOINT = 'models\water\checkpoints\SAM-water-hf\checkpoint_1200'  # e.g., 'models/water/checkpoints/SAM-water-hf/checkpoint_2000' or None for auto-latest

preproc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/NEW2-AerialImageDataset/AerialImageDataset/test/convert_tiff_to_jpg_and_preproc.py'))
spec = importlib.util.spec_from_file_location('convert_tiff_to_jpg_and_preproc', preproc_path)
preproc_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preproc_mod)
kaggle_preprocessing = preproc_mod.kaggle_preprocessing
github_preprocessing_pipeline = preproc_mod.github_preprocessing_pipeline
advanced_preprocessing_pipeline = preproc_mod.advanced_preprocessing_pipeline

# --- Preprocessing function for images and masks ---
def preprocess_image_and_mask(image_path, mask_path, target_size=(1024, 1024)):
    # Load and resize image
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size, Image.BILINEAR)
    # Normalize image to [0, 1]
    image_np = np.array(image).astype(np.float32) / 255.0
    # Optionally, use mean/std normalization (SAM default is ImageNet)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - imagenet_mean) / imagenet_std
    image = Image.fromarray(np.clip((image_np * 255), 0, 255).astype(np.uint8))
    # Load and resize mask
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize(target_size, Image.NEAREST)
    mask_np = np.array(mask)
    return image, mask_np

# --- Preprocessing pipelines ---
def kaggle_preprocessing(img, target_size=(1024, 1024)):
    img = img.resize(target_size, Image.BILINEAR)
    img_np = np.array(img)
    img_np = img_np / 255.0
    img_clahe = np.zeros_like(img_np)
    for i in range(3):
        channel = (img_np[..., i] * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe[..., i] = clahe.apply(channel) / 255.0
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_clahe - imagenet_mean) / imagenet_std
    img_out = np.clip((img_norm * 255), 0, 255).astype(np.uint8)
    return Image.fromarray(img_out)

def github_preprocessing_pipeline(img, target_size=(1024, 1024)):
    img_gray = img.convert('L').resize(target_size, Image.BILINEAR)
    img_np = np.array(img_gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_np)
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    _, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin_rgb = np.stack([img_bin]*3, axis=-1)
    return Image.fromarray(img_bin_rgb)

def advanced_preprocessing_pipeline(img, target_size=(1024, 1024)):
    img_cv = np.array(img.convert('RGB').resize(target_size, Image.BILINEAR))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_clahe = clahe.apply(v)
    hsv_clahe = cv2.merge([h, s, v_clahe])
    preproc = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2RGB)
    return Image.fromarray(preproc)

def stack_preprocessing_variants(image_path, target_size=(1024, 1024)):
    """
    Loads an image and returns a 15-channel numpy array stacking:
    - original RGB
    - kaggle preprocessed RGB
    - github preprocessed RGB
    - advanced preprocessed RGB
    - original normalized RGB (ImageNet mean/std)
    All resized to target_size.
    """
    img = Image.open(image_path).convert('RGB').resize(target_size, Image.BILINEAR)
    # 1. Original
    arr_original = np.array(img)
    # 2. Kaggle
    arr_kaggle = np.array(kaggle_preprocessing(img, target_size=target_size))
    # 3. GitHub
    arr_github = np.array(github_preprocessing_pipeline(img, target_size=target_size))
    # 4. Advanced
    arr_advanced = np.array(advanced_preprocessing_pipeline(img, target_size=target_size))
    # 5. Original normalized (ImageNet mean/std)
    arr_norm = np.array(img).astype(np.float32) / 255.0
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    arr_norm = (arr_norm - imagenet_mean) / imagenet_std
    arr_norm = np.clip((arr_norm * 255), 0, 255).astype(np.uint8)
    # Stack all (H, W, 15)
    stacked = np.concatenate([
        arr_original, arr_kaggle, arr_github, arr_advanced, arr_norm
    ], axis=-1)
    return stacked

def patch_first_conv(model, new_in_channels=15):
    import torch.nn as nn
    old_conv = model.vision_encoder.patch_embed.projection
    if old_conv.in_channels == new_in_channels:
        print(f'First conv already has {new_in_channels} channels, skipping patch.')
        return model
    print('Patching vision_encoder.patch_embed.projection for', new_in_channels, 'channels')
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        if new_in_channels > 3:
            new_conv.weight[:, 3:, :, :] = 0
        if old_conv.bias is not None:
            new_conv.bias = old_conv.bias
    model.vision_encoder.patch_embed.projection = new_conv
    return model

if __name__ == "__main__":
    # Download the dataset from Kaggle if not already present
    kaggle_dataset_path = os.path.join('data', 'raw', 'water_bodies')
    if not os.path.exists(kaggle_dataset_path):
        os.makedirs(kaggle_dataset_path, exist_ok=True)
        print('Downloading dataset from Kaggle...')
        os.system('kaggle datasets download -d franciscoescobar/satellite-images-of-water-bodies -p "data/raw/water_bodies"')
        # Unzip the dataset
        zip_files = [f for f in os.listdir(kaggle_dataset_path) if f.endswith('.zip')]
        for zip_file in zip_files:
            os.system(f'powershell.exe Expand-Archive -Path "{os.path.join(kaggle_dataset_path, zip_file)}" -DestinationPath "{kaggle_dataset_path}"')
        print('Dataset downloaded and extracted.')
    else:
        print('Dataset already exists. Skipping download.')

    # Path to the extracted dataset
    images_dir = os.path.join('data', 'raw', 'water_bodies', 'Water Bodies Dataset', 'Images')
    masks_dir = os.path.join('data', 'raw', 'water_bodies', 'Water Bodies Dataset', 'Masks')

    # Pair images and masks by filename (without extension)
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    mask_files = glob.glob(os.path.join(masks_dir, '*.jpg'))  # CHANGED from .png to .jpg

    image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    mask_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}

    paired = [(image_dict[k], mask_dict[k]) for k in image_dict if k in mask_dict]
    if not paired:
        print('No matching image/mask pairs found!')
        exit(1)

    print(f'Found {len(paired)} image/mask pairs.')

    # --- Use the full dataset for training/validation ---
    paired_subset = paired  # Use all pairs

    # Shuffle before splitting into train/val
    random.shuffle(paired_subset)

    # Shuffle and split into train/val
    train_size = int(0.85 * len(paired_subset))
    train_pairs = paired_subset[:train_size]
    val_pairs = paired_subset[train_size:]
    print(f"Using {len(train_pairs)} for training, {len(val_pairs)} for validation (out of {len(paired_subset)} total samples)")

    # --- Auto-resume from latest checkpoint if available ---
    import re
    checkpoint_base = os.path.join('models', 'water', 'checkpoints', 'SAM-water-hf')
    if RESUME_CHECKPOINT and os.path.isdir(RESUME_CHECKPOINT):
        print(f"Resuming from user-specified checkpoint: {RESUME_CHECKPOINT}")
        model = SamModel.from_pretrained(RESUME_CHECKPOINT, ignore_mismatched_sizes=True)
        processor = SamProcessor.from_pretrained(RESUME_CHECKPOINT)
    else:
        checkpoint_dirs = [d for d in os.listdir(checkpoint_base) if re.match(r'checkpoint_\d+', d)]
        if checkpoint_dirs:
            # Sort by number, descending
            checkpoint_dirs.sort(key=lambda x: int(re.findall(r'\d+', x)[0]), reverse=True)
            latest_checkpoint = os.path.join(checkpoint_base, checkpoint_dirs[0])
            print(f"Resuming from latest checkpoint: {latest_checkpoint}")
            model = SamModel.from_pretrained(latest_checkpoint, ignore_mismatched_sizes=True)
            processor = SamProcessor.from_pretrained(latest_checkpoint)
        else:
            print("No checkpoint found, starting from base model.")
            model = SamModel.from_pretrained('models/water/checkpoints/SAM-water-hf', ignore_mismatched_sizes=True)
            processor = SamProcessor.from_pretrained('models/water/checkpoints/SAM-water-hf')
    model = patch_first_conv(model, new_in_channels=15)
    # Update model config for new input channels
    if hasattr(model.config, 'num_channels'):
        model.config.num_channels = 15
    elif hasattr(model.config, 'in_channels'):
        model.config.in_channels = 15
    else:
        print('WARNING: Could not set input channel count in model config. You may need to update the config manually.')

    # Using a checkpoint helps: it allows the model to start from a previously learned state, speeding up convergence and improving results.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model device: {next(model.parameters()).device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    epochs = 2  # Now 2 epochs as requested
    batch_size = 1

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        model.train()
        running_loss = 0.0
        batch_images = []
        batch_masks = []
        for idx, (img_path, mask_path) in enumerate(tqdm(train_pairs, total=len(train_pairs))):
            stacked = stack_preprocessing_variants(img_path, target_size=(1024, 1024))
            mask_np = np.array(Image.open(mask_path).convert('L').resize((1024, 1024), Image.NEAREST))
            if np.sum(mask_np) == 0:
                continue
            batch_images.append(stacked.transpose(2, 0, 1))  # (C, H, W)
            batch_masks.append(mask_np)
            if len(batch_images) == batch_size or idx == len(train_pairs) - 1:
                mask_stack = np.stack([(m > 127).astype(np.float32) for m in batch_masks])
                mask_tensor = torch.from_numpy(mask_stack).unsqueeze(1).to(device)
                image_tensor = torch.from_numpy(np.stack(batch_images)).float().to(device) / 255.0
                optimizer.zero_grad()
                outputs = model(image_tensor)
                pred = outputs.pred_masks
                print(f"pred shape before mask selection: {pred.shape}")
                # Robustly handle 5D/4D outputs
                if pred.ndim == 5:
                    pred = pred[:, 0, 0:1, :, :]
                elif pred.ndim == 4 and pred.shape[1] > 1:
                    pred = pred[:, 0:1, :, :]
                print(f"pred shape after mask selection: {pred.shape}")
                if pred.shape != mask_tensor.shape:
                    pred = torch.nn.functional.interpolate(pred, size=mask_tensor.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(pred, mask_tensor)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batch_images = []
                batch_masks = []
                torch.cuda.empty_cache()
            # --- Save checkpoint every 100 files processed ---
            if (idx + 1) % 100 == 0:
                checkpoint_dir = os.path.join('models', 'water', 'checkpoints', 'SAM-water-hf', f'checkpoint_{idx+1}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                print(f"Saving checkpoint at {checkpoint_dir} (file {idx+1})...")
                model.save_pretrained(checkpoint_dir)
                processor.save_pretrained(checkpoint_dir)
        train_loss = running_loss / max(1, len(train_pairs)//batch_size)
        print(f'Epoch {epoch+1} train loss: {train_loss:.4f}')

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        batch_images = []
        batch_masks = []
        with torch.no_grad():
            for idx, (img_path, mask_path) in enumerate(val_pairs):
                stacked = stack_preprocessing_variants(img_path, target_size=(1024, 1024))
                mask_np = np.array(Image.open(mask_path).convert('L').resize((1024, 1024), Image.NEAREST))
                if np.sum(mask_np) == 0:
                    continue
                batch_images.append(stacked.transpose(2, 0, 1))  # (C, H, W)
                batch_masks.append(mask_np)
                if len(batch_images) == batch_size or idx == len(val_pairs) - 1:
                    mask_stack = np.stack([(m > 127).astype(np.float32) for m in batch_masks])
                    mask_tensor = torch.from_numpy(mask_stack).unsqueeze(1).to(device)
                    image_tensor = torch.from_numpy(np.stack(batch_images)).float().to(device) / 255.0
                    outputs = model(image_tensor)
                    pred = outputs.pred_masks
                    print(f"pred shape before mask selection: {pred.shape}")
                    if pred.ndim == 5:
                        pred = pred[:, 0, 0:1, :, :]
                    elif pred.ndim == 4 and pred.shape[1] > 1:
                        pred = pred[:, 0:1, :, :]
                    print(f"pred shape after mask selection: {pred.shape}")
                    if pred.shape != mask_tensor.shape:
                        pred = torch.nn.functional.interpolate(pred, size=mask_tensor.shape[-2:], mode='bilinear', align_corners=False)
                    loss = criterion(pred, mask_tensor)
                    val_loss += loss.item()
                    batch_images = []
                    batch_masks = []
        val_loss = val_loss / max(1, len(val_pairs)//batch_size)
        print(f'Epoch {epoch+1} val loss: {val_loss:.4f}')

    # Save the fine-tuned model
    save_dir = os.path.join('models', 'water', 'checkpoints', 'SAM-water-hf')
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print('Fine-tuning complete. Model saved as SAM-water-hf.')

    # --- Quick test after training ---
    import matplotlib.pyplot as plt

    # Pick 3 random images from the paired list
    sampled = random.sample(paired, min(3, len(paired)))
    for img_path, mask_path in sampled:
        img = Image.open(img_path).convert('RGB')
        stacked = stack_preprocessing_variants(img_path, target_size=(1024, 1024))
        image_tensor = torch.from_numpy(stacked.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            outputs = model(image_tensor)
            pred = outputs.pred_masks
            print(f"pred shape before mask selection: {pred.shape}")
            if pred.ndim == 5:
                pred = pred[:, 0, 0:1, :, :]
            elif pred.ndim == 4 and pred.shape[1] > 1:
                pred = pred[:, 0:1, :, :]
            print(f"pred shape after mask selection: {pred.shape}")
            # Squeeze to 4D
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
            mask_pred = (mask_pred > 0.5).astype('uint8') * 255
            # Find bounding box of predicted mask
            ys, xs = np.where(mask_pred > 0)
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                # Draw rectangle and label on a copy of the original image
                img_annot = img.copy()
                import PIL.ImageDraw, PIL.ImageFont
                draw = PIL.ImageDraw.Draw(img_annot)
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline='blue', width=3)
                # Draw label above the box
                label_y = max(y_min - 20, 0)
                draw.text((x_min, label_y), 'water', fill='blue')
            else:
                img_annot = img
            # Show original, mask, and annotated image
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(img)
            plt.title('Original')
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(mask_pred, cmap='gray', vmin=0, vmax=255)
            plt.title('Predicted Mask')
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(img_annot)
            plt.title('Detected Water')
            plt.axis('off')
            plt.show()