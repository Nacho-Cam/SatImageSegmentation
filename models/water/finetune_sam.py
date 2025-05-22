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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

RESUME_CHECKPOINT = 'models\water\checkpoints\SAM-water-hf\checkpoint_1200'  # e.g., 'models/water/checkpoints/SAM-water-hf/checkpoint_2000' or None for auto-latest

# --- STUBS for missing src.common.preprocessing ---
def stack_preprocessing_variants(image_path, target_size=(1024, 1024)):
    # Minimal stub: just load and resize the image, stack as 3 channels (RGB)
    img = Image.open(image_path).convert('RGB').resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    # If model expects more channels, pad with zeros
    if arr.shape[2] < 15:
        pad = np.zeros((arr.shape[0], arr.shape[1], 15 - arr.shape[2]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=2)
    return arr

def patch_first_conv(model, in_channels=15):
    # Minimal stub: do nothing, just return model
    return model

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

    # --- Add LoveDA water images/masks ---
    loveda_img_dir = os.path.join('data', 'raw', 'landcover', 'LoveDA', 'Images')
    loveda_mask_dir = os.path.join('data', 'raw', 'landcover', 'LoveDA', 'Masks')
    loveda_img_files = glob.glob(os.path.join(loveda_img_dir, '*.png'))
    loveda_mask_files = glob.glob(os.path.join(loveda_mask_dir, '*_mask.png'))
    for img_path, mask_path in zip(sorted(loveda_img_files), sorted(loveda_mask_files)):
        mask_np = np.array(Image.open(mask_path))
        if np.any(mask_np == 3):  # Water class in LoveDA
            paired.append((img_path, mask_path))

    random.shuffle(paired)

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
    if not os.path.exists(checkpoint_base):
        os.makedirs(checkpoint_base, exist_ok=True)
    def load_and_patch_model(checkpoint_path):
        config = SamModel.from_pretrained(checkpoint_path).config
        for cfg in [config, getattr(config, 'vision_config', None)]:
            if cfg is not None:
                for field in ['num_channels', 'in_channels', 'input_channels']:
                    setattr(cfg, field, 15)
        model = SamModel.from_pretrained(checkpoint_path, config=config, ignore_mismatched_sizes=True)
        model = patch_first_conv(model, in_channels=15)
        return model
    if RESUME_CHECKPOINT and os.path.isdir(RESUME_CHECKPOINT):
        print(f"Resuming from user-specified checkpoint: {RESUME_CHECKPOINT}")
        model = load_and_patch_model(RESUME_CHECKPOINT)
        processor = SamProcessor.from_pretrained(RESUME_CHECKPOINT)
    else:
        checkpoint_dirs = [d for d in os.listdir(checkpoint_base) if re.match(r'checkpoint_\d+', d)]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(re.findall(r'\d+', x)[0]), reverse=True)
            latest_checkpoint = os.path.join(checkpoint_base, checkpoint_dirs[0])
            print(f"Resuming from latest checkpoint: {latest_checkpoint}")
            model = load_and_patch_model(latest_checkpoint)
            processor = SamProcessor.from_pretrained(latest_checkpoint)
        else:
            print("No checkpoint found, starting from base model.")
            model = load_and_patch_model('models/water/checkpoints/SAM-water-hf')
            processor = SamProcessor.from_pretrained('models/water/checkpoints/SAM-water-hf')
    if hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder, 'config'):
        for field in ['num_channels', 'in_channels', 'input_channels']:
            setattr(model.vision_encoder.config, field, 15)
    # DEBUG: print all config fields
    print(f"DEBUG: model.config.input_channels: {getattr(model.config, 'input_channels', None)}")
    print(f"DEBUG: model.config.num_channels: {getattr(model.config, 'num_channels', None)}")
    print(f"DEBUG: model.config.in_channels: {getattr(model.config, 'in_channels', None)}")
    if hasattr(model.config, 'vision_config'):
        print(f"DEBUG: model.config.vision_config.input_channels: {getattr(model.config.vision_config, 'input_channels', None)}")
        print(f"DEBUG: model.config.vision_config.num_channels: {getattr(model.config.vision_config, 'num_channels', None)}")
        print(f"DEBUG: model.config.vision_config.in_channels: {getattr(model.config.vision_config, 'in_channels', None)}")
    if hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder, 'config'):
        print(f"DEBUG: model.vision_encoder.config.input_channels: {getattr(model.vision_encoder.config, 'input_channels', None)}")
        print(f"DEBUG: model.vision_encoder.config.num_channels: {getattr(model.vision_encoder.config, 'num_channels', None)}")
        print(f"DEBUG: model.vision_encoder.config.in_channels: {getattr(model.vision_encoder.config, 'in_channels', None)}")
        if hasattr(model.vision_encoder, 'patch_embed') and hasattr(model.vision_encoder.patch_embed, 'projection'):
            print(f"DEBUG: model.vision_encoder.patch_embed.projection.in_channels: {getattr(model.vision_encoder.patch_embed.projection, 'in_channels', None)}")

    # Using a checkpoint helps: it allows the model to start from a previously learned state, speeding up convergence and improving results.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model device: {next(model.parameters()).device}")

    # Freeze all parameters except the final mask decoder head for fastest training
    for name, param in model.named_parameters():
        if not ("mask_decoder" in name):
            param.requires_grad = False
    print("All model parameters except the mask decoder are frozen.")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    epochs = 3
    batch_size = 2

    best_val_acc = 0.0
    checkpoint_path = os.path.join('models', 'water', 'checkpoints', 'SAM-water-hf', 'sam_water_finetuned_best.pth')

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
                # Move tensors to the same device as the model
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
        train_loss = running_loss / max(1, len(train_pairs)//batch_size)
        print(f'Epoch {epoch+1} train loss: {train_loss:.4f}')

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_running_correct = 0
        val_running_total = 0
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
                    # Calculate accuracy (binary, threshold 0.5)
                    preds = (pred > 0.5).long()
                    valid = (mask_tensor != 255)
                    val_running_correct += (preds[valid] == mask_tensor[valid]).sum().item()
                    val_running_total += valid.sum().item()
                    batch_images = []
                    batch_masks = []
        val_loss = val_loss / max(1, len(val_pairs)//batch_size)
        val_acc = val_running_correct / val_running_total if val_running_total > 0 else 0.0
        print(f'Epoch {epoch+1} val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

        # --- Save best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'processor': processor,  # Save processor config if needed
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc
            }, checkpoint_path)
            print(f"Best model updated at epoch {epoch+1} (Val Pixel Accuracy: {val_acc:.4f})")

    # Save the fine-tuned model
    save_dir = os.path.join('models', 'water', 'checkpoints', 'SAM-water-hf')
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print('Fine-tuning complete. Model saved as SAM-water-hf.')

    # --- OUTPUT EXAMPLE: Save a prediction for a random validation image ---
    import matplotlib.pyplot as plt
    import random
    model.eval()
    with torch.no_grad():
        idx = random.randint(0, len(val_pairs) - 1)
        val_img_path, val_mask_path = val_pairs[idx]
        val_img = Image.open(val_img_path).convert('RGB')
        val_mask = np.array(Image.open(val_mask_path).convert('L').resize((1024, 1024), Image.NEAREST))
        stacked = stack_preprocessing_variants(val_img_path, target_size=(1024, 1024))
        val_img_tensor = torch.from_numpy(stacked.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        outputs = model(val_img_tensor)
        pred = outputs.pred_masks
        if pred.ndim == 5:
            pred = pred[:, 0, 0:1, :, :]
        elif pred.ndim == 4 and pred.shape[1] > 1:
            pred = pred[:, 0:1, :, :]
        if pred.shape != val_mask.shape:
            pred = torch.nn.functional.interpolate(pred, size=val_mask.shape[-2:], mode='bilinear', align_corners=False)
        pred_mask = (pred[0, 0].cpu().numpy() > 0.5).astype('uint8') * 255
        os.makedirs('outputs', exist_ok=True)
        plt.imsave('outputs/val_pred_mask.png', pred_mask, cmap='gray')
        plt.imsave('outputs/val_gt_mask.png', val_mask, cmap='gray')
        overlay = np.array(val_img)
        overlay_mask = (pred_mask > 0).astype(np.uint8) * 255
        overlay_img = Image.fromarray(overlay).convert('RGB')
        overlay_img.putalpha(Image.fromarray(overlay_mask))
        overlay_img.save('outputs/val_pred_overlay.png')
        print('Validation prediction and overlay saved to outputs/.')

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