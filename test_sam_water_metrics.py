import os
import random
import numpy as np
from PIL import Image
import torch
from transformers import SamModel, SamProcessor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# Paths
MODEL_DIR = 'models/water/checkpoints/SAM-water-hf'
IMAGES_DIR = 'data/raw/water_bodies/Water Bodies Dataset/Images/'
MASKS_DIR = 'data/raw/water_bodies/Water Bodies Dataset/Masks/'

# Reference image for histogram matching (from your training set)
REF_IMAGE_PATH = os.path.join(IMAGES_DIR, 'water_body_101.jpg')  # Change to a typical training image
if os.path.exists(REF_IMAGE_PATH):
    ref_img = Image.open(REF_IMAGE_PATH).convert('RGB')
else:
    ref_img = None
    print('Reference image for histogram matching not found!')

# Pair images and masks by filename (without extension)
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')]
mask_files = [f for f in os.listdir(MASKS_DIR) if f.lower().endswith('.jpg')]
image_dict = {os.path.splitext(f)[0]: os.path.join(IMAGES_DIR, f) for f in image_files}
mask_dict = {os.path.splitext(f)[0]: os.path.join(MASKS_DIR, f) for f in mask_files}
paired = [(image_dict[k], mask_dict[k]) for k in image_dict if k in mask_dict]

COMPLETED_FILE = "completed.txt"
if os.path.exists(COMPLETED_FILE):
    with open(COMPLETED_FILE, "r") as f:
        completed = set(line.strip() for line in f)
else:
    completed = set()

if len(paired) < 100:
    print(f"Not enough pairs found: {len(paired)}")
    test_pairs = paired
else:
    test_pairs = random.sample(paired, 100)

# Load model
model = SamModel.from_pretrained(MODEL_DIR)
processor = SamProcessor.from_pretrained(MODEL_DIR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def match_histogram_to_reference(new_img, ref_img):
    if ref_img is None:
        return new_img
    new_np = np.array(new_img)
    ref_np = np.array(ref_img)
    matched = match_histograms(new_np, ref_np, channel_axis=-1)
    return Image.fromarray(np.uint8(np.clip(matched, 0, 255)))

# Preprocessing (same as training, but with histogram matching)
def preprocess_image(img, target_size=(256, 256)):
    img = img.convert('RGB').resize(target_size, Image.BILINEAR)
    # Histogram match to reference
    img = match_histogram_to_reference(img, ref_img)
    img_np = np.array(img).astype(np.float32) / 255.0
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - imagenet_mean) / imagenet_std
    img = Image.fromarray(np.clip((img_np * 255), 0, 255).astype(np.uint8))
    return img

def preprocess_mask(mask, target_size=(256, 256)):
    mask = mask.convert('L').resize(target_size, Image.NEAREST)
    mask_np = np.array(mask)
    mask_bin = (mask_np > 127).astype(np.uint8)
    return mask_bin

# Collect all predictions and ground truths
y_true = []
y_pred = []

for img_path, mask_path in test_pairs:
    img_name = os.path.basename(img_path)
    if img_name in completed:
        continue  # Skip already processed
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    img_prep = preprocess_image(img)
    mask_bin = preprocess_mask(mask)
    inputs = processor(images=img_prep, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to GPU if available
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.pred_masks
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
        mask_pred_bin = (mask_pred > 0.5).astype(np.uint8)
    # Flatten for metrics
    y_true.extend(mask_bin.flatten())
    y_pred.extend(mask_pred_bin.flatten())
    # Save progress
    with open(COMPLETED_FILE, "a") as f:
        f.write(f"{img_name}\n")

# Metrics
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("Confusion Matrix:\n", cm)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optional: plot confusion matrix
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], ['Background', 'Water'])
plt.yticks([0, 1], ['Background', 'Water'])
plt.colorbar()
plt.show()
