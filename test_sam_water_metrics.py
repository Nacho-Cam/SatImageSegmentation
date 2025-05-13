import os
import random
import numpy as np
from PIL import Image
import torch
from transformers import SamModel, SamProcessor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
import importlib.util
from sklearn.metrics import jaccard_score

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return 2. * intersection / (np.sum(y_true) + np.sum(y_pred) + 1e-8)

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

# Dynamically import stack_preprocessing_variants and patch_first_conv from finetune_sam.py
finetune_path = os.path.join(os.path.dirname(__file__), 'models/water/finetune_sam.py') if not os.path.exists('finetune_sam.py') else 'finetune_sam.py'
spec = importlib.util.spec_from_file_location('finetune_sam', finetune_path)
finetune_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finetune_mod)
stack_preprocessing_variants = finetune_mod.stack_preprocessing_variants
patch_first_conv = finetune_mod.patch_first_conv

# Load model and patch for 15 channels
model = SamModel.from_pretrained(MODEL_DIR, ignore_mismatched_sizes=True)
model = patch_first_conv(model, new_in_channels=15)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Use 1024x1024 as in training
TARGET_SIZE = (1024, 1024)

y_true = []
y_pred = []

for img_path, mask_path in test_pairs:
    stacked = stack_preprocessing_variants(img_path, target_size=TARGET_SIZE)
    mask_np = np.array(Image.open(mask_path).convert('L').resize(TARGET_SIZE, Image.NEAREST))
    mask_bin = (mask_np > 127).astype(np.uint8)
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
        mask_pred_bin = (mask_pred > 0.5).astype(np.uint8)
    y_true.extend(mask_bin.flatten())
    y_pred.extend(mask_pred_bin.flatten())

# Metrics
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
iou = jaccard_score(y_true, y_pred, zero_division=0)
dice = dice_score(np.array(y_true), np.array(y_pred))

print("Confusion Matrix:\n", cm)
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"IoU (Jaccard): {iou:.4f}")
print(f"Dice Score: {dice:.4f}")

plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.show()
