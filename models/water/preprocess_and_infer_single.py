import os
import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
import torch
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints', 'SAM-water-hf')
IMAGES_DIR = r'C:\Users\nycca\OneDrive\Documentos\NACHO\Universidad\Curso 3\sistemasDePercepcionArtificialYVisionArtificial\Ejercicio_1\Proyecto final\SatImageSegmentation\data\raw\NEW2-AerialImageDataset\AerialImageDataset\test\images'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference image for histogram matching (from your training set)
REF_IMAGE_PATH = os.path.join(IMAGES_DIR, 'water_body_101.jpg')  # Change to a typical training image if needed
if os.path.exists(REF_IMAGE_PATH):
    ref_img = Image.open(REF_IMAGE_PATH).convert('RGB')
else:
    ref_img = None
    print('Reference image for histogram matching not found!')

def match_histogram_to_reference(new_img, ref_img):
    if ref_img is None:
        return new_img
    new_np = np.array(new_img)
    ref_np = np.array(ref_img)
    matched = match_histograms(new_np, ref_np, channel_axis=-1)
    return Image.fromarray(np.uint8(np.clip(matched, 0, 255)))

def preprocess_image(img, target_size=(256, 256)):
    img = img.convert('RGB').resize(target_size, Image.BILINEAR)
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

def main():
    # Process all TIFF images in the folder and save masks as PNG, with interactive feedback
    for fname in os.listdir(IMAGES_DIR):
        if not fname.lower().endswith(('.tif', '.tiff')):
            continue
        img_path = os.path.join(IMAGES_DIR, fname)
        img = Image.open(img_path)
        img_prep = preprocess_image(img)
        # Save preprocessed input for inspection
        preproc_out_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + '_preprocessed.png')
        img_prep.save(preproc_out_path)
        print(f'Saved preprocessed input: {preproc_out_path}')
        # Load model
        model = SamModel.from_pretrained(MODEL_DIR)
        processor = SamProcessor.from_pretrained(MODEL_DIR)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        inputs = processor(images=img_prep, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
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
            mask_pred_bin = (mask_pred > 0.5).astype(np.uint8) * 255
        # Show image, mask, and overlay for feedback
        # Upscale mask to original image size for overlay
        mask_pred_bin_up = Image.fromarray(mask_pred_bin).resize(img.size, Image.NEAREST)
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(mask_pred_bin_up, cmap='gray')
        plt.title('Predicted Mask (Upscaled)')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(img)
        plt.imshow(mask_pred_bin_up, cmap='jet', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        feedback = input("Is the prediction correct? (y/n): ").strip().lower()
        if feedback == 'n':
            out_name = os.path.splitext(fname)[0] + '_mask_wrong.png'
        else:
            out_name = os.path.splitext(fname)[0] + '_mask_correct.png'
        out_path = os.path.join(OUTPUT_DIR, out_name)
        Image.fromarray(mask_pred_bin).save(out_path)
        print(f'Saved mask: {out_path}')

if __name__ == '__main__':
    main()
