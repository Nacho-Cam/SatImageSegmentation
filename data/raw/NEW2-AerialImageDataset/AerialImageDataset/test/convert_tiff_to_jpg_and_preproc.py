import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

TIFF_DIR = r'C:/Users/nycca/OneDrive/Documentos/NACHO/Universidad/Curso 3/sistemasDePercepcionArtificialYVisionArtificial/Ejercicio_1/Proyecto final/SatImageSegmentation/data/raw/NEW2-AerialImageDataset/AerialImageDataset/test/images'
TRAIN_DIR = r'C:/Users/nycca/OneDrive/Documentos/NACHO/Universidad/Curso 3/sistemasDePercepcionArtificialYVisionArtificial/Ejercicio_1/Proyecto final/SatImageSegmentation/data/raw/water_bodies/Water Bodies Dataset/Images'

# Comment out all batch preprocessing and saving functions
# PREPROC_DIR = ...
# os.makedirs(PREPROC_DIR, exist_ok=True)
# TRAIN_IMAGES_DIR = ...
# TRAIN_PREPROC_DIR = ...
# os.makedirs(TRAIN_PREPROC_DIR, exist_ok=True)
# REF_IMAGE_PATH = ...
# if os.path.exists(REF_IMAGE_PATH):
#     ref_img = Image.open(REF_IMAGE_PATH).convert('RGB').resize((256, 256), Image.BILINEAR)
# else:
#     ref_img = None
#     print('Reference image for histogram matching not found!')
# def preprocess_image(...):
#     ...
# def convert_and_save(...):
#     ...
# def convert_and_save_train(...):
#     ...
# SAMPLE_TRAIN_IMAGE = ...
# def show_preprocessing_example(...):
#     ...

def advanced_preprocessing_pipeline(img):
    # Convert PIL Image to OpenCV format
    img_cv = np.array(img.convert('RGB'))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    results = {}

    # 1. Original
    results['original'] = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # 2. Normalized H channel (HSV)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_norm = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX)
    results['canal_h_norm'] = cv2.merge([h_norm, h_norm, h_norm])

    # 3. Gaussian smoothing
    gauss = cv2.GaussianBlur(img_cv, (7, 7), 0)
    results['gauss'] = cv2.cvtColor(gauss, cv2.COLOR_BGR2RGB)

    # 4. Adaptive preprocessing (CLAHE on V channel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_clahe = clahe.apply(v)
    hsv_clahe = cv2.merge([h, s, v_clahe])
    preproc = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2RGB)
    results['preproc'] = preproc

    # 5. Binary image after Otsu thresholding on V channel
    _, bin_img = cv2.threshold(v_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['bin_otsu'] = cv2.merge([bin_img, bin_img, bin_img])

    # 6. Morphological processing (closing)
    kernel = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    results['morph'] = cv2.merge([morph, morph, morph])

    # 7. Contours detected
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = results['original'].copy()
    cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
    results['contours_detected'] = contour_img

    # 8. Contour analysis (min area rects and labels)
    analysis_img = contour_img.copy()
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 500:  # Ignore small contours (tune this threshold as needed)
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(analysis_img, [box], 0, (255,0,0), 2)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(analysis_img, f'#{i} A={int(area)}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    results['contours_analysis'] = analysis_img

    return results

def show_advanced_preprocessing_examples():
    test_files = [f for f in os.listdir(TIFF_DIR) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))][:5]
    for fname in test_files:
        img = Image.open(os.path.join(TIFF_DIR, fname))
        results = advanced_preprocessing_pipeline(img)
        fig, axs = plt.subplots(2, 4, figsize=(18, 9))
        keys = ['original', 'canal_h_norm', 'gauss', 'preproc', 'bin_otsu', 'morph', 'contours_detected', 'contours_analysis']
        for i, key in enumerate(keys):
            ax = axs[i//4, i%4]
            ax.imshow(results[key])
            ax.set_title(key)
            ax.axis('off')
        plt.suptitle(f'Preprocessing steps for {fname}')
        plt.tight_layout()
        plt.show()

def kaggle_preprocessing(img, target_size=(256, 256)):
    # Resize
    img = img.resize(target_size, Image.BILINEAR)
    # Convert to numpy array
    img_np = np.array(img)
    # Normalize to [0, 1]
    img_np = img_np / 255.0
    # Optionally, apply CLAHE to each channel (as in the Kaggle notebook)
    img_clahe = np.zeros_like(img_np)
    for i in range(3):
        channel = (img_np[..., i] * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe[..., i] = clahe.apply(channel) / 255.0
    # Standardize (mean/std from ImageNet)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_clahe - imagenet_mean) / imagenet_std
    # Convert back to uint8 for visualization
    img_out = np.clip((img_norm * 255), 0, 255).astype(np.uint8)
    return img_out

def show_kaggle_preprocessing_examples():
    test_files = [f for f in os.listdir(TIFF_DIR) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))][:5]
    for fname in test_files:
        img = Image.open(os.path.join(TIFF_DIR, fname)).convert('RGB')
        img_kaggle = kaggle_preprocessing(img)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img)
        axs[0].set_title('Original')
        axs[0].axis('off')
        axs[1].imshow(img_kaggle)
        axs[1].set_title('Kaggle Preprocessed')
        axs[1].axis('off')
        plt.suptitle(f'Kaggle Preprocessing for {fname}')
        plt.tight_layout()
        plt.show()

def show_kaggle_preprocessing_examples_train():
    train_files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))][:5]
    for fname in train_files:
        img = Image.open(os.path.join(TRAIN_DIR, fname)).convert('RGB')
        img_kaggle = kaggle_preprocessing(img)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img)
        axs[0].set_title('Original')
        axs[0].axis('off')
        axs[1].imshow(img_kaggle)
        axs[1].set_title('Kaggle Preprocessed')
        axs[1].axis('off')
        plt.suptitle(f'Kaggle Preprocessing for {fname} (train)')
        plt.tight_layout()
        plt.show()

def github_preprocessing_pipeline(img, target_size=(256, 256)):
    # Resize and convert to grayscale
    img_gray = img.convert('L').resize(target_size, Image.BILINEAR)
    img_np = np.array(img_gray)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_np)
    # Gaussian blur
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    # Otsu's thresholding
    _, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {
        'original': np.array(img.resize(target_size, Image.BILINEAR)),
        'grayscale': img_np,
        'clahe': img_clahe,
        'blur': img_blur,
        'binary': img_bin
    }

def show_github_preprocessing_examples_train_and_test():
    # Show for 5 train images
    print('Showing GitHub preprocessing for 5 SAM-water train images:')
    train_files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))][:5]
    for fname in train_files:
        img = Image.open(os.path.join(TRAIN_DIR, fname)).convert('RGB')
        results = github_preprocessing_pipeline(img)
        fig, axs = plt.subplots(1, 5, figsize=(18, 5))
        keys = ['original', 'grayscale', 'clahe', 'blur', 'binary']
        for i, key in enumerate(keys):
            ax = axs[i]
            if key == 'original':
                ax.imshow(results[key])
            else:
                ax.imshow(results[key], cmap='gray')
            ax.set_title(key)
            ax.axis('off')
        plt.suptitle(f'GitHub Preprocessing for {fname} (train)')
        plt.tight_layout()
        plt.show()
    # Show for 5 test images (new dataset)
    print('Showing GitHub preprocessing for 5 images from new dataset:')
    test_files = [f for f in os.listdir(TIFF_DIR) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))][:5]
    for fname in test_files:
        img = Image.open(os.path.join(TIFF_DIR, fname)).convert('RGB')
        results = github_preprocessing_pipeline(img)
        fig, axs = plt.subplots(1, 5, figsize=(18, 5))
        for i, key in enumerate(keys):
            ax = axs[i]
            if key == 'original':
                ax.imshow(results[key])
            else:
                ax.imshow(results[key], cmap='gray')
            ax.set_title(key)
            ax.axis('off')
        plt.suptitle(f'GitHub Preprocessing for {fname} (test)')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    show_github_preprocessing_examples_train_and_test()
