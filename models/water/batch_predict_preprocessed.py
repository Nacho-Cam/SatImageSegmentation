import os
from PIL import Image
import torch
import numpy as np
from transformers import SamModel, SamProcessor

MODEL_DIR = r'c:\Users\nycca\OneDrive\Documentos\NACHO\Universidad\Curso 3\sistemasDePercepcionArtificialYVisionArtificial\Ejercicio_1\Proyecto final\SatImageSegmentation\models\water\checkpoints\SAM-water-hf'
IMAGES_DIR = r'c:\Users\nycca\OneDrive\Documentos\NACHO\Universidad\Curso 3\sistemasDePercepcionArtificialYVisionArtificial\Ejercicio_1\Proyecto final\SatImageSegmentation\data\raw\NEW2-AerialImageDataset\AerialImageDataset\test\images_preprocessed'
OUTPUT_DIR = r'c:\Users\nycca\OneDrive\Documentos\NACHO\Universidad\Curso 3\sistemasDePercepcionArtificialYVisionArtificial\Ejercicio_1\Proyecto final\SatImageSegmentation\models\water\outputs_pred'
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = SamModel.from_pretrained(MODEL_DIR)
processor = SamProcessor.from_pretrained(MODEL_DIR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

for fname in os.listdir(IMAGES_DIR):
    if not fname.lower().endswith('.png'):
        continue
    img_path = os.path.join(IMAGES_DIR, fname)
    img = Image.open(img_path)
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.pred_masks
        while pred.ndim > 4:
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
    out_name = os.path.splitext(fname)[0] + '_mask_pred.png'
    out_path = os.path.join(OUTPUT_DIR, out_name)
    Image.fromarray(mask_pred_bin).save(out_path)
    print(f'Saved predicted mask: {out_path}')
