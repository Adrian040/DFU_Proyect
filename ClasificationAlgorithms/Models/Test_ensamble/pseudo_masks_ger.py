"""Este script es para crear pseudo máscaras dada una carpeta de imágenes y el modelo de predicción."""
import json
import torch
import pandas as pd
import os
import sys
import numpy as np
import time
from PIL import Image
from main_models import UNET, ResUnet, UnetPlusPlus
from utils import get_preds_loader

# ------------- Parámetros ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
# test_image_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/test_images"
# test_mask_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/test_masks"
# raw_images_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/raw_images_68"
# pred_masks_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/pred_masks_4classes_68_EnsembleArt"
raw_images_dir = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_semisup_padded/unlabel_data_padded"
pred_masks_dir = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_semisup_padded/pseudo_masks"


checkpoint1 = torch.load("output_assets_model/best_model_checkpoint_ResUnet.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
model1 = ResUnet(in_channels=3, out_channels=4).to(DEVICE)    ## ------------ Aquí al cambiar de modelo -------------.
model1.load_state_dict(checkpoint1["state_dict"])
model1.eval()

checkpoint2 = torch.load("output_assets_model/best_model_checkpoint_Unet++2.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
model2 = UnetPlusPlus(in_channels=3, out_channels=4).to(DEVICE)    ## ------------ Aquí al cambiar de modelo -------------.
model2.load_state_dict(checkpoint2["state_dict"])
model2.eval()

loader = get_preds_loader(raw_images_dir, batch_size= batch_size,  image_height=240, image_width=240, num_workers=0, pin_memory=True) # Load images and masks

def save_predictions(loader, model1, model2, save_dir, device):
    model1.eval()
    model2.eval()
    w_1 = 0.68
    w_2 = 1 - w_1

    os.makedirs(save_dir, exist_ok=True)
    total_segmented = 0
    timess = []
    # Define a color palette for the 4 classes
    palette = [
        0, 0, 0,        # Class 0: Black (Background)
        255, 0, 0,      # Class 1: Red (Fibrin)
        0, 255, 0,      # Class 2: Green (Granulation, este es el que realmente es más rojizo)
        0, 0, 255,      # Class 3: Blue (Callus)
    ]
    with torch.no_grad():
        for idx, (x, _) in enumerate(loader):
            x = x.to(device)

            start_time = time.time()
            preds1 = model1(x)
            preds2 = model2(x)
            preds_mean = w_1*preds1 + w_2*preds2
            end_time = time.time()
            timess.append(end_time-start_time)
            preds = torch.softmax(preds_mean, dim=1)
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            
            for i in range(preds.shape[0]):
                pred = preds[i]
                pred = pred.astype(np.uint8)
                pred_image = Image.fromarray(pred, mode='P')
                pred_image.putpalette(palette)

                # Obtener el nombre original de la imagen (porque se revuelven los datos al cargarlos con el loader)
                original_image_name = loader.dataset.images[idx * loader.batch_size + i]
                original_image_name = original_image_name.replace(".jpg", ".png")
                print(f"Image {original_image_name} saved")
                pred_image.save(os.path.join(save_dir, original_image_name))

                total_segmented += 1
        
    print(f"{total_segmented} predictions saved succesfully \n Saved on {pred_masks_dir} \n With a mean time prediction of: {np.mean(timess)}s.")

# Uso de la función
save_predictions(loader, model1, model2, pred_masks_dir, DEVICE)