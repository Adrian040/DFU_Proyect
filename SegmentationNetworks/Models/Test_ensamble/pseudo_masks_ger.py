import json
import torch
import pandas as pd
import os
import numpy as np
import time
from PIL import Image
from main_models import UNET, ResUnet, AttentionUnet, UnetPlusPlus
from utils import get_preds_loader
from metrics import calculate_metrics

# ------------- Parámetros ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
# test_image_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/test_images"
# test_mask_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/test_masks"
raw_images_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/raw_images_68"
pred_masks_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/pred_masks_68_EnsembleArtColab"
# raw_image_dir = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/Images_Gerardo/raw_images_68/IM1.jpg"


# checkpoint = torch.load("output_assets_model/best_model_checkpoint.pth", weights_only=True)
# print("Checkpoint keys:", checkpoint["state_dict"].keys())

# model = UNET(in_channels=3, out_channels=1).to(DEVICE)
# print("Model keys:", model.state_dict().keys())


# checkpoint = torch.load("output_assets_model/best_model_checkpoint_ResUnet_125e.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
# model = ResUnet(in_channels=3, out_channels=1).to(DEVICE)    ## ------------ Aquí al cambiar de modelo -------------.
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()

checkpoint1 = torch.load("output_assets_model/best_model_checkpoint_ResUnetArtColab.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
model1 = ResUnet(in_channels=3, out_channels=1).to(DEVICE)    ## ------------ Aquí al cambiar de modelo -------------.
model1.load_state_dict(checkpoint1["state_dict"])
model1.eval()

checkpoint2 = torch.load("output_assets_model/best_model_checkpoint_UnetPlusPlusArtColab.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
model2 = UnetPlusPlus(in_channels=3, out_channels=1).to(DEVICE)    ## ------------ Aquí al cambiar de modelo -------------.
model2.load_state_dict(checkpoint2["state_dict"])
model2.eval()

loader = get_preds_loader(raw_images_dir, batch_size= batch_size,  image_height=240, image_width=240, num_workers=0, pin_memory=True) # Load images and masks

def save_predictions(loader, model1, model2, save_dir, device):
    model1.eval()
    model2.eval()
    os.makedirs(save_dir, exist_ok=True)
    total_segmented = 0
    timess = []
    with torch.no_grad():
        for idx, (x, _) in enumerate(loader):
            x = x.to(device)
            w_m1 = w_m1 = 0.50  # Para ResUnetArtColab + UnetPlusPlusArtColab (Usados en el ensamble usado en el artículo Italia).
            w_m2 = 1-w_m1

            start_time = time.time()
            preds1 = torch.sigmoid(model1(x))
            preds2 = torch.sigmoid(model2(x))
            preds = (w_m1*preds1 + w_m2*preds2)  # Promedio de las predicciones
            end_time = time.time()
            timess.append(end_time-start_time)
            preds = (preds > 0.5).float()
            
            for i in range(preds.shape[0]):
                pred = preds[i].cpu().numpy().squeeze()
                pred = (pred * 255).astype(np.uint8)
                pred_image = Image.fromarray(pred)

                # Obtener el nombre original de la imagen
                original_image_name = loader.dataset.images[idx * loader.batch_size + i]
                print(f"Image {original_image_name} saved")
                pred_image.save(os.path.join(save_dir, original_image_name))

                total_segmented += 1
        
    print(f"{total_segmented} predictions saved succesfully \n Saved on {pred_masks_dir} \n With a mean time prediction of: {np.mean(timess)}s.")

# Uso de la función
save_predictions(loader, model1, model2, pred_masks_dir, DEVICE)