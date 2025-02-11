import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from main_models import UnetPlusPlus, ResUnet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define a color palette for the 4 classes
palette = [
    0, 0, 0,        # Class 0: Black (Background)
    255, 0, 0,      # Class 1: Red (Fibrin)
    0, 255, 0,      # Class 2: Green (Granulation)
    0, 0, 255,      # Class 3: Blue (Callus)
]

def load_model(checkpoint_path, model_class):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model = model_class(in_channels=3, out_channels=4).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def predict_and_save(image_path, model1, model2, output_folder="output_assets_model/saved_images/"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # Store original size
    
    # Resize image to 240x240
    image = image.resize((240, 240), Image.Resampling.BILINEAR)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    w_1 = 0.68
    w_2 = 1 - w_1
    
    with torch.no_grad():
        preds1 = model1(image)
        preds2 = model2(image)
        preds_mean = w_1 * preds1 + w_2 * preds2
        preds = torch.softmax(preds_mean, dim=1)
        preds = torch.argmax(preds, dim=1).cpu().numpy()
    
    pred_img = Image.fromarray(preds[0].astype(np.uint8), mode='P')
    pred_img.putpalette(palette)
    
    # Resize prediction to original image size
    pred_img = pred_img.resize(original_size, Image.Resampling.NEAREST)
    
    pred_img.save('preds_w_aruco/prediction.png')

# Load models
model1 = load_model("output_assets_model/best_model_checkpoint_ResUnet.pth", ResUnet)
model2 = load_model("output_assets_model/best_model_checkpoint_Unet++2.pth", UnetPlusPlus)

# Example usage
predict_and_save("preds_w_aruco/Aruco3.png", model1, model2)
# Ejemplo de uso
# predict_and_save("C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/Models/Test_ensamble/preds_w_aruco/Aruco3.png", "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/Models/Test_ensamble/preds_w_aruco/Aruco3_calssified.png")