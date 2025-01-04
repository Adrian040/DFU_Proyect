import json
import torch
import pandas as pd
from main import UNET #, ResUnet, etc.
from metrics import calculate_metrics

# ------------- Parámetros ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
img_size_for_test = 240
test_image_dir = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/test_images"
test_mask_dir = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/test_masks"

# ----- Cargamos el modelo entrenado con las mejores métricas ----------
checkpoint = torch.load("output_assets_model/best_model_checkpoint.pth", weights_only=True)  ## Nota: el argumento weights_only=True es para evitar el warning que indica que de esta forma se carga con mayor seguridad el modelo. Sin embargo no se están cargando otros datos como el optimizador. En resumen, esto es solo para quitar el warning pues en principio no hay datos maliciosos en la forma en que se guarda el modelo localmente.
model = UNET(in_channels=3, out_channels=4).to(DEVICE)    ## ------------ Aquí al cambiar de modelo -------------.
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# ----- Calculamos las métricas --------------

print("Calculating test metrics...")
dict_test_metrics = calculate_metrics(test_image_dir, test_mask_dir, model, num_classes=4, device=DEVICE, image_height=img_size_for_test, image_width=img_size_for_test)

# ----- Guardamos las métricas en un archivo .csv --------------
pd.DataFrame(dict_test_metrics).to_csv("output_assets_model/test_metrics.csv", index=False) # Sin índices.
# # Guardar las métricas en un archivo JSON
# with open("output_assets_model/test_metrics.json", "w") as outfile:
#     json.dump(test_metrics, outfile)

# ----- Imprimimos las métricas (opcional) --------------
print("Métricas calculadas para el test set:")
print(pd.DataFrame(dict_test_metrics))


# ------------------- Comparación del cálculo de métricas de validación (desp. del entrenamiento) -------------------
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_masks"
print('========\n', 'Métricas de validación (después del entrenamiento, con el mejor estado del modelo)\n', '=====================')
dict_val_metrics = calculate_metrics(VAL_IMG_DIR, VAL_MASK_DIR, model,model, num_classes=4, device=DEVICE, image_height=img_size_for_test, image_width=img_size_for_test)
print(pd.DataFrame(dict_val_metrics))