import torch
import numpy as np
from PIL import Image
import os
import pandas as pd
from dataset import DFUTissueDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory = True,
):
    train_ds = DFUTissueDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = DFUTissueDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

# def save_predictions_as_imgs(loader, model, folder="output_assets_model/saved_images/", device="cuda"):

#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     model.eval()

#     # for idx, (x, y) in enumerate(loader):
#     #     x = x.to(device=device)
#     #     with torch.no_grad():
#     #         preds = model(x)
#     #         preds = torch.softmax(preds, dim=1)
#     #         preds = torch.argmax(preds, dim=1).unsqueeze(1).float()
#     #         print("shapee: ", preds[0].shape)
#     #     torchvision.utils.save_image(
#     #         preds, f"{folder}/pred_{idx}.png"
#     #     )
#     #     torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/target_{idx}.png")

#     # Definir la paleta de colores para las clases
#     palette = [
#         0, 0, 0,        # Clase 0 - Negro
#         255, 0, 0,      # Clase 1 - Rojo
#         0, 255, 0,      # Clase 2 - Verde
#         0, 0, 255,      # Clase 3 - Azul
#     ]

#     def save_pred_as_image(pred, path, palette):
#         """
#         Guarda la predicción como una imagen en modo 'P' con una paleta de colores.

#         Args:
#             pred (torch.Tensor): Tensor de tamaño (H, W) con valores de clase.
#             path (str): Ruta donde guardar la imagen.
#             palette (list): Lista de colores en formato [R, G, B, ...].
#         """
#         pred = pred.cpu().numpy().astype('uint8')  # Convertir a numpy
#         img = Image.fromarray(pred, mode='P')  # Crear imagen en modo 'P'
#         img.putpalette(palette)  # Asignar la paleta de colores
#         img.save(path)  # Guardar imagen

#     # Proceso completo
#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = model(x)  # Salidas del modelo [4, 4, 240, 240]
#             preds = torch.softmax(preds, dim=1)
#             preds = torch.argmax(preds, dim=1)  # De [4, 4, 240, 240] a [4, 240, 240]

#         # Guardar cada imagen en el batch
#         for i in range(preds.shape[0]):  # Iterar sobre las imágenes del batch
#             save_pred_as_image(preds[i], f"{folder}/pred_{idx}_{i}.png", palette)


#     model.train()

def save_predictions_as_imgs(loader, model, folder="output_assets_model/saved_images/", device="cuda"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    model.eval()

    # Define a color palette for the 4 classes
    palette = [
        0, 0, 0,        # Class 0: Black
        255, 0, 0,      # Class 1: Red
        0, 255, 0,      # Class 2: Green
        0, 0, 255,      # Class 3: Blue
    ]

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1).cpu().numpy()

        # Convert predictions to PIL Image and apply palette
        pred_img = Image.fromarray(preds[0].astype(np.uint8), mode='P')
        pred_img.putpalette(palette)

        # Convert target to PIL Image and apply palette
        target_img = Image.fromarray(y[0].cpu().numpy().astype(np.uint8), mode='P')
        target_img.putpalette(palette)

        pred_img.save(f"{folder}/pred_{idx}.png")
        target_img.save(f"{folder}/target_{idx}.png")

    model.train()



    


## Otros:

def replace_backslashes(input_string):
  """Replaces all backslashes '\' in a string with forward slashes '/'."""
  return input_string.replace("\\", "/")
# ex: a = replace_backslashes(r"C:\Users\user\Documents\file.txt")


def plot_dice_loss(L_dice_result, L_loss_result, show_plot=False):
    epochs = range(1, len(L_dice_result) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))    # Crear la figura y los ejes

    # Graficar la curva de Dice Score
    color = 'tab:blue'
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Dice Score', color=color)
    ax1.plot(epochs, L_dice_result, color=color, label=f'Dice Score. Max: {max(L_dice_result):.3f}', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    # Crear un segundo eje y para la Loss Function
    ax2 = ax1.twinx()  # Compartir el eje x
    color = 'tab:red'
    ax2.set_ylabel('Loss Function', color=color)
    ax2.plot(epochs, L_loss_result, color=color, label=f'Loss Function. Min: {min(L_loss_result):.3f}')
    ax2.tick_params(axis='y', labelcolor=color)
    # Agregar leyenda y título
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title('Curvas de Dice Score y Loss Function')

    # Guardado de la gráfica:
    plt.savefig("output_assets_model/dice_loss_graph.png")
    print(f"Gráfica de Loss y Dice guardada en: output_assets_model/dice_loss_graph.png")
    # Mostrar la gráfica (opcional):
    if show_plot:
        plt.grid(True)
        plt.show()


def get_test_loader(test_image_dir, test_mask_dir, batch_size=4, image_height=240, image_width=240, num_workers=0, pin_memory=True):
    val_transforms = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    test_ds = DFUTissueDataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        transform=val_transforms,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return test_loader


def concat_dicts_to_dataframe_reset_index(dict_list, reset_threshold=3):
    """
    Convierte una lista de diccionarios en DataFrames, los concatena en un único DataFrame,
    y reinicia el índice después de alcanzar el umbral especificado. En este caso el umbral 3 indica que el índice irá de 0 a 3 indicando cada una de las 4 clases (Cada 4 renglones es otra época).
    
    Args:
        dict_list (list): Lista de diccionarios. Cada diccionario tiene la estructura
                          {key: metrics[key].tolist() for key in metrics}.
        reset_threshold (int): Valor después del cual el índice se reinicia (por defecto 3).
    
    Returns:
        pd.DataFrame: DataFrame concatenado con el índice reiniciado según el umbral.
    """
    # Convertir cada diccionario en un DataFrame
    dataframes = [pd.DataFrame(d) for d in dict_list]
    
    # Concatenar todos los DataFrames
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    
    # Reiniciar el índice con el patrón especificado
    concatenated_df.index = [i % (reset_threshold + 1) for i in range(len(concatenated_df))]
    
    return concatenated_df
