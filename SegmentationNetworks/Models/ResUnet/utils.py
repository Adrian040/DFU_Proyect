import torch
import torchvision
import os
from dataset import MiccaiDataset
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
    train_ds = MiccaiDataset(
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

    val_ds = MiccaiDataset(
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

def save_predictions_as_imgs(loader, model, folder="output_assets_model/saved_images/", device="cuda"):

    if not os.path.exists(folder):
        os.makedirs(folder)
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}")  # para colab
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/target_{idx}.png")

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
    plt.savefig("output_assets_model/dice_loss_graph_ResUnet.png")
    print(f"Gráfica de Loss y Dice guardada en: output_assets_model/dice_loss_graph_ResUnet.png")
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

    test_ds = MiccaiDataset(
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


def get_preds_loader(image_dir, batch_size=4, image_height=240, image_width=240, num_workers=0, pin_memory=True):
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

    preds_ds = MiccaiDataset(
        image_dir=image_dir,
        transform=val_transforms,
    )

    preds_loader = DataLoader(
        preds_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return preds_loader