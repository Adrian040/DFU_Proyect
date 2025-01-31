import json
import os
import pandas as pd
import zipfile
import torch
import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time
import torch.optim as optim
from main import ResUnet
from metrics import check_metrics, dice_loss, calculate_metrics
from utils import save_predictions_as_imgs, load_checkpoint, get_loaders, plot_dice_loss


# ------------------- Parámetros de entrenamiento --------------------

NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",DEVICE, "is available \n ----------------------")
BATCH_SIZE = 4
NUM_WORKERS = 0
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False    # True if you want to load a pre-trained model
SAVE_IMS = True
SAVE_MODEL = True  # ! IMPORTANTE: debe esta en True para guardar el modelo y sus datos.
DROPOUT_P = 0.30  # Dropout probability (0.0 to not implement dropout)
PATIENCE = 50  # for Early stopping (set big to avoid it)
TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/train_images"
TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/train_masks"
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/val_images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/val_masks"


#------------------- Funciones de entrenamiento -------------------

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0  # Inicializar la pérdida total
    num_batches = 0  # Inicializar el contador de batches

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward:
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item)

        # Acumular la pérdida y contar los batches
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    return avg_loss  # Devolver la pérdida promedio

def main(NUM_EPOCHS=NUM_EPOCHS):

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ],
    )

    model = ResUnet(in_channels=3, out_channels=1, dropout=DROPOUT_P).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5) # Reduce LR if validation loss plateaus

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    os.makedirs("output_assets_model", exist_ok=True)   # Asegurarse de que la carpeta output_assets_model en donde guardamos métricas y demás cosas, exista.

    if LOAD_MODEL:
        try:
            load_checkpoint(torch.load("C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/Models/Unet/output_assets_model/best_model_checkpoint_ResUnet.pth",  weights_only=True), model)
            print("Model loaded successfully!")
        except:
            print("Model not found. Training from scratch...")

    scaler = torch.amp.GradScaler('cuda')
    L_dice = []
    L_IoU = []
    L_loss = []
    L_accuracy = []
    L_precision = []
    L_recall = []
    L_f1_score = []
    best_dice_score = 0.0
    cnt_patience = 0

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch + 1}")
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        L_loss.append(epoch_loss)
        scheduler.step(epoch_loss) # Update scheduler based on training loss

        # Check accuracy on validation set:
        dc, IoU, acc, prec, rec, f1_s = check_metrics(val_loader, model, device=DEVICE)
        L_dice.append(dc)
        L_IoU.append(IoU) 
        L_accuracy.append(acc)
        L_precision.append(prec)
        L_recall.append(rec)
        L_f1_score.append(f1_s)

        if SAVE_MODEL:
            # Save best model, based in dice score:
            if dc > best_dice_score:
                best_dice_score = dc
                cnt_patience = 0
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                # torch.save(checkpoint, f"model_checkpoint_epoch_{epoch+1}.pth")

                # Guardar el modelo en .pth y en .zip:
                torch.save(checkpoint, "output_assets_model/best_model_checkpoint_ResUnet.pth")
                # torch.save(checkpoint, "my_checkpoint.pth.tar")
                with zipfile.ZipFile("output_assets_model/best_model_checkpoint_ResUnet.zip", 'w') as zipf:
                    zipf.write("output_assets_model/best_model_checkpoint_ResUnet.pth")
            else:
                cnt_patience += 1
        # Early stopping:
        if cnt_patience >= PATIENCE:
            print(f"===Early stopping at epoch: {epoch:04d}===")
            break
        # Save some example predictions to a folder
        if SAVE_IMS:
            save_predictions_as_imgs( val_loader, model, folder="output_assets_model/saved_images/", device=DEVICE)
        end_time = time.time()

    if SAVE_MODEL:
        print("Saving metrics...")
        # Save metrics for each epoch:
            # Crear un DataFrame con las listas y las épocas
        data = {'Epoch': range(1, len(L_dice) + 1), 'Loss': L_loss, 'Dice Score': L_dice, 'IoU': L_IoU, 'Accuracy': L_accuracy, 'Precision': L_precision, 'Recall': L_recall, 'F1 Score': L_f1_score}
        df = pd.DataFrame(data)
        df.to_csv('output_assets_model/metrics_ResUnet.csv', index=False)     # Guardar el DataFrame en un archivo CSV

        # Plot Dice and Loss:
        plot_dice_loss(L_dice, L_loss, show_plot=False)

        # Save best val metrics during training in a csv file:
        best_metrics = {'Best Dice Score': max(L_dice), 'Best IoU': max(L_IoU), 'Best Accuracy': max(L_accuracy), 'Best Precision': max(L_precision), 'Best Recall': max(L_recall), 'Best F1 Score': max(L_f1_score)}
        pd.DataFrame(best_metrics, index=[0]).to_csv('output_assets_model/best_metrics_val(during_training)_ResUnet.csv', index=False)

        # Save parameters:
        parameters = {'Num Epochs': NUM_EPOCHS, 'Learning Rate': LEARNING_RATE, 'Batch Size': BATCH_SIZE, 'Image Height': IMAGE_HEIGHT, 'Image Width': IMAGE_WIDTH, 'Device': str(DEVICE), 'Num Workers': NUM_WORKERS, 'Pin Memory': PIN_MEMORY, 'Load Model': LOAD_MODEL, 'Save Images': SAVE_IMS, 'Train Image Dir': TRAIN_IMG_DIR, 'Val Image Dir': VAL_IMG_DIR, 'Elapsed Time[m]': round((end_time - start_time)/60, 4), 'Dropout Probability': DROPOUT_P, 'Patience (early_stop)': PATIENCE, 'Optimizer': 'Adam', 'Scheduler': 'ReduceLROnPlateau', 'Loss Function': 'Dice Loss', 'Scaler': 'GradScaler', 'Execution date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Model': 'ResUnet'}
        pd.DataFrame(parameters, index=[0]).to_csv('output_assets_model/parameters_ResUnet.csv', index=False)    # Guardar los parámetros en un archivo CSV
            # Guardar los parámetros como un archivo .json:
        with open('output_assets_model/parameters_ResUnet.json', 'w') as json_file:
            json.dump(parameters, json_file, indent=4)

        print("Metrics saved successfully!")
    return model
    # return model, L_dice, L_loss, L_accuracy, L_precision, L_recall, L_f1_score


# ------------------- Entrenamiento -------------------
num_ep = input("Enter # of epochs: ")
# Modl = main(NUM_EPOCHS=int(num_ep))
try:
    Modl = main(NUM_EPOCHS=int(num_ep))
except NameError:
    print("Número inválido. Se entrenará con 10 épocas por default.")
    Modl = main()
