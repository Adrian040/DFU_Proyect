import json
import pandas as pd
import zipfile
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time
import numpy as np
import torch.optim as optim
from main import UNET
from metrics import check_metrics, dice_loss_multiclass, calculate_metrics
from utils import save_predictions_as_imgs, load_checkpoint, get_loaders, plot_dice_loss, concat_dicts_to_dataframe


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
# For early stopping
EARLY_STOP = True # True to activate early stopping
PATIENCE = 50 # for early stopping
# Dropout
dropout = True
p_dropout = 0.30

TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/train_images"
TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/train_masks"
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_masks"


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

    model = UNET(in_channels=3, out_channels=4, impl_dropout = dropout, prob_dropout=p_dropout).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = dice_loss
    loss_fn = dice_loss_multiclass
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

    if LOAD_MODEL:
        load_checkpoint(torch.load("C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/Models/Unet/output_assets_model/best_model_checkpoint.pth",  weights_only=True), model)
        print("Model loaded successfully!")

    scaler = torch.amp.GradScaler('cuda')
    L_dicts_metrics = []  # Lista de diccionarios de métricas de cada época
    L_loss = []  # Lista de pérdidas
    best_dice = 0.0
    best_loss = 0.0
    cnt_patience = 0

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch + 1}")

        # Train the model:
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        L_loss.append(epoch_loss)
        scheduler.step(epoch_loss) # Update scheduler based on training loss

        # Check accuracy on validation set:
        dict_metrics_per_class = check_metrics(val_loader, model, device=DEVICE)
        epoch_mean_dice = np.mean(dict_metrics_per_class["dice_coefficient"])  # Coeficiente dice promedio de todas las clases en la época correspondiente
        L_dicts_metrics.append(dict_metrics_per_class)
        print(f"mean dice: {epoch_mean_dice}")

        if SAVE_MODEL:
            if epoch == 1:
                best_loss = epoch_loss
                best_dice = epoch_mean_dice
            # Save best model, based in dice:
            # if epoch_loss <= best_loss:
            if epoch_mean_dice >= best_dice:
                print(f"Model saved with loss: {epoch_loss} and mean dice: {epoch_mean_dice}")
                best_loss = epoch_loss
                best_dice = epoch_mean_dice
                best_model_epoch = epoch
                cnt_patience = 0

                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                # torch.save(checkpoint, f"model_checkpoint_epoch_{epoch+1}.pth")

                # Guardar el modelo en .pth y en .zip:
                torch.save(checkpoint, "output_assets_model/best_model_checkpoint2.pth")
                # torch.save(checkpoint, "my_checkpoint.pth.tar")
                with zipfile.ZipFile("output_assets_model/best_model_checkpoint2.zip", 'w') as zipf:
                    zipf.write("output_assets_model/best_model_checkpoint2.pth")
            else:
                cnt_patience += 1 # Resetear el contador de paciencia si el modelo no mejora.
        # Early stopping
        if EARLY_STOP and cnt_patience >= PATIENCE:
            print(f"===Early stopping at epoch: {epoch:04d}===")
            break
            
        # Save some example predictions to a folder
        if SAVE_IMS:
            print("saving image in training...")
            save_predictions_as_imgs(val_loader, model, folder="output_assets_model/saved_images/", device=DEVICE)
    end_time = time.time()


    if SAVE_MODEL:
        print("Saving metrics...")
        # Save metrics for each epoch:
        df_metrics = concat_dicts_to_dataframe(L_dicts_metrics)
        df_metrics.to_csv('output_assets_model/metrics_per_epoch.csv', index=False)
        
        # Plot Dice (for class 0 only, it can be any class but just one at time) and Loss:
        L_dice_0 = df_metrics[df_metrics.Class == 0]['dice_coefficient'].tolist()
        plot_dice_loss(L_dice_0, L_loss, show_plot=False)

        # Save best val metrics during training in a csv file:
        cols = ['Best Dice Score', 'Best IoU', 'Best Accuracy', 'Best Precision', 'Best Recall', 'Best F1 Score']
        best_metrics_c0 = [max(df_metrics[df_metrics.Class == 0]['dice_coefficient']), max(df_metrics[df_metrics.Class == 0]['IoU']), max(df_metrics[df_metrics.Class == 0]['accuracy']), max(df_metrics[df_metrics.Class == 0]['precision']), max(df_metrics[df_metrics.Class == 0]['recall']), max(df_metrics[df_metrics.Class == 0]['f1_score'])]
        best_metrics_c1 = [max(df_metrics[df_metrics.Class == 1]['dice_coefficient']), max(df_metrics[df_metrics.Class == 1]['IoU']), max(df_metrics[df_metrics.Class == 1]['accuracy']), max(df_metrics[df_metrics.Class == 1]['precision']), max(df_metrics[df_metrics.Class == 1]['recall']), max(df_metrics[df_metrics.Class == 1]['f1_score'])]
        best_metrics_c2 = [max(df_metrics[df_metrics.Class == 2]['dice_coefficient']), max(df_metrics[df_metrics.Class == 2]['IoU']), max(df_metrics[df_metrics.Class == 2]['accuracy']), max(df_metrics[df_metrics.Class == 2]['precision']), max(df_metrics[df_metrics.Class == 2]['recall']), max(df_metrics[df_metrics.Class == 2]['f1_score'])]
        best_metrics_c3 = [max(df_metrics[df_metrics.Class == 3]['dice_coefficient']), max(df_metrics[df_metrics.Class == 3]['IoU']), max(df_metrics[df_metrics.Class == 3]['accuracy']), max(df_metrics[df_metrics.Class == 3]['precision']), max(df_metrics[df_metrics.Class == 3]['recall']), max(df_metrics[df_metrics.Class == 3]['f1_score'])]
        best_metrics_df = pd.DataFrame([best_metrics_c0, best_metrics_c1, best_metrics_c2, best_metrics_c3], columns=cols, index=[0, 1, 2, 3])
        best_metrics_df.index.name = 'Class'
        best_metrics_df.to_csv('output_assets_model/best_metrics_val(during_training).csv', index=True)

        # Save parameters:
        parameters = {'Num Epochs': NUM_EPOCHS, 'Learning Rate': LEARNING_RATE, 'Batch Size': BATCH_SIZE, 'Image Height': IMAGE_HEIGHT, 'Image Width': IMAGE_WIDTH, 'Device': str(DEVICE), 'Num Workers': NUM_WORKERS, 'Pin Memory': PIN_MEMORY, 'Load Model': LOAD_MODEL, 'Save Images': SAVE_IMS, 'Train Image Dir': TRAIN_IMG_DIR, 'Val Image Dir': VAL_IMG_DIR, 'Elapsed Time[m]': round((end_time - start_time)/60, 4), 'Best_model_epoch': best_model_epoch, 'Early Stopping': EARLY_STOP, 'Patience': PATIENCE, 'Dropout' : dropout, 'Prob. Dropout' : 0.0 if not dropout else p_dropout}  
        pd.DataFrame(parameters, index=[0]).to_csv('output_assets_model/parameters.csv', index=False)    # Guardar los parámetros en un archivo CSV
            # Guardar los parámetros como un archivo .json:
        with open('output_assets_model/parameters.json', 'w') as json_file:
            json.dump(parameters, json_file, indent=4)

        print('Best model epoch:', best_model_epoch)
        print("Metrics saved successfully!")
    return model


# ------------------- Entrenamiento -------------------
num_ep = input("Enter # of epochs: ")
# Modl = main(NUM_EPOCHS=int(num_ep))
try:
    print("Training for ", num_ep, " epochs.")
    Modl = main(NUM_EPOCHS=int(num_ep))
except NameError:
    print("Número inválido. Se entrenará con 10 épocas por default.")
    Modl = main()