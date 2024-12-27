from re import L
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from main import UNET, get_loaders
from metrics import check_metrics
from utils import save_predictions_as_imgs
from metrics import dice_loss


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",DEVICE)
BATCH_SIZE = 4
# NUM_EPOCHS = 83
NUM_WORKERS = 0
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_IMS = True

TRAIN_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/train_images"
TRAIN_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/train_masks"
VAL_IMG_DIR = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/val_images"
VAL_MASK_DIR = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/data_DFU_images/data_MICCAI/val_masks"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0  # Inicializar la pérdida total
    num_batches = 0  # Inicializar el contador de batches

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward:
        with torch.cuda.amp.autocast():
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


def main(NUM_EPOCHS=10):
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

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
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

    # if LOAD_MODEL:
    #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    L_dice = []
    L_loss = []
    L_accuracy = []
    L_precision = []
    L_recall = []
    L_f1_score = []
    best_dice_score = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch + 1}")
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        L_loss.append(epoch_loss)
        scheduler.step(epoch_loss) # Update scheduler based on training loss

        # Check accuracy on validation set
        dc, acc, prec, rec, f1_s = check_metrics(val_loader, model, device=DEVICE)
        L_dice.append(dc)
        L_accuracy.append(acc)
        L_precision.append(prec)
        L_recall.append(rec)
        L_f1_score.append(f1_s)

        # Save best model, based in dice score.
        if dc > best_dice_score:
          best_dice_score = dc
          checkpoint = {
              "state_dict": model.state_dict(),
              "optimizer": optimizer.state_dict(),
          }
          # torch.save(checkpoint, f"model_checkpoint_epoch_{epoch+1}.pth")
          torch.save(checkpoint, "output_assets_model/best_model_checkpoint.pth")

        # Save some example predictions to a folder
        if SAVE_IMS:
            save_predictions_as_imgs(
                val_loader, model, folder="output_assets_model/saved_images/", device=DEVICE
            )
    return model, L_dice, L_loss, L_accuracy, L_precision, L_recall, L_f1_score


# Start training:
trained_model, L_dice_result, L_loss_result, L_acc_result, L_prec_result, L_rec_result, L_f1s_result = main(NUM_EPOCHS = 2)

