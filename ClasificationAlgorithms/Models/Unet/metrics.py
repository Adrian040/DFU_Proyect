import torch
import torch.nn.functional as F
import numpy as np
from utils import get_test_loader

def check_metrics(loader, model, device="cuda"):
    num_classes = 4
    metrics = {
        "dice_coefficient": torch.zeros(num_classes, device=device),
        "IoU": torch.zeros(num_classes, device=device),
        "accuracy": torch.zeros(num_classes, device=device),
        "precision": torch.zeros(num_classes, device=device),
        "recall": torch.zeros(num_classes, device=device),
        "f1_score": torch.zeros(num_classes, device=device),
    }
    class_counts = torch.zeros(num_classes, device=device)
    
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.softmax(model(x), dim=1)
            preds = torch.argmax(preds, dim=1)

            for cls in range(num_classes):
                true_positive = ((preds == cls) & (y == cls)).sum().float()
                false_positive = ((preds == cls) & (y != cls)).sum().float()
                false_negative = ((preds != cls) & (y == cls)).sum().float()
                true_negative = ((preds != cls) & (y != cls)).sum().float()

                metrics["dice_coefficient"][cls] += (2 * true_positive) / (2 * true_positive + false_positive + false_negative + 1e-8)
                metrics["IoU"][cls] += true_positive / (true_positive + false_positive + false_negative + 1e-8)
                metrics["accuracy"][cls] += (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative + 1e-8)
                metrics["precision"][cls] += true_positive / (true_positive + false_positive + 1e-8)
                metrics["recall"][cls] += true_positive / (true_positive + false_negative + 1e-8)
                metrics["f1_score"][cls] += 2 * (metrics["precision"][cls] * metrics["recall"][cls]) / (metrics["precision"][cls] + metrics["recall"][cls] + 1e-8)
                class_counts[cls] += 1

    for key in metrics:
        metrics[key] /= class_counts

    # Print metrics:
    for cls in range(num_classes):
        print(f"Class {cls}:")
        print(f"  Dice Coefficient: {metrics['dice_coefficient'][cls].item():.4f}")
        print(f"  IoU: {metrics['IoU'][cls].item():.4f}")
        print(f"  Accuracy: {metrics['accuracy'][cls].item():.4f}")
        print(f"  Precision: {metrics['precision'][cls].item():.4f}")
        print(f"  Recall: {metrics['recall'][cls].item():.4f}")
        print(f"  F1 Score: {metrics['f1_score'][cls].item():.4f}")

    dict_metrics= {key: metrics[key].tolist() for key in metrics}
    model.train()
    return dict_metrics

def calculate_metrics(test_image_dir, test_mask_dir, model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), image_height=240, image_width=240, num_workers=0, batch_size=4, pin_memory=True):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    model.eval()

    loader = get_test_loader(test_image_dir, test_mask_dir, batch_size= batch_size,  image_height=image_height, image_width=image_width, num_workers=num_workers, pin_memory=pin_memory)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            true_positive += ((preds == 1) & (y == 1)).sum()
            false_positive += ((preds == 1) & (y == 0)).sum()
            false_negative += ((preds == 0) & (y == 1)).sum()

    dice_coefficient = dice_score / len(loader) if len(loader) > 0 else 0
    IoU = true_positive / (true_positive + false_positive + false_negative) if (true_positive + false_positive + false_negative) > 0 else 0
    accuracy = num_correct / num_pixels if num_pixels > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    model.train() # regresarlo a su estado original si se quiere seguir entrenando el modelo.

    return dice_coefficient.item(), IoU.item(), accuracy.item(), precision.item(), recall.item(), f1_score.item()

def dice_loss(input, target):
    smooth = 1.0
    input = torch.sigmoid(input)  # Aplicar sigmoide para obtener probabilidades
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


# def dice_loss_multiclass(pred, target, epsilon=1e-6):
#     """
#     Calcula la Dice Loss para segmentación multiclase.

#     Args:
#         pred (torch.Tensor): Salidas del modelo (logits) de tamaño (batch_size, num_classes, H, W).
#         target (torch.Tensor): Etiquetas verdaderas de tamaño (batch_size, H, W).
#         epsilon (float): Pequeño valor para evitar división por cero.

#     Returns:
#         torch.Tensor: Valor escalar de la pérdida Dice.
#     """
#     # Aplicar Softmax a las predicciones para convertir logits a probabilidades
#     pred = F.softmax(pred, dim=1)

#     # Convertir las etiquetas a one-hot encoding
#     target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

#     # Calcular el Dice para cada clase
#     intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
#     union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))

#     dice = (2.0 * intersection + epsilon) / (union + epsilon)
#     dice_loss = 1 - dice.mean()

#     return dice_loss

def dice_loss_multiclass(pred, target, epsilon=1e-6):
    """
    Calcula la Dice Loss para segmentación multiclase.

    Args:
        pred (torch.Tensor): Salidas del modelo (logits) de tamaño (batch_size, num_classes, H, W).
        target (torch.Tensor): Etiquetas verdaderas de tamaño (batch_size, H, W).
        epsilon (float): Pequeño valor para evitar división por cero.

    Returns:
        torch.Tensor: Valor escalar de la pérdida Dice.
    """
    # Asegurarse de que las etiquetas estén en el tipo correcto
    if target.dtype != torch.long:
        target = target.long()

    # Verifica dimensiones del target antes de one-hot
    print(f"Dimensiones del target antes de one-hot: {target.shape}")

    # Aplicar Softmax a las predicciones para convertir logits a probabilidades
    pred = F.softmax(pred, dim=1)

    # Convertir las etiquetas a one-hot encoding y ajustar dimensiones
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1])  # (batch_size, H, W, num_classes)
    target_one_hot = target_one_hot.squeeze(1).permute(0, 3, 1, 2).float()  # (batch_size, num_classes, H, W)

    # Verifica dimensiones después del procesamiento
    print(f"Dimensiones del target después de one-hot: {target_one_hot.shape}")

    # Calcular el Dice para cada clase
    intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
    union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))

    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    dice_loss = 1 - dice.mean()

    return dice_loss


