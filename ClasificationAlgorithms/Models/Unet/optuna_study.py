import optuna
import torch
import torch.optim as optim
from main import UNET
from train import get_loaders, train_fn
from metrics import dice_loss_multiclass

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    batch_size = trial.suggest_int('batch_size', 2, 8)
    dropout_probability = trial.suggest_uniform('dropout_probability', 0.0, 0.5)

    # Load data
    train_loader, val_loader = get_loaders(
        "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/train_images",
        "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/train_masks",
        "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_images",
        "C:/Users/am969/Documents/DFU_Proyect/ClasificationAlgorithms/data_TissueSegNet/data_padded/val_masks",
        batch_size,
        None,  # Placeholder for train_transform
        None,  # Placeholder for val_transforms
        0,     # NUM_WORKERS
        True   # PIN_MEMORY
    )

    # Initialize model
    model = UNET(in_channels=3, out_channels=4, impl_dropout=True, prob_dropout=dropout_probability).to("cuda" if torch.cuda.is_available() else "cpu")

    # Select optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Loss function
    loss_fn = dice_loss_multiclass

    # Training loop (simplified for optimization)
    total_loss = 0
    for epoch in range(1):  # Run for a single epoch for optimization
        epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, None)  # scaler is None for simplicity
        total_loss += epoch_loss

    return total_loss

# Run the Optuna study
if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    # Save the best trial
    trial = study.best_trial
    print("Best trial:")
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))