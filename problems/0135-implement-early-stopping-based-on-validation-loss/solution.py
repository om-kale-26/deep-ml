from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    # Your code 

    best_loss = val_losses[0]
    best_epcoh = 0
    epochs_without_improvement = 0

    for epoch in  range(1, len(val_losses)):
        current_loss = val_losses[epoch]

        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_epcoh = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1 

        if epochs_without_improvement >= patience:
            return epoch, best_epcoh
        
    return len(val_losses) - 1, best_epcoh
