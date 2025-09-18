import os
import torch

# -----------------------------
# Speichern und Laden des Modells
# -----------------------------
def save_checkpoint(model, optimizer, epoch, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch} -> {filepath}")

def load_checkpoint(model, optimizer, filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    start_epoch = checkpoint['epoch']

    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Checkpoint loaded (partial) from {filepath}, resuming at epoch {start_epoch}")
    except RuntimeError as e:
        print(f"Warning: Could not fully load checkpoint: {e}")
        print("Model weights will be partially loaded or reinitialized.")

    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded.")
    except Exception as e:
        print(f"Warning: Could not load optimizer state: {e}")

    return start_epoch
