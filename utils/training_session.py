import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from conditional_dynamic_unet import *
from adaptive_scheduler import *
from trainer import *
import utils

import ssl
import certifi

ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------
# Main
# -----------------------------
filepath = './model/cfm_dynamic_unet_checkpoint.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

model = DynamicUNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
params_dict = {
    'lr_normal': 1e-4,
    'lr_plateau': 5e-5,
    'lr_precision': 1e-5,
    'lr_warmup_start': 1e-3,
    'lr_human': 2e-4,
    'warmup_steps': 30,
    'step_size': 2,
    'step_gamma': 0.5,
    'cosine_Tmax': 5,
    'jump_prob': 0.2,
    'jump_factor': 0.1
}
scheduler = AdaptiveScheduler(optimizer, params_dict)
trainer = Trainer(model, optimizer, scheduler, criterion, train_loader, device)

num_epochs = 100

# -----------------------------
# Trainings Loop
# -----------------------------
"""
vgl.: "FLow Matching for Generative Modeling"; https://www.youtube.com/watch?v=7NNxK3CqaDk
"""
load_model = True
if os.path.exists(filepath) and load_model:
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_epoch = checkpoint['epoch']
else:
    global_epoch = 0

total_epochs = global_epoch + num_epochs

for epoch in range(global_epoch, total_epochs):
    avg_loss = trainer.train_epoch()
    print(f"Epoch {epoch+1}/{total_epochs}, Loss={avg_loss:.6f}")
    trainer.generate_image([0,1,2], epoch)

    scheduler.step(avg_loss)

    checkpoint_threshold = epoch + 1
    if checkpoint_threshold % 5 == 0:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at epoch {epoch + 1} -> {filepath}")

