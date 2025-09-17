import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

checkpoint_path = './data/models/checkpoint.pth'

# ---------------------------
# 1. CIFAR-10 Daten
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(
    root='./data/cifar10', train=True, download=False, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ---------------------------
# 2. FlowNet mit Klassenkonditionierung
# ---------------------------
class ConditionalFlowNet(nn.Module):
    def __init__(self, in_channels=3, hidden=64, num_classes=10):
        super().__init__()
        # Klasseneinbettung
        self.class_emb = nn.Embedding(num_classes, hidden)
        self.time_emb = nn.Linear(1, hidden)

        # Encoder/Decoder
        self.enc1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.enc2 = nn.Conv2d(hidden, hidden*2, 3, padding=1)
        self.dec1 = nn.Conv2d(hidden*2, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, in_channels, 3, padding=1)

        # Projektionen für Konditionierung
        self.proj_enc2 = nn.Linear(hidden, hidden*2)
        self.proj_dec1 = nn.Linear(hidden, hidden)

    def forward(self, x, t, y):
        # Zeit + Klasse → Konditionsvektor
        t_emb = self.time_emb(t.view(-1, 1))   # [B, hidden]
        y_emb = self.class_emb(y)              # [B, hidden]
        cond = t_emb + y_emb                   # [B, hidden]

        # CNN mit Kondition
        h = F.relu(self.enc1(x))

        # enc2 + proj_cond
        cond2 = self.proj_enc2(cond).unsqueeze(-1).unsqueeze(-1)  # [B, 128,1,1]
        h = F.relu(self.enc2(h) + cond2)

        # dec1 + proj_cond
        cond1 = self.proj_dec1(cond).unsqueeze(-1).unsqueeze(-1)  # [B, 64,1,1]
        h = F.relu(self.dec1(h) + cond1)

        return self.out(h)

# ---------------------------
# 3. Setup
# ---------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConditionalFlowNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 4. Training Loop
# ---------------------------

# --- Check if checkpoint exists and load ---
start_epoch = 1
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

epochs = 20  # zum Testen erstmal kleiner wählen

for epoch in range(epochs):
    for x_real, y in train_loader:
        x_real, y = x_real.to(device), y.to(device)
        B, C, H, W = x_real.shape

        # Start- und Zielbilder
        x0 = torch.randn_like(x_real)
        x1 = x_real

        # Zeit in [0,1]
        t = torch.rand(B, 1, 1, 1, device=device)
        x_t = (1 - t) * x0 + t * x1

        # True Flow
        v_true = x1 - x0

        # Vorhersage
        v_pred = model(x_t, t, y)

        # Loss
        loss = criterion(v_pred, v_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # --- Checkpoint speichern alle 5 Epochen ---
    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved -> {checkpoint_path}")

# ---------------------------
# 5. Generierung mit Kategorie
# ---------------------------
def generate_images(model, label, steps=20, n=8):
    model.eval()
    with torch.no_grad():
        x = torch.randn(n, 3, 32, 32, device=device)
        dt = 1.0 / steps
        y = torch.full((n,), label, device=device, dtype=torch.long)

        for i in range(steps):
            t = torch.full((n,1,1,1), i*dt, device=device)
            v = model(x, t, y)
            x = x + dt * v

        x = (x.clamp(-1, 1) + 1) / 2
    return x.cpu()

# Beispiel: 3 = Katze, 5 = Hund (CIFAR-10 Labels)
x_cat = generate_images(model, label=3, steps=20, n=8)
x_dog = generate_images(model, label=5, steps=20, n=8)

# ---------------------------
# 6. Ergebnisse anzeigen
# ---------------------------
def show_grid(imgs, title=""):
    grid = torch.cat([imgs[i] for i in range(len(imgs))], dim=2).permute(1,2,0)
    plt.figure(figsize=(12,2))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(title)
    plt.show()

show_grid(x_cat, "Kategorie: Katze")
show_grid(x_dog, "Kategorie: Hund")
