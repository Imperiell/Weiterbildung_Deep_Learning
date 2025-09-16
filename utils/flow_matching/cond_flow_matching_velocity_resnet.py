import os

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torch.nn.functional as F

# KI für Konzeption und Debugging des Codes verwendet.
# Adaptionen aus Papers oder anderen Quellen sind an entsprechender Stelle vermerkt.

#TODO: Nummerierung

# -----------------------------
# 1. Sinusoidal Time Embedding
# -----------------------------

# warum kein RNN -> unstabiler

"""
Die Zeit ist eine kontinuierliche Variable t E [0,1], die durch ein lineares Einbetten oft nicht
ausreicht, um komplexe Abhängigkeiten zu lernen.
Die Sinusfunktion erlaubt nichtlineare Muster über die Zeit zu lernen.
Das Ziel ist ein Werteverlauf [-1, 1] für verschiedene Frequenzen.
Mit jeder Frequenz wird eine andere Zeitskala codiert.

-> Zusammenfassung:
Auf diese Weise lässt sich das Modell auf unterschiedliche "Abtastraten" konditionieren.

vgl.: "Attention is All You Need"; https://arxiv.org/abs/1706.03762 ; Kapitel 3.5
"""
def sinusoidal_embedding(t, dim=128):
    # Dimension des Zeit-Embeddings, also wie viele Zahlen pro Zeitpunkt erzeugt werden sollen.
    # dim = 32 -> jeder Zeitpunkt t wird durch einen Vektor mit 32 Elementen dargestellt.

    # t repräsentiert den Zeitpunkt und ist ein Tensor der Form [B] für die Batchgröße B.
    # Ziel: Das Erzeugen eines kontinuierlichen sinusoidalen Embedding der Form [B, dim].

    # Embedding-Dimension in zwei Hälften teilen
    # Erste Hälfte Sinus, Zweite Kosinus
    # ACHTUNG: Standardtrick bei Flow-Embedding -> Zeit wird in unterschiedlichen Frequenzen codiert.
    half_dim = dim // 2
    # wi = exp(-i * (ln10000 / half_dim - 1))
    freqs = torch.exp(-torch.arange(half_dim, device=t.device).float() * (torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
    # Zeiten mit Frequenzen multiplizieren
    # Idee: kleine t -> lansame Oszillation, große t -> schnellere Oszillation
    angles = t * freqs
    # Erste Hälfte: sin(t * wi)
    # Zweite Hälfte: cos(t * wi)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # [B, dim]

# -----------------------------
# 2. Residual Block
# -----------------------------
"""
Warum Residual Blöcke?
Beim Flow Matching trainiert ein Modell, das Vektorfelder (Geschwindigkeiten) im Bildraum approximiert.
Solche Modelle müssen **fein abgestufte Unterschiede** zwischen dem Zielwert x_t und dem nächsten Wert
x_{t + dt} lernen, ohne vanishing oder exploding Gradients zu erzeugen.
Bei CNNs ohne Residual-Verbindungen werden Signale stark abgeschwächt oder verstärkt, was zu instabiler
Approximation der Geschwindigkeit führt.
Durch die Identität wird das ResNet deutlich stabiler.
"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding = 1),
            nn.Conv2d(out_channel, out_channel, 3, padding = 1)
        )
        self.skip = nn.Conv2d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()

    def forward(self, x):
        return F.leaky_relu(self.model(x) + self.skip(x), negative_slope=0.01)

# -----------------------------
# 3. Conditional Velocity Net
# -----------------------------
class ConditionalVelocityNet(nn.Module):
    def __init__(self, num_classes = 10, time_dim = 128, latent_dim = 128): # latent_dim
        super().__init__()
        # Ein direkter linearer Zeitwert t kann keine komplexen, nichtlinearen Abhängigkeiten zwischen Zeit und
        # Feature-Flow lernen.
        # Mit dem Sinusoidalen Embedding und MLP kann gelernt werden, welche Frequenzen für welche Features relevant
        # sind. Somit kann das Modell flexibel auf die unterschiedlichen Zeitpunkte reagieren.
        # Das Flow-Modell kann so feine Abstufungen in der dynamischen Entwicklung von x0 -> x1 lernen.
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128,128)
        )

        self.label_emb = nn.Embedding(num_classes, 128)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding = 1),
            ResidualBlock(32, 64),
            ResidualBlock(64, latent_dim)
        )

        self.decoder = nn.Sequential(
            ResidualBlock(latent_dim, 64),
            ResidualBlock(64, 32),
            nn.Conv2d(32,1, 3, padding = 1)
        )

    def forward(self, x_t, t, label):
        # Zeit Embedding, für Latent space
        t_emb = sinusoidal_embedding(t)
        t_emb = self.time_projection(t_emb).unsqueeze(-1).unsqueeze(-1)

        # Label Embedding, für Latent space
        y_emb = self.label_emb(label).unsqueeze(-1).unsqueeze(-1)

        # Feature Flow
        h = self.encoder(x_t)
        h = h + t_emb + y_emb # Latent space
        h = self.decoder(h)

        return h

# -----------------------------
# ODE - Ordinary Differntial Equations
# -----------------------------
class ODEFunc(nn.Module):
    def __init__(self, model, label):
        super().__init__()
        self.model = model
        self.label = label

    def forward(self, t, x):
        tb = t.expand(x.size(0), 1) # Skalar Zeit `t` wird in die Form [batch_size, 1] gebracht
        return self.model(x, tb, self.label)

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
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # nächste epoch starten
    print(f"Checkpoint loaded from {filepath}, resuming at epoch {start_epoch}")
    return start_epoch

# -----------------------------
# 4. Daten
# -----------------------------
# TODO: Normalisieren im NN
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
    # ACHTUNG: Pixel beim Decodieren auf [0,1] zurücktranformieren
    # x_gen = (x_gen + 1) / 2  # [-1,1] -> [0,1]
])
train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# -----------------------------
# 5. Hyperparameter
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConditionalVelocityNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr = 1e-4)

num_epochs = 5

filepath = './model/cfmvr_checkpoint.pth'

# -----------------------------
# 6. Trainings Loop
# -----------------------------
"""
vgl.: "FLow Matching for Generative Modeling"; https://www.youtube.com/watch?v=7NNxK3CqaDk
"""
if os.path.exists(filepath):
    global_epoch = load_checkpoint(model, optimizer, filepath, device=device)
else:
    print("No checkpoint found, starting from scratch.")
    global_epoch = 0
    save_checkpoint(model, optimizer, global_epoch, filepath)

total_epochs = global_epoch + num_epochs

for epoch in range(global_epoch, global_epoch + num_epochs):
    model.train()
    total_loss = 0.0

    for x1, y in train_loader:
        x1, y = x1.to(device), y.to(device)

        # Zufällige Startverteilung
        x0 = torch.randn_like(x1)

        # Zufälliger Zeitpunkt t für jeden Batch-Eintrag, zwischen 0 und 1.
        # Wird für die lineare Interpolation zwischen x0 und x1 verwendet.
        t = torch.rand(x1.size(0), 1, device = device)

        # Lineare Interpolation zwischen x0 und x1
        x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1

        # Zielgeschwindigkeitsvektor
        v_target = x1 - x0

        # Vorhersage des Geschwindigkeitsvektors
        v_pred = model(x_t, t, y)

        # Loss: MESLoss
        loss = criterion(v_pred, v_target)

        # Gradient zurücksetzen
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Nächster Schritt
        optimizer.step()

        total_loss += loss.item() * x1.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{total_epochs}, Loss: {avg_loss:.6f}")

    if (epoch + 1) % 5 == 0:
        save_checkpoint(model, optimizer, epoch + 1, filepath)

# -----------------------------
# 7. Sampling via ODE (conditional)
# -----------------------------
@torch.no_grad()
def sample_flow_ode_cond(model, y_label, steps = 200, method = 'dopri5'):
    model.eval()

    # Zufällige Startverteilung
    x0 = torch.randn(1, 1, 28, 28, device=device)

    t_span = torch.linspace(0.0, 1.0, steps, device = device)

    y = torch.tensor([y_label], device = device)

    ode_func = ODEFunc(model, y)

    x_sequence = odeint(ode_func, x0, t_span, method = method)

    return x_sequence[-1].cpu()

# -----------------------------
# Bild Generierung
# -----------------------------
sample = sample_flow_ode_cond(model, y_label = 3)
plt.imshow(sample.squeeze().numpy(), cmap="gray")
plt.title("Conditional Flow Matching Sample: 3")
plt.axis("off")
plt.show()