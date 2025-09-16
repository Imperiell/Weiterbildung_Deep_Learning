import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1. Datensatz und Transformation
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    # ACHTUNG: Pixel beim Decodieren auf [0,1] zurücktranformieren
    # x_gen = (x_gen + 1) / 2  # [-1,1] -> [0,1]
])

train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# -----------------------------
# 2. Zeit-Embedding (Sinusoidal)
# -----------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Dimension des Zeit-Embeddings, also wie viele Zahlen pro Zeitpunkt erzeugt werden sollen.
        # dim = 32 -> jeder Zeitpunkt t wird durch einen Vektor mit 32 Elementen dargestellt.
        self.dim = dim

    def forward(self, t):
        # t repräsentiert den Zeitpunkt und ist ein Tensor der Form [B] für die Batchgröße B.
        # Ziel: Das Erzeugen eines kontinuierlichen sinusoidalen Embedding der Form [B, dim].

        # Embedding-Dimension in zwei Hälften teilen
        # Erste Hälfte Sinus, Zweite Kosinus
        # ACHTUNG: Standardtrick bei Flow-Embedding -> Zeit wird in unterschiedlichen Frequenzen codiert.
        half_dim = self.dim // 2
        # wi = exp(-i * (ln10000 / half_dim - 1))
        emb = torch.exp(-torch.arange(half_dim, device=t.device) * torch.log(torch.tensor(10000.0)) / (half_dim-1))
        # Zeiten mit Frequenzen multiplizieren
        # Idee: kleine t -> lansame Oszillation, große t -> schnellere Oszillation
        emb = t[:, None] * emb[None, :]
        # Erste Hälfte: sin(t * wi)
        # Zweite Hälfte: cos(t * wi)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # [B, dim]

# -----------------------------
# 3. Encoder/Decoder (Latent Space)
# -----------------------------
latent_dim = 64  # Größe des Latent Space

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: Bild -> Latent
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder: Latent -> Bild
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def encode(self, x):
        z = self.encoder(x.view(x.size(0), -1))
        return z  # Latent Space [B, latent_dim]

    def decode(self, z):
        x_rec = self.decoder(z)
        return x_rec.view(-1,1,28,28)

# -----------------------------
# 4. Flow Network im Latent Space
# -----------------------------
class FlowNetwork(nn.Module):
    def __init__(self, time_dim=32):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, z, t):
        # z: Latent Representation
        te = self.time_emb(t)
        x_input = torch.cat([z, te], dim=-1)
        v = self.net(x_input)  # Flow in latent space
        return v

# -----------------------------
# 5. Initialisierung
# -----------------------------
ae = Autoencoder().to(device)
flow_model = FlowNetwork().to(device)
optimizer = torch.optim.Adam(list(ae.parameters()) + list(flow_model.parameters()), lr=1e-3)

# -----------------------------
# 6. Training
# -----------------------------
dt = 0.01
for epoch in range(20):
    for x, _ in train_loader:
        x = x.to(device)
        # Encode: Bild -> Latent Space
        z = ae.encode(x)
        # Zufällige Zeit t
        t = torch.rand(x.size(0), device=device)
        # Flow in Latent Space modelliert
        v = flow_model(z, t)
        # Ziel: hier Demo mit Rauschen
        z_target = torch.randn_like(z)
        loss = F.mse_loss(z + v*dt, z_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# -----------------------------
# 7. Bildgenerierung
# -----------------------------
z_gen = torch.randn(1, latent_dim, device=device)  # Start im Latent Space
for i in range(100):
    t = torch.full((1,), i/100, device=device)
    v = flow_model(z_gen, t)
    z_gen = z_gen + v*dt  # Euler Integration im Latent Space

x_gen = ae.decode(z_gen)  # Decoder: Latent -> Bild

# Anzeige
import matplotlib.pyplot as plt
plt.imshow(x_gen[0,0].detach().cpu(), cmap='gray')
plt.show()
