import os

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torch.nn.functional as F

# KI für Konzeption (Komponenten und Schritte planen) und Debugging des Codes verwendet.
# Adaptionen aus Papers oder anderen Quellen sind an entsprechender Stelle vermerkt.

#############################
#############################
# ÄNDERUNGEN für Version 2
#############################
#############################
# Der gesamte Code ist jetzt an die Verwendung von Named Tensors angepasst.
#############################
#############################
#TODO: Ideen für Verbesserungen:
# lr Tuning
# Scheduler feinjustieren
# Loss anpassen -> kombinierte Loss?
# Batchsize anpassen
# Regularisierung anpassen -> Weight Decay, Dropout etc
# Latent Space -> Gewichte/ Priorisierung?
# UNet?
# Tiefer?

# -----------------------------
# NamedTensor Wrapper
# -----------------------------

# Named Dimensions
DIM_BATCH = "batch"
DIM_CHANNELS = "channel"
DIM_HEIGHT = "height"
DIM_WIDTH = "width"
DIM_TIME = "time"

class NamedTensor:
    def __init__(self, tensor: torch.Tensor, names: tuple[str, ...]):
        if tensor.ndim != len(names):
            raise ValueError(f"Expected {len(names)} dimensions, got {tensor.ndim}")
        self.tensor = tensor
        self.names = names

    def __repr__(self):
        return f"NamedTensor(names={self.names}, shape={tuple(self.tensor.shape)})"

    def __getattr__(self, attr):
        """
        Unbekannte Attribute werden an den inneren Tensor weitergeleitet.
        Ist das Ergebnis ein Tensor, wrappe es neu.
        """
        t_attr = getattr(self.tensor, attr)

        if callable(t_attr):
            def wrapper(*args, **kwargs):
                result = t_attr(*args, **kwargs)

                # Fallback: Namen beibehalten
                if isinstance(result, torch.Tensor):
                    return NamedTensor(result, self.names[:result.ndim])
                return result

            return wrapper
        else:
            return t_attr

    # Zugriff auf rohe Tensoren
    def raw(self):
        return self.tensor

    # Hilfsfunktionen für Namen
    def dim_index(self, name: str) -> int:
        return self.names.index(name)

    def rename(self, **rename_map):
        new_names = tuple(rename_map.get(n, n) for n in self.names)
        return NamedTensor(self.tensor, new_names)

    def permute(self, *order):
        # permute nach Namen oder Indizes
        if all(isinstance(x, str) for x in order):
            indices = [self.dim_index(n) for n in order]
            new_names = order
        else:
            indices = order
            new_names = tuple(self.names[i] for i in indices)

        result = self.tensor.permute(*indices)
        return NamedTensor(result, new_names)

    def mean(self, dim=None, *args, **kwargs):
        # Mean mit Namen
        if isinstance(dim, str):
            dim = self.dim_index(dim)
        result = self.tensor.mean(dim=dim, *args, **kwargs)
        if dim is None:
            return NamedTensor(result, ())
        else:
            new_names = self.names[:dim] + self.names[dim+1:]
            return NamedTensor(result, new_names)

    # Operatoren
    def __add__(self, other):
        if isinstance(other, NamedTensor):
            # naive Variante: gleiche Reihenfolge annehmen
            return NamedTensor(self.tensor + other.tensor, self.names)
        return NamedTensor(self.tensor + other, self.names)

    def __sub__(self, other):
        if isinstance(other, NamedTensor):
            return NamedTensor(self.tensor - other.tensor, self.names)
        return NamedTensor(self.tensor - other, self.names)

    def __mul__(self, other):
        if isinstance(other, NamedTensor):
            return NamedTensor(self.tensor * other.tensor, self.names)
        return NamedTensor(self.tensor * other, self.names)

    def __truediv__(self, other):
        if isinstance(other, NamedTensor):
            return NamedTensor(self.tensor / other.tensor, self.names)
        return NamedTensor(self.tensor / other, self.names)

    def __getitem__(self, item):
        result = self.tensor[item]
        new_names = self.names[:result.ndim]
        return NamedTensor(result, new_names)

"""
# --- Beispiel ---
x = NamedTensor(torch.randn(10, 20, 32), ("batch", "time", "features"))
print(x)

# Mittelwert über Dimension
m = x.mean("time")
print(m)

# Permute nach Namen
p = x.permute("features", "batch", "time")
print(p)

# Addition mit normalem Tensor
y = torch.randn(10, 20, 32)
z = x + y
print(z)
"""

# -----------------------------
# Sinusoidal Time Embedding
# -----------------------------

# warum kein RNN -> instabiler

"""
Die Zeit ist eine kontinuierliche Variable t E [0,1], die durch ein lineares Einbetten oft nicht
ausreicht, um komplexe Abhängigkeiten zu lernen.
Die Sinusfunktion erlaubt nichtlineare Muster über die Zeit zu lernen.
Das Ziel ist ein Werteverlauf [-1, 1] für verschiedene Frequenzen.
Mit jeder Frequenz wird eine andere Zeitskala codiert.

-> Zusammenfassung:
Auf diese Weise lässt sich das Modell auf unterschiedliche "Abtastraten" konditionieren.

vgl.: "Attention is All You Need"; https://arxiv.org/abs/1706.03762 ; Kapitel 3.5
vgl.: "Inside Sinusoidal Position Embeddings: A Sense of Order"; https://learnopencv.com/sinusoidal-position-embeddings/?utm_source=chatgpt.com
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
# Residual Block
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
            nn.BatchNorm2d(out_channel), # In forward() finden Additionen statt, daher Werte in ResidualBlock stabilisieren.
            nn.Conv2d(out_channel, out_channel, 3, padding = 1),
            nn.BatchNorm2d(out_channel)
        )
        self.skip = nn.Conv2d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()

    def forward(self, x):
        return F.leaky_relu(self.model(x) + self.skip(x), negative_slope=0.01)

# -----------------------------
# Conditional UNet
# -----------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, time_dim=128, num_classes=10):
        super().__init__()

        self.time_dim = time_dim

        # Zeit-Embedding Projektion
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, base_channels),
            nn.LeakyReLU(),
            nn.Linear(base_channels, base_channels)
        )

        # Label-Embedding
        self.label_emb = nn.Embedding(num_classes, base_channels)

        # Encoder
        self.enc1 = ResidualBlock(in_channels, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck/ Latent Space
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels)

        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: NamedTensor, t: NamedTensor, label: NamedTensor) -> NamedTensor:
        h = x.raw()

        e1 = self.enc1(h)

        # Zeit Embedding
        t_emb = sinusoidal_embedding(t.raw(), dim=self.time_dim)  # [B, time_dim]
        t_emb = self.time_projection(t_emb).unsqueeze(-1).unsqueeze(-1)
        t_emb = t_emb.expand(-1, -1, h.size(2), h.size(3))
        h = h + t_emb

        # Label Embedding
        y_emb = self.label_emb(label.raw()).unsqueeze(-1).unsqueeze(-1)
        y_emb = y_emb.expand(-1, -1, h.size(2), h.size(3))
        h = h + y_emb

        # Encoder

        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder + Skip Connections
        d3 = self.up3(b)
        # d3: Tensor mit Shape [Batch, Channel, Height, Width]
        # d3 auf die Shape von e3 interpolieren.
        d3 = F.interpolate(d3, size=(e3.size(2), e3.size(3)), mode='nearest')
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=(e2.size(2), e2.size(3)), mode='nearest')
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=(e1.size(2), e1.size(3)), mode='nearest')
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        return NamedTensor(out, (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))

# -----------------------------
# ODE - Ordinary Differntial Equations
# -----------------------------
class ODEFunc(nn.Module):
    def __init__(self, model, label: NamedTensor):
        super().__init__()
        self.model = model
        self.label = label

    def forward(self, t, x):
        # x in NamedTensor konvertieren
        x_named = NamedTensor(x, (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))

        # t als NamedTensor für Zeit-Embedding
        tb = NamedTensor(t.expand(x_named.raw().size(0), 1), ("batch", "time"))

        # Modelvorhersage als NamedTensor
        out_named = self.model(x_named, tb, self.label)

        # odeint erwartet normalen Tensor zurück
        return out_named.raw()

# -----------------------------
# AdaptiveScheduler
# -----------------------------
class AdaptiveScheduler:
    def __init__(self, optimizer, step_size=2, step_gamma=0.5,
                 plateau_factor=0.5, plateau_patience=1,
                 cosine_Tmax=5, overshoot_thresh=1.05, plateau_thresh=1e-4):
        """
        Adaptive Kombination von StepLR, ReduceLROnPlateau und CosineAnnealingLR.
        - StepLR: Standardfall
        - ReduceLROnPlateau: wenn der Loss stagniert
        - CosineAnnealingLR: wenn Overshooting erkannt wird
        """
        self.optimizer = optimizer

        # Scheduler Instanzen
        self.step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_gamma)
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=plateau_factor, patience=plateau_patience
        )
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_Tmax)

        # Tracking
        self.best_loss = float('inf')
        self.plateau_count = 0
        self.overshoot_detected = False
        self.active_scheduler = self.step_scheduler

        # Parameter
        self.overshoot_thresh = overshoot_thresh
        self.plateau_thresh = plateau_thresh

    def step(self, loss: float):
        """Aktualisiert Scheduler-Strategie anhand des Loss"""
        if loss < self.best_loss - self.plateau_thresh:
            self.best_loss = loss
            self.plateau_count = 0
            self.overshoot_detected = False
        elif loss > self.best_loss * self.overshoot_thresh:
            self.overshoot_detected = True
        else:
            self.plateau_count += 1

        # Scheduler wählen
        if self.overshoot_detected:
            self.active_scheduler = self.cosine_scheduler
        elif self.plateau_count > 3: # evtl. als variable auslagern
            self.active_scheduler = self.plateau_scheduler
        else:
            self.active_scheduler = self.step_scheduler

        if isinstance(self.active_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.active_scheduler.step(loss) # Brauch loss als Parameter
        else:
            self.active_scheduler.step()

    def get_lr(self):
        """Aktuelle Lernrate zurückgeben"""
        return self.optimizer.param_groups[0]['lr']

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

# -----------------------------
# Daten
# -----------------------------
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# -----------------------------
# Hyperparameter/Parameter
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(1, 1, 32, 128, 10).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr = 1e-4)

num_epochs = 5

filepath = './model/cfmunet_checkpoint.pth'

# -----------------------------
# Sampling via ODE (conditional)
# -----------------------------
@torch.no_grad()
def sample_flow_ode_cond(model, y_label, steps=200, method='dopri5'):
    model.eval()

    x0 = NamedTensor(torch.randn(1, 1, 28, 28, device=device),
                     (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))

    t_span = torch.linspace(0.0, 1.0, steps, device=device)

    y = NamedTensor(torch.tensor([y_label], device=device), (DIM_BATCH,))

    ode_func = ODEFunc(model, y)

    x_sequence = odeint(ode_func, x0.raw(), t_span, method=method)

    # x_sequence: Tensor [T, B, C, H, W] → NamedTensor pro Zeitpunkt
    x_sequence_named = [NamedTensor(x, (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH)) for x in x_sequence]

    return x_sequence_named[-1]  # Letzter Zeitpunkt


# -----------------------------
# Bild Generierung
# -----------------------------
def generate_image(model, y_labels: list[int]):
    images = []
    for label in y_labels:
        # Bild generieren
        sample = sample_flow_ode_cond(model, label)
        images.append(sample)

        # Plot
        plt.figure()
        plt.imshow(sample.raw().squeeze().numpy(), cmap="gray")
        plt.title(f"Conditional Flow Matching Sample: {label}")
        plt.axis("off")
        plt.show()

# -----------------------------
# Trainings Loop
# -----------------------------
"""
vgl.: "FLow Matching for Generative Modeling"; https://www.youtube.com/watch?v=7NNxK3CqaDk
"""
load_model = False
if os.path.exists(filepath) and load_model:
    global_epoch = load_checkpoint(model, optimizer, filepath, device=device)
else:
    print("No checkpoint found, starting from scratch.")
    global_epoch = 0
    save_checkpoint(model, optimizer, global_epoch, filepath)

total_epochs = global_epoch + num_epochs

# IDEE: Scheduler phasenweise und performanceabhängig -> TESTEN, ob sinnvoll!
# Schnelleres Lernen grober Features vor feinerer Anpassung.
#step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.5) # gamma = lr senkung -> evtl 0.1
# Stagniert der Loss über mehrere Epochen, wird die Lernrate angepasst.
# Das Modell kann kleine Geschwindigkeitsänderungen präziser lernen, ohne dass die Lernrate zu groß ist.
#plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = 2) # factor entspricht ca gamma
# Gegen Ende des Trainings kann die Lernrate gegen 0 konvergieren, um Overshooting zu verhindern
# und die finale Gewichtsanpassung zu stabilisieren.
#cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5)

scheduler = AdaptiveScheduler(optimizer)

for epoch in range(global_epoch, global_epoch + num_epochs):
    model.train()
    total_loss = 0.0

    for x1_raw, y_raw in train_loader:
        x1 = NamedTensor(x1_raw.to(device), (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))
        y = NamedTensor(y_raw.to(device), (DIM_BATCH,))

        # Zufällige Startverteilung
        x0 = NamedTensor(torch.randn_like(x1.raw()), (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))

        # Zufälliger Zeitpunkt t für jeden Batch-Eintrag, zwischen 0 und 1.
        # Wird für die lineare Interpolation zwischen x0 und x1 verwendet.
        t = NamedTensor(torch.rand(x1.raw().size(0), 1, device = device), (DIM_BATCH, DIM_TIME))

        # Lineare Interpolation zwischen x0 und x1
        x_t = NamedTensor((1 - t.raw().view(-1, 1, 1, 1)) * x0.raw() + t.raw().view(-1, 1, 1, 1) * x1.raw(),
                          (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))

        # Zielgeschwindigkeitsvektor
        v_target = x1 - x0

        # Vorhersage des Geschwindigkeitsvektors
        v_pred = model(x_t, t, y)

        # Loss: MESLoss
        loss = criterion(v_pred.raw(), v_target.raw())

        # Gradient zurücksetzen
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Nächster Schritt
        optimizer.step()

        total_loss += loss.item() * x1.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{total_epochs}, Loss: {avg_loss:.6f}")

    scheduler.step(avg_loss)

    generate_image(model, [3])

    checkpoint_threshold = epoch + 1
    if checkpoint_threshold % 5 == 0:
        save_checkpoint(model, optimizer, checkpoint_threshold, filepath)

    """match epoch:
        case e if 0 <= e <= 4:
            step_scheduler.step()
        case e if 5 <= e <= 14:
            plateau_scheduler.step(avg_loss)
        case e if 15 <= e <= 19:
            cosine_scheduler.step()"""


