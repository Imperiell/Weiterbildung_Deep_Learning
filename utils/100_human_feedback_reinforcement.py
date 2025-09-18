import os
from datetime import datetime
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torch.nn.functional as F
import threading
import keyboard

import ssl
import certifi

ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context



# KI für Konzeption (Komponenten und Schritte planen) und Debugging des Codes verwendet.
# Adaptionen aus Papers oder anderen Quellen sind an entsprechender Stelle vermerkt.

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
vgl.: "Inside Sinusoidal Position Embeddings: A Sense of Order"; https://learnopencv.com/sinusoidal-position-embeddings
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
# Conditional Dynamic UNet
# -----------------------------
class DynamicUNet(nn.Module):
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

        # Encoder: ModuleList -> dynamisch Channels anpassen
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = in_channels
        for mult in [1, 2, 4]:
            out_ch = base_channels * mult
            self.encoders.append(ResidualBlock(prev_channels, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            prev_channels = out_ch

        # Latent Space
        self.bottleneck = ResidualBlock(prev_channels, prev_channels * 2)

        # Decoder: ModuleList -> dynamisch Channels zusammenführen
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        decoder_channels = [prev_channels * 2, base_channels * 4, base_channels * 2]
        for in_ch, out_ch in zip(decoder_channels, [base_channels * 4, base_channels * 2, base_channels]):
            self.upsamples.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            self.decoders.append(ResidualBlock(in_ch, out_ch))  # in_ch = skip+upsample

        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: NamedTensor, t: NamedTensor, label: NamedTensor) -> NamedTensor:
        h = x.raw()

        skip_connections = []

        # 1. Encoder
        h = self.encoders[0](h)
        skip_connections.append(h)
        h = self.pools[0](h)

        # Zeit-Embedding
        t_emb = sinusoidal_embedding(t.raw(), dim=self.time_dim)
        t_emb = self.time_projection(t_emb).unsqueeze(-1).unsqueeze(-1)
        t_emb = t_emb.expand(-1, -1, h.size(2), h.size(3))
        h = h + t_emb

        # Label-Embedding
        y_emb = self.label_emb(label.raw()).unsqueeze(-1).unsqueeze(-1)
        y_emb = y_emb.expand(-1, -1, h.size(2), h.size(3))
        h = h + y_emb

        # Restliche Encoder
        # Sammelt Aktivierungen für das Einspeißen in Decoder
        for enc, pool in zip(self.encoders[1:], self.pools[1:]):
            h = enc(h)
            skip_connections.append(h)
            h = pool(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder (mit Skip Connections)
        # Skip Connections müssen reversed werden, damit die Werte an der richtigen Stelle eingefügt werden.
        # Hier wird die U-Form des UNets deutlich:
        # Das erste Downsampling des Encoders liefert den Skip-Wert für das letzte Upsampling des Decoders.
        # Downsampling = (d0, d1, ..., dn)
        # Upsampling = (u0, u1, ..., un)
        # Pairs = ((dn, u0), (dn-1, u1), ..., (d0, un))
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skip_connections)):
            h = up(h)
            h = F.interpolate(h, size=(skip.size(2), skip.size(3)), mode='nearest')
            h = dec(torch.cat([h, skip], dim=1))

        # Sigmoid: Gut für normalisierte Bilder
        out = torch.sigmoid(self.out_conv(h))
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
        tb = NamedTensor(t.expand(x_named.raw().size(0), 1), (DIM_BATCH, DIM_TIME))

        # Modelvorhersage als NamedTensor
        out_named = self.model(x_named, tb, self.label)

        # odeint erwartet normalen Tensor zurück
        return out_named.raw()

# -----------------------------
# AdaptiveScheduler
# -----------------------------
#TODO: Für Human Feedback LR höher setzen. -> Automatisieren
class AdvancedAdaptiveScheduler:
    def __init__(self, optimizer, params_dict=None):
        if params_dict is None:
            params_dict = {}

        self.optimizer = optimizer

        # Lernraten
        self.lr_normal = params_dict.get('lr_normal', 1e-4)
        self.lr_plateau = params_dict.get('lr_plateau', 5e-5)
        self.lr_precision = params_dict.get('lr_precision', 1e-5)
        self.lr_warmup_start = params_dict.get('lr_warmup_start', 1e-3)  # beginnt hoch
        self.lr_human = params_dict.get('lr_human', 2e-4)

        # Scheduler Instanzen
        self.step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params_dict.get('step_size', 2),
                                                        gamma=params_dict.get('step_gamma', 0.5))
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=params_dict.get('plateau_factor', 0.5),
            patience=params_dict.get('plateau_patience', 1)
        )
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                     T_max=params_dict.get('cosine_Tmax', 5))

        # Tracking
        self.best_loss = float('inf')
        self.plateau_count = 0
        self.overshoot_detected = False
        self.active_scheduler = self.step_scheduler

        # Warmup
        self.warmup_steps = params_dict.get('warmup_steps', 100)
        self.step_num = 0
        self.lr_base = self.lr_normal

        # initiale LR auf lr_warmup_start setzen
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_warmup_start

        # Sprung-Parameter
        self.jump_prob = params_dict.get('jump_prob', 0.1)
        self.jump_factor = params_dict.get('jump_factor', 0.05)

    def step(self, loss: float, human_input=False):
        self.step_num += 1

        # Warmup linear von lr_warmup_start → lr_base
        if self.step_num <= self.warmup_steps:
            lr = self.lr_warmup_start - (self.lr_warmup_start - self.lr_base) * (self.step_num / self.warmup_steps)
            self._set_lr(lr)
        else:
            # Normales adaptives Scheduling
            if loss < self.best_loss - 1e-4:
                self.best_loss = loss
                self.plateau_count = 0
                self.overshoot_detected = False
                self._set_lr(self.lr_normal)
            elif loss > self.best_loss * 1.05:
                self.overshoot_detected = True
                self._set_lr(self.lr_precision)
            else:
                self.plateau_count += 1
                if self.plateau_count > 3:
                    self._set_lr(self.lr_plateau)

            # Human Input override
            if human_input:
                self._set_lr(self.lr_human)

        # Stochastic LR Jump
        self._maybe_jump_lr()

        # Scheduler step
        match type(self.active_scheduler):
            case optim.lr_scheduler.ReduceLROnPlateau:
                self.active_scheduler.step(loss)
            case _:
                self.active_scheduler.step()

    def _maybe_jump_lr(self):
        import random
        if random.random() < self.jump_prob:
            direction = 1 if random.random() > 0.5 else -1
            new_lr = self.get_lr() * (1 + direction * self.jump_factor)
            self._set_lr(new_lr)

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
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
model = DynamicUNet(1, 1, 32, 128, 10).to(device)
#criterion = nn.SmoothL1Loss() # nn.MSELoss ist oft zu "weich" für Bilder mit scharfen Kanten wie MNIST.
# SmoothL1Loss ist stabiler für kleine Gradienten und bessere Kantenwiedergabe.
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr = 1e-4)

num_epochs = 100

filepath = './model/cfm_dynamic_unet_checkpoint.pth'

# Feedback Storage
feedback_data_path = "./feedback_data.pt"

if os.path.exists(feedback_data_path):
    feedback_dataset = torch.load(feedback_data_path)
else:
    feedback_dataset = []

# -----------------------------
# Sampling via ODE (conditional)
# -----------------------------
@torch.no_grad()
def sample_flow_ode_cond(model, y_label, steps=200, method='dopri5'):
    model.eval()

    x0 = NamedTensor(torch.randn(1, 1, 28, 28, device=device),
                     (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))

    t_span = torch.linspace(0.0, 1.0, steps, device=device)

    #TODO: DIM für label hinzufügen?
    y = NamedTensor(torch.tensor([y_label], device=device), (DIM_BATCH,))

    ode_func = ODEFunc(model, y)

    x_sequence = odeint(ode_func, x0.raw(), t_span, method=method)

    # x_sequence: Tensor [T, B, C, H, W] → NamedTensor pro Zeitpunkt
    x_sequence_named = [NamedTensor(x, (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH)) for x in x_sequence]

    return x_sequence_named[-1]  # Letzter Zeitpunkt

# Feedback Training
def feedback_boost_training(model, optimizer, num_samples = 10, augment = False):
    global feedback_dataset
    model.train()
    samples, labels = [], []

    for i in range(num_samples):
        label = int(input(f"Label für Sample {i + 1}: "))
        sample = sample_flow_ode_cond(model, label)

        plt.imshow(sample.raw().squeeze().cpu().numpy(), cmap = "gray")
        plt.title(f"Sample {i + 1} - Label {label}")
        plt.show()

        grading = input("Ist das Bild als die gelabelte Zahl erkennbar? 0 = nein/ 1 = ja:")
        while grading not in ("0", "1"):
            grading = input("Nur 0 und 1 sind erlaubt. Bitte erneut versuchen: ")
        if int(grading) == 1:
            samples.append(sample.raw())
            labels.append(label)

    if len(samples) > 0:
        x_feedback = torch.cat(samples, dim = 0)
        y_feedback = torch.tensor(labels, device = device)
        feedback_dataset.append((x_feedback, y_feedback))
        torch.save(feedback_dataset, feedback_data_path)
        print(f"Feedback gespeichert ({len(samples)} Bilder)")

        # Training auf Feedback
        optimizer.zero_grad()
        x_named = NamedTensor(x_feedback, (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))
        y_named = NamedTensor(y_feedback, (DIM_BATCH,))
        t_named = NamedTensor(torch.zeros(len(x_feedback), 1, device=device), (DIM_BATCH, DIM_TIME))
        v_pred = model(x_named, t_named, y_named)
        # Das Modell soll Richtung der erkennbaren Bilder gepushed werden.
        # Wenn das NN gut erkennbare Bilder erzeugt, soll es wissen, dass diese gut sind und weiter
        # in diese Richtung arbeiten.
        # Delta-Training -> Es wird ein Gradient in Richtung der bereits positiven Samples erzeugt,
        # was ein Delta zur Verstärkung dieser Richtung ist.
        v_target = v_pred

        loss = criterion(v_pred.raw(), v_target.raw())
        loss.backward()
        optimizer.step()
        print(f"Feedback-Delta-Training abgeschlossen, Loss={loss.item():.6f}")
    else:
        print("Keine positiven Samples für Feedback erhalten.")

# Keyboard Listener für Soft-Stop
feedback_flag = False

def keyboard_listener(event):
    global feedback_flag
    feedback_flag = True
    print("Feedback-Flag aktiviert!")

# keyboard.on_press_key('space', keyboard_listener)
# -----------------------------
# Bild Generierung + Speichern
# -----------------------------
def generate_image(model, y_labels: list[int], epoch: int):
    os.makedirs("./artifacts", exist_ok=True)

    for label in y_labels:
        # Bild generieren
        sample = sample_flow_ode_cond(model, label)

        # Plot und Speichern
        plt.figure()
        plt.imshow(sample.raw().squeeze().cpu().numpy(), cmap="gray")
        plt.title(f"Conditional Flow Matching Sample: {label}")
        plt.axis("off")

        # Speichern
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"./artifacts/image_{label}_epoch_{epoch}_{now}.png"
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved image: {filename}")

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

# IDEE: Scheduler phasenweise und performanceabhängig -> TESTEN, ob sinnvoll!
# Schnelleres Lernen grober Features vor feinerer Anpassung.
#step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.5) # gamma = lr senkung -> evtl 0.1
# Stagniert der Loss über mehrere Epochen, wird die Lernrate angepasst.
# Das Modell kann kleine Geschwindigkeitsänderungen präziser lernen, ohne dass die Lernrate zu groß ist.
#plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.5, patience = 2) # factor entspricht ca gamma
# Gegen Ende des Trainings kann die Lernrate gegen 0 konvergieren, um Overshooting zu verhindern
# und die finale Gewichtsanpassung zu stabilisieren.
#cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5)

params_dict = {
    'lr_normal': 1e-4,
    'lr_plateau': 5e-5,
    'lr_precision': 1e-5,
    'lr_warmup_start': 1e-3,
    'lr_human': 2e-4,
    'warmup_steps': 10,
    'step_size': 2,
    'step_gamma': 0.5,
    'cosine_Tmax': 5,
    'jump_prob': 0.2,
    'jump_factor': 0.1
}

scheduler = AdvancedAdaptiveScheduler(optimizer, params_dict)

for epoch in range(global_epoch, total_epochs):
    print("Drücke 'Space', um eine Feedback-Session nach der aktuellen Epoche zu starten.")
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

    generate_image(model, [3], epoch = epoch + 1)

    checkpoint_threshold = epoch + 1
    if checkpoint_threshold % 5 == 0:
        save_checkpoint(model, optimizer, checkpoint_threshold, filepath)

    # Soft-Stop + Feedback
    if feedback_flag:
        print(f"\n Soft-Stop Feedback-Session nach Epoche {epoch + 1} gestartet!")
        feedback_boost_training(model, optimizer, num_samples = 5)
        feedback_flag = False

