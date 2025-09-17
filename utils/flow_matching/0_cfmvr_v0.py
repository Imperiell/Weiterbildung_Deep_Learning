import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# torchdiffeq
from torchdiffeq import odeint_adjoint as odeint  # oder odeint

# --- Encoder + Decoder ---
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (64,7,7)),
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,1,3,stride=2,padding=1,output_padding=1)
        )
    def forward(self, z):
        return self.net(z)

# --- Velocity Model ---
class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.fc_time = nn.Linear(1, 128)
        self.decoder = Decoder()

    def forward(self, x_t, t):
        z = self.encoder(x_t)
        t_embed = torch.sin(self.fc_time(t))
        z = z + t_embed
        v = self.decoder(z)
        return v

# --- Data ---
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data/mnist", train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VelocityNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training ---
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for x1, _ in train_loader:
        x1 = x1.to(device)
        x0 = torch.randn_like(x1)

        # Zeit t zufällig
        t = torch.rand(x1.size(0), 1, device=device)

        # Interpolation x_t
        x_t = (1 - t.view(-1,1,1,1)) * x0 + t.view(-1,1,1,1) * x1

        # Zielgeschwindigkeit
        v_target = x1 - x0

        v_pred = model(x_t, t)

        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * x1.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

# --- Sampling via ODE mit torchdiffeq ---
class ODEFunc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        tb = t.expand(x.size(0), 1)
        return self.model(x, tb)

@torch.no_grad()
def sample_flow_ode(model, steps=50, method='dopri5'):
    model.eval()
    # Startpunkt in der Rauschverteilung
    x0 = torch.randn(1, 1, 28, 28, device=device)

    # Zeitschritte von 0 bis 1
    t_span = torch.linspace(0.0, 1.0, steps, device=device)

    ode_func = ODEFunc(model)  # <-- jetzt ein nn.Module

    # odeint erwartet Input: func, initial_state, t_span
    xs = odeint(ode_func, x0, t_span, method=method)
    x_final = xs[-1]  # letzter Zeitschritt

    return x_final.cpu()

# --- Test-Sampling ---
sample = sample_flow_ode(model, steps=80, method='dopri5')
plt.imshow(sample.squeeze().numpy(), cmap='gray')
plt.title("MNIST Sample via ODE with torchdiffeq")
plt.axis('off')
plt.show()
