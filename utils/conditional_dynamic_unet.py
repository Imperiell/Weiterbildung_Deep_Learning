import torch
from torch import nn
import torch.nn.functional as F
from residual_block import *
from named_tensors import *
from time_embedding import *
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
