# -----------------------------
# ODE - Ordinary Differntial Equations
# -----------------------------
from torch import nn
from named_tensors import *


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