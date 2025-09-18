# -----------------------------
# Trainer
# -----------------------------
import os
from matplotlib import pyplot as plt
from named_tensors import *

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, train_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for x_raw, y_raw in self.train_loader:
            x = NamedTensor(x_raw.to(self.device), (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))
            y = NamedTensor(y_raw.to(self.device), (DIM_BATCH,))
            x0 = NamedTensor(torch.randn_like(x.raw()), (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))
            t = NamedTensor(torch.rand(x.raw().size(0), 1, device=self.device), (DIM_BATCH, DIM_TIME))
            x_t = NamedTensor((1 - t.raw().view(-1,1,1,1)) * x0.raw() + t.raw().view(-1,1,1,1) * x.raw(), (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))
            v_target = x - x0
            v_pred = self.model(x_t, t, y)
            loss = self.criterion(v_pred.raw(), v_target.raw())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.raw().size(0)
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss

    def generate_image(self, y_labels: list[int], epoch: int):
        os.makedirs("./artifacts", exist_ok=True)
        for label in y_labels:
            sample = NamedTensor(torch.randn(1,1,28,28, device=self.device), (DIM_BATCH, DIM_CHANNELS, DIM_HEIGHT, DIM_WIDTH))
            plt.imshow(sample.raw().squeeze().cpu(), cmap="gray")
            # filepath variable
            filename = f"./artifacts/image_{label}_epoch_{epoch + 1}.png"
            plt.savefig(filename)
            plt.close()
            print(f"Saved image: {filename}")

# Das selbe wie Trainer, nur auf anderem Datensatz
class FeedbackTrainer:
    pass

# Stubsen
class FeedbackBoostTrainer:
    pass