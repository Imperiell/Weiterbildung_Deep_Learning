import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
import os
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------
# EventHandler
# -------------------------
class EventHandler(ABC):
    def __init__(self):
        self.events: List["Event"] = []

    def register_event(self, event: "Event"):
        self.events.append(event)

    def unregister_event(self, event: "Event"):
        if event in self.events:
            self.events.remove(event)

    def handle_event(self, event_name: str, **event_data):
        for event in self.events:
            event.on_event(event_name, **event_data)


# -------------------------
# Abstraktes Event
# -------------------------
class Event(ABC):
    @abstractmethod
    def on_event(self, event: str, **event_data):
        pass


# -------------------------
# TensorBoard Logger
# -------------------------
class TensorBoardLogger(Event):
    def __init__(self, log_dir="runs/default"):
        self.writer = SummaryWriter(log_dir)

    def on_event(self, event_name: str, **event_data):
        match event_name:
            case "train_start":
                print(f"TensorBoard logging started at {self.writer.log_dir}")

            case "epoch_end":
                epoch = event_data.get("epoch")
                logs = event_data.get("logs", {})
                for k, v in logs.items():
                    self.writer.add_scalar(k, v, epoch)

            case "train_end":
                self.writer.close()

            #case _:
                #print(f"Unknown event {event_name} in {self.__class__.__name__}")


# -------------------------
# ModelCheckpoint
# -------------------------
class ModelCheckpoint(Event):
    def __init__(self, directory="modelcheckpoints", monitor=None, save_last=True, save_interval: Optional[int] = None):
        self.directory = directory
        self.monitor = monitor or ["loss"]
        self.save_last = save_last
        self.save_interval = save_interval

        if not os.path.exists(self.directory):
            try:
                os.makedirs(self.directory)
                print(f"Created directory {self.directory} for model checkpoints")
            except Exception as e:
                raise ValueError(f"Cannot create directory '{self.directory}' in {self.__class__.__name__}: {e}")

    def _save_model(self, model, epoch, logs, prefix="epoch"):
        if model is None:
            raise ValueError(f"'model' is None in {self.__class__.__name__}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # falls epoch None ist (z. B. beim train_end)
        if epoch is not None:
            filename = f"{timestamp}_{prefix}{epoch + 1}"
        else:
            filename = f"{timestamp}_{prefix}"

        if logs:
            metric_str = "_".join([f"{m}{logs[m]:.4f}" for m in logs if m in self.monitor])
            if metric_str:
                filename += f"_{metric_str}"

        filepath = os.path.join(self.directory, filename + ".pth")
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def on_event(self, event_name: str, **event_data):
        match event_name:
            case "epoch_end":
                model = event_data.get("model")
                epoch = event_data.get("epoch")
                logs = event_data.get("logs", {})

                if model is None:
                    raise ValueError(f"'model' is None in {self.__class__.__name__}")
                if epoch is None:
                    raise ValueError(f"'epoch' is None in {self.__class__.__name__}")

                # Speichern nach Interval (falls gesetzt)
                if self.save_interval and (epoch + 1) % self.save_interval == 0:
                    self._save_model(model, epoch, logs, prefix="interval")

            case "train_end":
                model = event_data.get("model")
                epoch = event_data.get("epoch", -1)
                logs = event_data.get("logs", {})

                if self.save_last and model is not None:
                    self._save_model(model, epoch, logs, prefix="last")

            case "print_weights":
                model = event_data.get("model")

                try:
                    # Ersten Layer (Conv2d) finden
                    first_layer = None
                    for name, module in self.model.named_modules(): #TODO Adjust for pattern
                        if isinstance(module, torch.nn.Conv2d):
                            first_layer = module
                            break

                    if first_layer is None:
                        print("Kein Conv2d-Layer im Modell gefunden.")
                        return

                    weights = first_layer.weight.data.clone().cpu()  # (out_channels, in_channels, H, W)
                    if weights.shape[1] == 3:  # RGB-Filter
                        weights = torch.permute(weights, (0, 2, 3, 1))  # -> (out_channels, H, W, 3)

                    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
                    for i, ax in enumerate(axes.flat):
                        if i >= weights.shape[0]:
                            ax.axis("off")
                            continue
                        w = weights[i]
                        w = (w - w.min()) / (w.max() - w.min())  # Normalisieren
                        ax.imshow(w.numpy())
                        ax.set_title(f"F{i}", fontsize=8)
                        ax.axis("off")

                    plt.tight_layout()
                    plt.show()

                except Exception as e:
                    print(f"Fehler bei der Visualisierung: {e}")

            #case _:
                #print(f"Unknown event {event_name} in {self.__class__.__name__}")


# -------------------------
# PipelineConfig (mit Pydantic)
# -------------------------
class PipelineConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: nn.Module

    train_loader: DataLoader
    test_loader: DataLoader
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer
    device: str = "cpu"
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    events: Optional[List[Event]] = Field(default_factory=list)


# -------------------------
# Pipeline
# -------------------------
class Pipeline(EventHandler):
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config

        self.model = config.model.to(config.device)
        self.optimizer = config.optimizer
        self.criterion = config.criterion
        self.train_loader = config.train_loader
        self.test_loader = config.test_loader
        self.scheduler = config.scheduler
        self.device = config.device

        for cb in config.events:
            self.register_event(cb)

    def _build_event_data(self, **extra):
        base_data = {
            "model": self.model,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "scheduler": self.scheduler,
            "device": self.device,
        }
        base_data.update(extra)
        return base_data

    def handle_event(self, event_name: str, **kwargs):
        event_data = kwargs.get("event_data", {})
        match event_name:
            case "train":
                self._train(event_data.get("num_epochs", 1))
            case "evaluate":
                return self._evaluate()

        super().handle_event(event_name, **event_data)
        return None

    def _train(self, num_epochs: int = 1):
        self.handle_event("train_start", event_data=self._build_event_data())

        for epoch in range(num_epochs):
            self.handle_event("epoch_start", event_data=self._build_event_data(epoch=epoch))

            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            logs = {"loss": avg_loss}
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

            self.handle_event("epoch_end", event_data=self._build_event_data(epoch=epoch, logs=logs))

            if self.scheduler:
                self.scheduler.step()

        self.handle_event("train_end", event_data=self._build_event_data(epoch=num_epochs - 1, logs=logs))
        self.handle_event(event_name="print_weights", event_data=self._build_event_data())

    def _evaluate(self):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy

# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(self, eventhandler: EventHandler):
        self.eventhandler = eventhandler

    def train(self, num_epochs: int = 1):
        self.eventhandler.handle_event("train", event_data={"num_epochs": num_epochs})

    def evaluate(self):
        return self.eventhandler.handle_event("evaluate", event_data={})


# -------------------------
# Beispiel-Setup
# -------------------------
if __name__ == "__main__":
    # Dummy-Daten
    X_train = torch.randn(500, 20)
    y_train = torch.randint(0, 2, (500,))
    X_test = torch.randn(100, 20)
    y_test = torch.randint(0, 2, (100,))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    config = PipelineConfig(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",
        events=[
            ModelCheckpoint(directory="checkpoints", monitor=["loss"], save_last=True, save_interval=2),
            TensorBoardLogger(log_dir="runs/exp1")
        ]
    )

    pipeline = Pipeline(config)
    trainer = Trainer(pipeline)

    trainer.train(num_epochs=5)
    trainer.evaluate()
