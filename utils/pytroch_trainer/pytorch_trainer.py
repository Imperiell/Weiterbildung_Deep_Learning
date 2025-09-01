import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


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

    def handle_event(self, event_name: str, **kwargs):
        for event in self.events:
            event.on_event(event_name, **kwargs)


# -------------------------
# Abstraktes Event
# -------------------------
class Event(ABC):
    @abstractmethod
    def on_event(self, event: str, **kwargs):
        pass


# -------------------------
# TensorBoard Logger
# -------------------------
class TensorBoardLogger(Event):
    def __init__(self, log_dir="runs/default"):
        self.writer = SummaryWriter(log_dir)

    def on_event(self, event_name: str, **kwargs):
        if event_name == "train_start":
            print(f"TensorBoard logging started at {self.writer.log_dir}")

        elif event_name == "epoch_end":
            epoch = kwargs.get("epoch")
            logs = kwargs.get("logs", {})
            for k, v in logs.items():
                self.writer.add_scalar(k, v, epoch)

        elif event_name == "train_end":
            self.writer.close()


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

        # events registrieren
        for cb in config.events:
            self.register_event(cb)

    def handle_event(self, event_name: str, **kwargs):
        # Pipeline reagiert auf Events
        if event_name == "train":
            self._train(kwargs.get("num_epochs", 1))
        elif event_name == "evaluate":
            return self._evaluate()

        # zusätzlich alle externen Events benachrichtigen
        super().handle_event(event_name, **kwargs)
        return None

    def _train(self, num_epochs: int = 1):
        self.handle_event("train_start")

        for epoch in range(num_epochs):
            self.handle_event("epoch_start", epoch=epoch)

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

            self.handle_event("epoch_end", epoch=epoch, logs=logs)

            if self.scheduler:
                self.scheduler.step()

        self.handle_event("train_end")

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
# Trainer (nur Events auslösen)
# -------------------------
class Trainer:
    def __init__(self, eventhandler: EventHandler):
        self.eventhandler = eventhandler

    def train(self, num_epochs: int = 1):
        self.eventhandler.handle_event("train", num_epochs=num_epochs)

    def evaluate(self):
        return self.eventhandler.handle_event("evaluate")


# -------------------------
# Beispiel-Setup
# -------------------------
if __name__ == "__main__":
    # Dummy-Daten!!!
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
        device="cpu"
    )

    pipeline = Pipeline(config)
    trainer = Trainer(pipeline)

    trainer.train(num_epochs=5)
    trainer.evaluate()
