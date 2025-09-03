from abc import ABC, abstractmethod
from datetime import datetime

import torch
import os

from actions import Action


class Event(ABC):
    # Definition of an event with the option to pass data
    @abstractmethod
    def on_event(self, event_name: str, **event_data):
        pass

class EventHandler:
    def __init__(self):
        self.events: list[Event] = []

    def register_event(self, event: Event):
        self.events.append(event)

    def trigger_event(self, event_name: str, **event_data):
        for event in self.events:
            event.on_event(event_name, **event_data)

class Pipeline(EventHandler):
    def __init__(self, model, optimizer, train_loader, test_loader, criterion, device="cpu"):
        super().__init__()
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.actions = {} # The actions used by the pipeline

    def register_action(self, name: str, action: Action):
        self.actions[name] = action

    def run_action(self, name: str, **kwargs):
        return self.actions[name].execute(self, **kwargs)

# ============================================================
# Beispiel Events
# ============================================================

class FreezingLogger(Event):
    def on_event(self, event_name: str, **kwargs):
        # Logs the freezing status of the layers per epoch
        if event_name == "epoch_start":
            model = kwargs["pipeline"].model
            print(f"[FreezingLogger] Layer Freezing Status at Epoch {kwargs['epoch']+1}:")
            for name, module in model._name_to_module.items():
                grad_status = any(p.requires_grad for p in module.parameters())
                print(f"  {name}: {'Trainable' if grad_status else 'Frozen'}")

#TODO: Kann raus?
class ModelCheckpoint(Event):
    def __init__(self, directory="checkpoints", monitor=None):
        self.directory = directory
        self.monitor = monitor or ["loss"]
        os.makedirs(self.directory, exist_ok=True)

    def on_event(self, event_name: str, **kwargs):
        if event_name == "epoch_end":
            model = kwargs["pipeline"].model
            epoch = kwargs["epoch"]
            logs = {"loss": kwargs.get("loss", 0.0)}
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_epoch{epoch+1}"
            metric_str = "_".join([f"{m}{logs[m]:.4f}" for m in logs if m in self.monitor])
            if metric_str:
                filename += f"_{metric_str}"
            filepath = os.path.join(self.directory, filename + ".pth")
            torch.save(model.state_dict(), filepath)
            print(f"[Checkpoint] Model saved to {filepath}")

#TODO: duplicate in actions.py
class MetricsLogger(Event):
    # Logs loss and accuracy at the end of each epoch
    def on_event(self, event_name: str, **kwargs):
        if event_name == "epoch_end":
            loss = kwargs.get("loss", 0.0)
            pipeline = kwargs.get("pipeline")
            accuracy = 0
            if pipeline is not None:
                model = pipeline.model
                test_loader = pipeline.test_loader
                device = pipeline.device
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        x = model.before_forward(images)
                        outputs = model(x)
                        outputs = model.after_forward(outputs)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total

            print(f"[MetricsLogger] Epoch {kwargs['epoch']+1}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")