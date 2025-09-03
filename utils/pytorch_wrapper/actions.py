import torch
import os
from abc import ABC, abstractmethod

class Action(ABC):
    # Abstract definition of an action used by a pipeline
    @abstractmethod
    def execute(self, pipeline, **kwargs):
        pass

class TrainEpoch(Action):
    def execute(self, pipeline, **kwargs):
        epoch = kwargs.get("epoch", 0)
        pipeline.trigger_event("epoch_start", epoch=epoch, pipeline=pipeline)
        model = pipeline.model
        optimizer = pipeline.optimizer
        criterion = pipeline.criterion
        device = pipeline.device

        model.train()
        running_loss = 0.0
        for images, labels in pipeline.train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            x = model.before_forward(images)
            outputs = model(x)
            outputs = model.after_forward(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(pipeline.train_loader)
        pipeline.trigger_event("epoch_end", epoch=epoch, loss=avg_loss, pipeline=pipeline)

class Evaluate(Action):
    def execute(self, pipeline, **kwargs):
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
        acc = 100 * correct / total
        print(f"[Evaluate] Accuracy: {acc:.2f}%")
        return acc

# Nicht für Checkpoints!?
class SaveModel(Action):
    def __init__(self, filepath, save_weights_only=True):
        """
        save_weights_only=True  -> save state_dict only
        save_weights_only=False -> save the whole model
        """
        self.filepath = filepath
        self.save_weights_only = save_weights_only

    def execute(self, pipeline, **kwargs):
        if self.save_weights_only:
            torch.save(pipeline.model.state_dict(), self.filepath)
            print(f"[SaveModel] Model weights saved to {self.filepath}")
        else:
            torch.save(pipeline.model, self.filepath)
            print(f"[SaveModel] Entire model saved to {self.filepath}")


class LoadModel(Action):
    def __init__(self, filepath, load_weights_only=True):
        """
        load_weights_only=True  -> load state_dict only
        load_weights_only=False -> load the whole model
        """
        self.filepath = filepath
        self.load_weights_only = load_weights_only

    def execute(self, pipeline, **kwargs):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File '{self.filepath}' does not exist")

        if self.load_weights_only:
            pipeline.model.load_state_dict(torch.load(self.filepath, map_location=pipeline.device))
            print(f"[LoadModel] Model weights loaded from {self.filepath}")
        else:
            # Attention: Overrides the current model
            loaded_model = torch.load(self.filepath, map_location=pipeline.device)
            pipeline.model = loaded_model.to(pipeline.device)
            print(f"[LoadModel] Entire model loaded from {self.filepath}")

class PrintConv2D(Action):
    def execute(self, pipeline, **kwargs):
        pass