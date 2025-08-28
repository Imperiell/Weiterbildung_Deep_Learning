import torch
from torch.utils.tensorboard import SummaryWriter
import shutil, glob, time

# Basic Callback
class Callback:
    def on_epoch_end(self, epoch, logs=None):
        pass
    def on_train_end(self, logs=None):
        pass

class TensorBoardLogger(Callback):
    def __init__(self, base_log_dir = "runs/trial", clear_old = False):
        # Alte Ordner löschen
        if clear_old:
            for folder in glob.glob(f"{base_log_dir}_*"):
                shutil.rmtree(folder)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = f"{base_log_dir}_{timestamp}"
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard Logs: {self.writer.log_dir}")

    def on_train_start(self):
        print(f"TensorBoard Logs: {self.writer.log_dir}")
        print(f"Start TensorBoard: tensorboard --logdir {self.writer.log_dir}")

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        for k, v in logs.items():
            self.writer.add_scalar(k, v, epoch)

    def on_train_end(self, logs=None):
        self.writer.close()

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device = 'cpu', callbacks = None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        # Ort (Hardware) der Berechnung; CPU, GPU
        self.device = device
        self.callbacks = callbacks or [] # Liste von Callbacks

    def train(self, num_epochs = 1):
        # Modell in den Trainingsmodus setzen
        self.model.train()
        print(f"Start Training: num_epochs={num_epochs}")

        for epoch in range(num_epochs):
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Optimizer zurücksetzen
                self.optimizer.zero_grad()
                # Prediction/ Forward-Pass / Forward-Step
                outputs = self.model(images)
                # Fehler/ Loss berechnen
                loss = self.criterion(outputs, labels)
                # Backpropagation
                loss.backward()
                # Gewichte aktualisieren
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

            # Logs für Tensorboard
            logs = {'loss': avg_loss}
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in self.callbacks:
            callback.on_train_end()
            print("#######")
            for cb in self.callbacks:
                if hasattr(cb, "on_train_start"):
                    cb.on_train_start()
            print("########")

    def evaluate(self):
        # Modell in den Evaluierungs- oder Predictmodeus versetzen
        self.model.eval()

        # Anzahl: Korrekte Outputs
        correct = 0
        # Anzahl: Alle Outputs
        total = 0

        # Schaltet die Gradientenberechnung ab
        with torch.no_grad():
            # Iterativ durch die Daten in einem Batch
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                # Predict
                outputs = self.model(images)
                # Gibt zwei Werte zurück:
                # Maximalwert entlang der angegebenen Dimension
                # Index des maximalen Werts entlang dieser Dimension
                # dim = 1 -> sucht pro Zeile den max Wert
                # dim = 0 -> sucht pro Spalten den max Wert
                _, predicted = torch.max(outputs, 1)
                # Addiere die Anzahl der Outputs pro Batch zu der Gesamtsumme
                total += labels.size(0)
                # Zähle die richtigen Putputs pro Batch und addiere diese zu der Gesamtsumme
                # Achtung: sum() funktioniert in Pytorch (wenn es sich um Tensoren handelt) und Numpy(bei np.arrays),
                # aber nicht mit Listen in Python!
                #
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")

        return accuracy

"""
# Beispiel: Trainer mit Logger für Tensorboard
trainer = Trainer(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device="cpu",
    callbacks=[TensorBoardLogger("runs/mnist_demo")]
)

trainer.train(num_epochs=5)
trainer.evaluate()
"""
