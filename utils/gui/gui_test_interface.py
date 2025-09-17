import sys
import threading
import time
import torch
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox
from PySide6.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Interface zwischen GUI und NN ---
class NNInterface:
    def __init__(self):
        # Flags von GUI → NN
        self.flags = {
            "intercept": False,
            "mode": "Normal"
        }
        # Informationen von NN → GUI
        self.info = {
            "loss": [],
            "last_image": None
        }
        self.lock = threading.Lock()  # Thread-safe

    # GUI setzt Flag
    def set_flag(self, key, value):
        with self.lock:
            self.flags[key] = value

    # NN liest Flag
    def get_flag(self, key):
        with self.lock:
            return self.flags[key]

    # NN aktualisiert Infos
    def update_info(self, loss=None, image=None):
        with self.lock:
            if loss is not None:
                self.info["loss"].append(loss)
            if image is not None:
                self.info["last_image"] = image

    # GUI liest Infos
    def get_info(self):
        with self.lock:
            return self.info.copy()

# --- GUI ---
class TrainingGUI(QWidget):
    def __init__(self, interface: NNInterface):
        super().__init__()
        self.interface = interface
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Modus-Auswahl
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Normal", "Human Feedback"])
        self.mode_combo.currentTextChanged.connect(lambda text: self.interface.set_flag("mode", text))
        layout.addWidget(self.mode_combo)

        # Loss Graph
        self.figure = Figure(figsize=(5,3))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Loss über die Zeit")
        self.ax.set_xlabel("Batch")
        self.ax.set_ylabel("Loss")

        # Letztes Bild
        self.image_label = QLabel("Letztes Bild")
        layout.addWidget(self.image_label)

        # Interception Button
        self.stop_button = QPushButton("Interception Flag")
        self.stop_button.clicked.connect(lambda: self.interface.set_flag("intercept", True))
        layout.addWidget(self.stop_button)

        self.setLayout(layout)
        self.setWindowTitle("Flow-Matching GUI")
        self.show()

        # Timer für GUI-Updates
        self.start_timer()

    def start_timer(self):
        from PySide6.QtCore import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100)  # alle 100ms

    def update_gui(self):
        info = self.interface.get_info()
        # Loss-Graph aktualisieren
        self.ax.clear()
        self.ax.plot(info["loss"], label="Loss")
        self.ax.set_title("Loss über die Zeit")
        self.ax.set_xlabel("Batch")
        self.ax.set_ylabel("Loss")
        self.ax.legend()
        self.canvas.draw()

        # Letztes Bild anzeigen
        if info["last_image"] is not None:
            img_tensor = info["last_image"]
            img = (img_tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            self.image_label.setPixmap(pixmap)

# --- Dummy NN Training ---
def nn_training(model, device, interface: NNInterface):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(10):
        for batch_idx in range(50):
            # Flags prüfen
            if interface.get_flag("intercept"):
                print("Interception aktiviert – Speichern & Abbruch")
                torch.save(model.state_dict(), f"model_epoch{epoch+1}.pt")
                interface.set_flag("intercept", False)
                return

            mode = interface.get_flag("mode")
            # Optional: unterschiedliche Trainingslogik nach Mode
            # z.B. Human Feedback → andere Loss-Berechnung
            x = torch.randn(4,3,64,64).to(device)
            y = torch.randn(4,3,64,64).to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # Interface aktualisieren
            interface.update_info(loss=loss.item(), image=output[0])
            time.sleep(0.05)

# --- Dummy Model ---
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3,3,3,padding=1)
    def forward(self,x):
        return self.conv(x)

# --- Main ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DummyModel().to(device)
    interface = NNInterface()

    app = QApplication(sys.argv)
    gui = TrainingGUI(interface)

    # Training im separaten Thread starten
    thread = threading.Thread(target=nn_training, args=(model, device, interface), daemon=True)
    thread.start()

    sys.exit(app.exec())
