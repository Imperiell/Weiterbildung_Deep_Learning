import sys
import torch
import threading
import time
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox
)
from PySide6.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TrainingGUI(QWidget):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.intercept_flag = False
        self.loss_history = []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Modus Auswahl
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Normal", "Human Feedback"])
        layout.addWidget(self.mode_combo)

        # Loss Graph
        self.figure = Figure(figsize=(5, 3))
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
        self.stop_button.clicked.connect(self.set_intercept_flag)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)
        self.setWindowTitle("Flow-Matching Training GUI")
        self.show()

    def set_intercept_flag(self):
        self.intercept_flag = True

    def update_loss(self, loss):
        self.loss_history.append(loss)
        self.ax.clear()
        self.ax.plot(self.loss_history, label="Loss")
        self.ax.set_title("Loss über die Zeit")
        self.ax.set_xlabel("Batch")
        self.ax.set_ylabel("Loss")
        self.ax.legend()
        self.canvas.draw()

    def update_image(self, img_tensor):
        # img_tensor: [C,H,W] Tensor, Werte zwischen 0 und 1
        img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pixmap)


# Dummy Trainingsloop
def train(model, device, gui):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    for epoch in range(10):
        for batch_idx in range(50):
            if gui.intercept_flag:
                print("Interception aktiviert – Epoche speichern und abbrechen")
                gui.intercept_flag = False
                return

            # Fake Daten
            x = torch.randn(4, 3, 64, 64).to(device)
            y = torch.randn(4, 3, 64, 64).to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            gui.update_loss(loss.item())
            gui.update_image(output[0])

            time.sleep(0.1)  # simuliert Trainingszeit


if __name__ == "__main__":
    # Dummy Model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, x):
            return self.conv(x)


    app = QApplication(sys.argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DummyModel().to(device)

    gui = TrainingGUI(model, device)

    # Trainingsloop in Thread, damit GUI reagiert
    thread = threading.Thread(target=train, args=(model, device, gui))
    thread.start()

    sys.exit(app.exec())
