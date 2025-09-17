import sys
import threading
import time
import torch
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QTabWidget
)
from PySide6.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Training Manager ---
class TrainingManager:
    def __init__(self):
        self.flags = {"intercept": False, "mode": "Normal", "stop": True}
        self.info = {"loss": [], "last_image": None}
        self.feedback_data = []
        self.lock = threading.Lock()

    def set_flag(self, key, value):
        with self.lock:
            self.flags[key] = value

    def get_flag(self, key):
        with self.lock:
            return self.flags[key]

    def update_info(self, loss=None, image=None):
        with self.lock:
            if loss is not None:
                self.info["loss"].append(loss)
            if image is not None:
                self.info["last_image"] = image

    def get_info(self):
        with self.lock:
            return self.info.copy()

    def update_feedback(self, label, image, feedback):
        with self.lock:
            self.feedback_data.append({"label": label, "image": image, "feedback": feedback})

    def get_feedback(self):
        with self.lock:
            data = self.feedback_data.copy()
            self.feedback_data.clear()
            return data

# --- Human Feedback Widget ---
class HumanFeedbackWidget(QWidget):
    def __init__(self, manager, max_images=5):
        super().__init__()
        self.manager = manager
        self.max_images = max_images
        self.selected = [False] * max_images
        self.labels = []
        self.images = []
        self.init_ui()

    def init_ui(self):
        self.main_layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()
        self.main_layout.addLayout(self.image_layout)

        self.image_widgets = []
        for i in range(self.max_images):
            vbox = QVBoxLayout()
            label_widget = QLabel(f"Label {i}")
            img_widget = QLabel()
            img_widget.setFixedSize(128, 128)
            img_widget.mousePressEvent = self.make_toggle(i)
            vbox.addWidget(label_widget)
            vbox.addWidget(img_widget)
            self.image_layout.addLayout(vbox)
            self.image_widgets.append((label_widget, img_widget))

        self.feedback_button = QPushButton("Feedback")
        self.feedback_button.setStyleSheet("background-color: blue; color: white;")
        self.feedback_button.clicked.connect(self.send_feedback)
        self.main_layout.addWidget(self.feedback_button)
        self.setLayout(self.main_layout)

    def make_toggle(self, index):
        def toggle(event):
            if not self.images:  # Klick nur möglich, wenn Bilder vorhanden sind
                return
            self.selected[index] = not self.selected[index]
            border = "3px solid green" if self.selected[index] else "2px solid gray"
            self.image_widgets[index][1].setStyleSheet(f"border: {border};")
        return toggle

    def update_images(self, labels, images):
        self.labels = labels
        self.images = images
        for i, (label, img_tensor) in enumerate(zip(labels, images)):
            label_widget, img_widget = self.image_widgets[i]
            label_widget.setText(label)
            img = (img_tensor.detach().permute(1, 2, 0).cpu().numpy().copy() * 255).astype(np.uint8)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img).scaled(128, 128)
            img_widget.setPixmap(pixmap)
            self.selected[i] = False
            img_widget.setStyleSheet("border:2px solid gray;")

    def send_feedback(self):
        if not self.images:
            return
        for sel, label, img in zip(self.selected, self.labels, self.images):
            feedback_value = 1 if sel else 0
            self.manager.update_feedback(label, img, feedback_value)
        print("Feedback gesendet")
        # Nach Feedback zurücksetzen
        self.selected = [False] * len(self.selected)
        self.labels = []
        self.images = []
        for _, img_widget in self.image_widgets:
            img_widget.clear()
            img_widget.setStyleSheet("border:2px solid gray;")

# --- Dummy Model ---
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        return self.conv(x)

# --- Dummy Training ---
def nn_training(model, device, manager: TrainingManager, epochs_to_run=5, batch_size=4, batches_per_epoch=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs_to_run):
        for batch_idx in range(batches_per_epoch):
            if manager.get_flag("intercept"):
                torch.save(model.state_dict(), f"model_epoch{epoch+1}.pt")
                manager.set_flag("intercept", False)
                return
            while manager.get_flag("stop"):
                time.sleep(0.1)
            x = torch.randn(batch_size, 3, 64, 64).to(device)
            y = torch.randn(batch_size, 3, 64, 64).to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            manager.update_info(loss=loss.item(), image=output[0])
            time.sleep(0.05)

    # Nach letzter Epoche automatisch stoppen
    manager.set_flag("stop", True)

# --- Training GUI ---
class TrainingGUI(QWidget):
    def __init__(self, manager: TrainingManager, human_feedback_widget, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.hf_widget = human_feedback_widget
        self.model = DummyModel()
        self.device = torch.device("cpu")
        self.training_thread = None
        self.batch_size = 4
        self.batches_per_epoch = 20
        self.init_ui()
        self.start_timer()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Steuerleiste
        control_layout = QHBoxLayout()
        self.toggle_button = QPushButton("Continue")
        self.toggle_button.clicked.connect(self.toggle_training)
        control_layout.addWidget(self.toggle_button)

        self.intercept_button = QPushButton("Interception Flag")
        self.intercept_button.setStyleSheet("background-color: blue; color: white;")
        self.intercept_button.clicked.connect(lambda: self.manager.set_flag("intercept", True))
        control_layout.addWidget(self.intercept_button)

        main_layout.addLayout(control_layout)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Normal Tab
        tab_normal = QWidget()
        layout_normal = QVBoxLayout()
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        layout_normal.addWidget(self.canvas)

        bottom_layout = QHBoxLayout()

        # Linke Seite (Bild)
        self.image_label = QLabel("Letztes Bild")
        self.image_label.setFixedSize(60, 150)
        bottom_layout.addWidget(self.image_label)

        # Rechte Seite
        right_layout = QVBoxLayout()
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Training Set (Epochen)"))
        self.epoch_input = QLineEdit("10")
        train_layout.addWidget(self.epoch_input)

        train_layout.addWidget(QLabel("Batchsize"))
        self.batchsize_field = QLineEdit(str(self.batch_size))
        self.batchsize_field.setReadOnly(True)
        train_layout.addWidget(self.batchsize_field)

        train_layout.addWidget(QLabel("Batches"))
        self.batches_field = QLineEdit(str(self.batches_per_epoch))
        self.batches_field.setReadOnly(True)
        train_layout.addWidget(self.batches_field)
        right_layout.addLayout(train_layout)

        # Modell-Pfad
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(QLabel("Model Filepath"))
        self.model_path_field = QLineEdit("model_saved.pt")
        model_path_layout.addWidget(self.model_path_field)
        self.save_model_button = QPushButton("Save Model")
        self.save_model_button.setStyleSheet("background-color: blue; color: white;")
        self.save_model_button.clicked.connect(self.save_model)
        model_path_layout.addWidget(self.save_model_button)
        right_layout.addLayout(model_path_layout)

        # Graph-Pfad
        graph_path_layout = QHBoxLayout()
        graph_path_layout.addWidget(QLabel("Graph Filepath"))
        self.graph_path_field = QLineEdit("loss_graph.png")
        graph_path_layout.addWidget(self.graph_path_field)
        self.save_graph_button = QPushButton("Save Graph")
        self.save_graph_button.setStyleSheet("background-color: blue; color: white;")
        self.save_graph_button.clicked.connect(self.save_graph)
        graph_path_layout.addWidget(self.save_graph_button)
        right_layout.addLayout(graph_path_layout)

        bottom_layout.addLayout(right_layout)
        layout_normal.addLayout(bottom_layout)
        tab_normal.setLayout(layout_normal)
        self.tabs.addTab(tab_normal, "Normal")

        # Human Feedback Tab
        tab_hf = QWidget()
        layout_hf = QVBoxLayout()
        self.sample_button = QPushButton("Sample")
        self.sample_button.setStyleSheet("background-color: blue; color: white;")
        self.sample_button.clicked.connect(self.generate_samples)
        layout_hf.addWidget(self.sample_button)
        layout_hf.addWidget(self.hf_widget)
        tab_hf.setLayout(layout_hf)
        self.tabs.addTab(tab_hf, "Human Feedback")

        self.tabs.currentChanged.connect(self.tab_changed)
        self.setLayout(main_layout)
        self.setWindowTitle("Flow-Matching GUI")
        self.show()

    def start_timer(self):
        from PySide6.QtCore import QTimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100)

    def toggle_training(self):
        current_stop = self.manager.get_flag("stop")
        self.manager.set_flag("stop", not current_stop)
        if not self.training_thread or not self.training_thread.is_alive():
            epochs_to_run = int(self.epoch_input.text())
            self.training_thread = threading.Thread(
                target=nn_training,
                args=(self.model, self.device, self.manager, epochs_to_run, self.batch_size, self.batches_per_epoch),
                daemon=True
            )
            self.training_thread.start()

    def save_model(self):
        if self.manager.get_flag("stop"):
            path = self.model_path_field.text() or "model_saved.pt"
            torch.save(self.model.state_dict(), path)
            print(f"Model saved as {path}")
        else:
            print("Training läuft – Model kann nur im gestoppten Zustand gespeichert werden.")

    def save_graph(self):
        path = self.graph_path_field.text() or "loss_graph.png"
        self.figure.savefig(path)
        print(f"Graph saved as {path}")

    def generate_samples(self):
        x = torch.randn(5, 3, 64, 64)
        model = DummyModel()
        with torch.no_grad():
            images = model(x)
        labels = [f"Pred {i}" for i in range(5)]
        self.hf_widget.update_images(labels, [images[i] for i in range(5)])

    def tab_changed(self, index):
        self.manager.set_flag("mode", self.tabs.tabText(index))

    def update_gui(self):
        info = self.manager.get_info()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(info["loss"], label="Loss")
        ax.set_title("Loss pro Batch über Zeit")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw()

        if info["last_image"] is not None:
            img_tensor = info["last_image"]
            img = (img_tensor.detach().permute(1, 2, 0).cpu().numpy().copy() * 255).astype(np.uint8)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            self.image_label.setPixmap(pixmap)

        if self.manager.get_flag("stop"):
            self.toggle_button.setText("Continue")
            self.toggle_button.setStyleSheet("background-color: green; color: white;")
        else:
            self.toggle_button.setText("Stop")
            self.toggle_button.setStyleSheet("background-color: red; color: white;")

# --- Main ---
if __name__ == "__main__":
    manager = TrainingManager()
    app = QApplication(sys.argv)
    hf_widget = HumanFeedbackWidget(manager)
    gui = TrainingGUI(manager, hf_widget)
    sys.exit(app.exec())
