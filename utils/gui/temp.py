import sys
import threading
import time
import torch
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QLineEdit
)
from PySide6.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer

# Manager
class TrainingManager:
    def __init__(self):
        self.flags = {"mode": "Normal", "stop": True}
        self.info = {"loss": [], "last_image": None}
        self.feedback_data = []
        self.lock = threading.Lock()

    # flags
    def set_flag(self, key, value):
        with self.lock:
            self.flags[key] = value

    def get_flag(self, key):
        with self.lock:
            return self.flags[key]

    # info
    def update_info(self, loss = None, image = None):
        with self.lock:
            if loss is not None:
                self.info["loss"].append(loss)
            if image is not None:
                self.info["last_image"] = image

    def get_info(self):
        with self.lock:
            return self.info.copy()

    # feedback
    def update_feedback(self, label, image, feedback):
        with self.lock:
            self.feedback_data.append({"label": label, "image": image, "feedback": feedback})

    def get_feedback(self):
        with self.lock:
            data = self.feedback_data.copy()
            self.feedback_data.clear()
            return data

# Training Loop
def training_loop(manager: TrainingManager, max_epochs = 100):
    # Trainings loop hier
    pass

# Human Feedback Widget
class HumanFeedbackWidget(QWidget):
    def __init__(self, manager: TrainingManager, max_images = 5):
        super().__init__()
        self.manager = manager
        self.max_images = max_images
        self.selected = [False] * max_images # Initialisiert Liste mit False mit der Größe max_images
        self.labels = []
        self.images = []
        self.init_ui()
        self.samples_generated = False

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

        btn_layout = QHBoxLayout()
        self.sample_button = QPushButton("Sample")
        self.sample_button.setStyleSheet("background-color: blue; color: white")
        self.sample_button.clicked.connect(self.generate_samples)
        btn_layout.addWidget(self.sample_button)

        self.feedback_button = QPushButton("Feedback")
        self.feedback_button.setStyleSheet("background-color: blue; color: white")
        self.feedback_button.clicked.connect(self.send_feedback)
        btn_layout.addWidget(self.feedback_button)

        self.main_layout.addLayout(btn_layout)
        self.setLayout(self.main_layout)

    def make_toggle(self, index):
        def toggle(event):
            if not self.samples_generated:
                return

            self.selected[index] = not self.selected[index]
            border = "3px solid blue" if self.selected[index] else "2px solid gray"
            self.image_widgets[index][1].setStyleSheet(f"border: {border};")

        return toggle

    def generate_samples(self):
        # Labels erstellen
        # self.labels =
        pass

    def send_feedback(self):
        if not self.samples_generated:
            return

        for selected, label, image in zip(self.selected, self.labels, self.images):
            feedback_value = 1 if selected else 0
            self.manager.update_feedback(label, image, feedback_value)
        print("Feedback gesendet")

        # Clear Samples nach Feedback
        self.selected = [False] * self.max_images
        self.labels = []
        self.images = []
        #TODO: labels updaten

# Training GUI
class TrainingGUI(QWidget):
    def __init__(self, manager: TrainingManager):
        super().__init__()
        self.manager = manager
        self.human_feedback_widget = HumanFeedbackWidget(manager)
        self.init_ui()
        self.start_timer()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Steuerleiste
        control_layout = QHBoxLayout()
        self.toggle_button = QPushButton("Continue")
        self.toggle_button.setStyleSheet("background-color: green; color: white")
        self.toggle_button.clicked.connect(self.toggle_training)
        control_layout.addWidget(self.toggle_button)

        main_layout.addLayout(control_layout)

        # Tab Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Normal Tab
        tab_normal = QWidget()
        layout_normal = QHBoxLayout()

        self.figure = Figure(figsize = (5, 3))
        self.canvas = FigureCanvas(self.figure)
        layout_normal.addWidget(self.canvas)

        # Normal Tab: Left layout
        content_layout = QVBoxLayout()
        self.image_label = QLabel("Letztes Bild")
        self.image_label.setFixedSize(160, 160)
        content_layout.addWidget(self.image_label)

        # Normal Tab: Right
        # Filepath
        right_layout = QVBoxLayout()
        self.graph_path = QLineEdit()
        self.graph_path.setPlaceholderText("Pfad zum Speichern des Graphen")
        right_layout.addWidget(QLabel("Pfad Graph speichern:"))
        right_layout.addWidget(self.graph_path)

        # Normal Tab: Right
        # Save path button
        self.save_graph_button = QPushButton("Save Graph")
        self.save_graph_button.setStyleSheet("background-color: blue; color: white;")
        self.save_graph_button.clicked.connect(self.save_graph)
        right_layout.addWidget(self.save_graph_button)
        content_layout.addLayout(right_layout)
        layout_normal.addLayout(content_layout)
        tab_normal.setLayout(layout_normal)
        self.tabs.addTab(tab_normal, "Normal")

        # Human Feedback Tab
        tab_hf = QWidget()
        layout_hf = QVBoxLayout()
        layout_hf.addWidget(self.human_feedback_widget)
        tab_hf.setLayout(layout_hf)
        self.tabs.addTab(tab_hf, "Human Feedback")

        self.setLayout(main_layout)
        self.setWindowTitle("Training GUI")

    def start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(500)

    def toggle_training(self):
        stop_flag = self.manager.get_flag("stop")
        if stop_flag:
            self.manager.set_flag("stop", False)
            self.toggle_button.setText("Stop")
            self.toggle_button.setStyleSheet("background-color: red; color: white;")
        else:
            self.manager.set_flag("stop", True)
            self.toggle_button.setText("Continue")
            self.toggle_button.setStyleSheet("background-color: green; color: white;")

    def save_graph(self):
        path = self.graph_path.text().strip()
        if not path:
            print("Bitte Pfad angeben.")
            return
        self.figure.savefig(path)
        print(f"Graph gespeichert: {path}")

    def update_gui(self):
        info = self.manager.get_info()
        loss = info["loss"]
        last_img = info["last_image"]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(loss, label="Loss")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss pro Batch über Zeit")
        ax.legend()
        self.canvas.draw()

        if last_img is not None and isinstance(last_img, torch.Tensor): # Named Tensor verwenden?
            img = (last_img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img).scaled(160, 160)
            self.image_label.setPixmap(pixmap)

# Main
if __name__ == "__main__":
    manager = TrainingManager()
    t = threading.Thread(target=training_loop, args=(manager,), daemon=True)
    t.start()

    app = QApplication(sys.argv)
    gui = TrainingGUI(manager)
    gui.show()
    sys.exit(app.exec())