import sys
import cv2
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import threading
from queue import Queue
import time
import math
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


def create_resnet50_model(seed=42):
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    torch.manual_seed(seed)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(2048, 1)
    )
    return model, transforms


def create_resnet18_model(classes=2, seed=42):
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    torch.manual_seed(seed)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    return model, transforms


def box_distance(b1, b2):
    cx1 = (b1[0] + b1[2]) / 2
    cy1 = (b1[1] + b1[3]) / 2
    cx2 = (b2[0] + b2[2]) / 2
    cy2 = (b2[1] + b2[3]) / 2
    return math.hypot(cx1 - cx2, cy1 - cy2)


class FrameProcessor(threading.Thread):
    def __init__(self, input_queue, output_queue, face_detector, age_model, eye_model, transform, device):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.face_detector = face_detector
        self.age_model = age_model
        self.eye_model = eye_model
        self.transform = transform
        self.device = device
        self.running = True
        self.face_cache = {}
        self.face_id_counter = 0

    def run(self):
        while self.running:
            if not self.input_queue.empty():
                frame = self.input_queue.get()
                if frame is None:
                    break
                processed_frame = self.process_frame(frame)
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except:
                        break
                self.output_queue.put(processed_frame)
            else:
                time.sleep(0.001)

    def stop(self):
        self.running = False

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = self.face_detector.detect(rgb_frame)
        current_time = time.time()
        sleeping_ages = []

        if boxes is not None and probs is not None:
            for box, prob in zip(boxes, probs):
                if prob < 0.9:
                    continue

                x1, y1, x2, y2 = box
                new_box = [x1, y1, x2, y2]

                matched_id = None
                for fid, data in self.face_cache.items():
                    dist = box_distance(new_box, data["box"])
                    if dist < 50:
                        matched_id = fid
                        break

                if matched_id is None:
                    matched_id = self.face_id_counter
                    self.face_id_counter += 1
                    face_crop = rgb_frame[int(y1):int(y2), int(x1):int(x2)]
                    if face_crop.size > 0:
                        face_img = Image.fromarray(face_crop)
                        face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            age = self.age_model(face_tensor).item()
                    else:
                        age = -1
                    self.face_cache[matched_id] = {"box": new_box, "age": age}
                else:
                    age = self.face_cache[matched_id]["age"]

                self.face_cache[matched_id]["box"] = new_box
                self.face_cache[matched_id]["updated"] = current_time

                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                eye_x = x + w // 4
                eye_y = y + h // 3
                eye_size = max(20, h // 6)

                eye_region = rgb_frame[
                    max(0, eye_y - eye_size // 2):min(rgb_frame.shape[0], eye_y + eye_size // 2),
                    max(0, eye_x - eye_size // 2):min(rgb_frame.shape[1], eye_x + eye_size // 2)
                ]

                sleeping = False
                if eye_region.size > 0:
                    eye_img = Image.fromarray(eye_region)
                    eye_tensor = self.transform(eye_img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        eye_output = self.eye_model(eye_tensor)
                        eye_pred = torch.argmax(eye_output, dim=1).item()
                        if eye_pred == 0:
                            sleeping = True

                if sleeping:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Sleeping", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    sleeping_ages.append(int(age))
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(frame, f"Age: {int(age)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        stale_ids = []
        for fid, data in self.face_cache.items():
            if data.get('updated', 0) < current_time - 2:
                stale_ids.append(fid)
        for fid in stale_ids:
            del self.face_cache[fid]

        return frame, sleeping_ages


class DrowsinessDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detection System")
        self.setGeometry(100, 100, 1200, 800)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.face_detector = MTCNN(keep_all=True, device=self.device, thresholds=[0.8, 0.9, 0.9], min_face_size=40)

        self.age_model, _ = create_resnet50_model()
        self.age_model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
        self.age_model.to(self.device)
        self.age_model.eval()

        self.eye_model, _ = create_resnet18_model()
        self.eye_model.load_state_dict(torch.load("eye_detector.pth", map_location=self.device))
        self.eye_model.to(self.device)
        self.eye_model.eval()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.input_queue = Queue(maxsize=2)
        self.output_queue = Queue(maxsize=2)

        self.processor_thread = FrameProcessor(
            self.input_queue, self.output_queue,
            self.face_detector, self.age_model, self.eye_model,
            self.transform, self.device
        )
        self.processor_thread.start()

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(800, 600)
        layout.addWidget(self.display_label)

        button_layout = QHBoxLayout()

        self.webcam_button = QPushButton("Open Webcam")
        self.webcam_button.clicked.connect(self.start_webcam)

        self.image_button = QPushButton("Open Image")
        self.image_button.clicked.connect(self.open_image)

        self.video_button = QPushButton("Open Video")
        self.video_button.clicked.connect(self.open_video)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.webcam_button)
        button_layout.addWidget(self.image_button)
        button_layout.addWidget(self.video_button)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))

            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except:
                    break
            self.input_queue.put(frame)

            if not self.output_queue.empty():
                out = self.output_queue.get()
                if isinstance(out, tuple):
                    frame, sleeping_ages = out
                else:
                    frame, sleeping_ages = out, []

                self.display_frame(frame)

                if sleeping_ages:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Drowsiness Alert")
                    msg.setText(f"Sleeping: {len(sleeping_ages)}\nAges: {', '.join(map(str, sleeping_ages))}")
                    msg.exec_()
        else:
            self.stop()

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(scaled_pixmap)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(30)
            self.update_button_states(True)

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            if self.cap.isOpened():
                self.timer.start(30)
                self.update_button_states(True)

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            frame = cv2.imread(file_name)
            if frame is not None:
                while not self.input_queue.empty():
                    try:
                        self.input_queue.get_nowait()
                    except:
                        break
                self.input_queue.put(frame)
                if not self.output_queue.empty():
                    out = self.output_queue.get()
                    if isinstance(out, tuple):
                        frame, sleeping_ages = out
                    else:
                        frame, sleeping_ages = out, []
                    self.display_frame(frame)
                    if sleeping_ages:
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Warning)
                        msg.setWindowTitle("Drowsiness Alert")
                        msg.setText(f"Sleeping: {len(sleeping_ages)}\nAges: {', '.join(map(str, sleeping_ages))}")
                        msg.exec_()

    def stop(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        while not self.input_queue.empty():
            self.input_queue.get()
        while not self.output_queue.empty():
            self.output_queue.get()
        self.update_button_states(False)

    def update_button_states(self, is_running):
        self.webcam_button.setEnabled(not is_running)
        self.video_button.setEnabled(not is_running)
        self.image_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)

    def closeEvent(self, event):
        self.stop()
        self.processor_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DrowsinessDetectionGUI()
    window.show()
    sys.exit(app.exec_())
