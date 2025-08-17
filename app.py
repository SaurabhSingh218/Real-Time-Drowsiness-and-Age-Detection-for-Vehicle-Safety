from flask import Flask, render_template, Response, request
from facenet_pytorch import MTCNN
import torch
import torchvision
from torch import nn
from PIL import Image
from torchvision import transforms
import cv2
import time
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- FACE DETECTOR ---
mtcnn = MTCNN(keep_all=True, device=device)

# --- MODELS ---
def create_resnet50_model():
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    torch.manual_seed(42)
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, 1))
    return model

def create_resnet18_model(seed=42):
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    torch.manual_seed(seed)
    model.fc = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 2)
    )
    return model

age_model = create_resnet50_model()
age_model.load_state_dict(torch.load("best_model.pth", map_location=device))
age_model.to(device).eval()

eye_model = create_resnet18_model()
eye_model.load_state_dict(torch.load("eye_detector.pth", map_location=device))
eye_model.to(device).eval()

# --- TRANSFORMS ---
age_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
eye_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- HELPERS ---
def is_eye_closed(face_img: Image.Image) -> bool:
    """Returns True if CLOSED. NOTE: change '== 1' to '== 0' if your labels are reversed."""
    t = eye_transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(eye_model(t), dim=1).item()
    return pred == 1  # 1 == closed, 0 == open (adjust if your model differs)

# Global cache for webcam age delay
last_age = None
age_calculated = False
first_seen_time = None

def process_frame(frame, source_type="webcam"):
    global last_age, age_calculated, first_seen_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None and len(boxes) > 0:
        # Take the first detected face
        x1, y1, x2, y2 = map(int, boxes[0])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

        if x2 > x1 and y2 > y1:
            face_img = Image.fromarray(rgb[y1:y2, x1:x2])

            # Age: delay only for webcam
            if source_type == "webcam":
                if not age_calculated:
                    if first_seen_time is None:
                        first_seen_time = time.time()
                    elif time.time() - first_seen_time >= 3:
                        with torch.no_grad():
                            age_val = age_model(age_transform(face_img).unsqueeze(0).to(device)).item()
                        last_age = int(age_val)
                        age_calculated = True
                age = last_age
            else:
                with torch.no_grad():
                    age_val = age_model(age_transform(face_img).unsqueeze(0).to(device)).item()
                age = int(age_val)

            # Eye state
            closed = is_eye_closed(face_img)

            # Draw face box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Show age (you had age-1; keep it if it's your calibration)
            if age is not None:
                cv2.putText(frame, f"Age: {age-1}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Show Drowsy only if closed; show nothing if open
            if closed:
                cv2.putText(frame, "Drowsy", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Not Drowsy", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        # No faces: reset webcam delay state
        if source_type == "webcam":
            age_calculated = False
            last_age = None
            first_seen_time = None

    return frame

def gen_frames(source_type="webcam", file_path=None):
    # Open capture per source
    if source_type == "webcam":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    elif source_type in ["video", "image"]:
        cap = cv2.VideoCapture(file_path)
    else:
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # For image input, process only the first frame (single shot)
        if source_type == "image":
            frame = process_frame(frame, source_type)
            ret, buf = cv2.imencode('.jpg', frame)
            if not ret: break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n\r\n')
            break

        # Webcam / Video continuous
        frame = process_frame(frame, source_type)
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret: break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n\r\n')

    cap.release()

# --- UPLOADS ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    source_type = request.args.get("source", "webcam")
    file_path = request.args.get("file", None)
    return Response(gen_frames(source_type, file_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file", 400
    f = request.files['file']
    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    return {"file_path": path}

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file", 400
    f = request.files['file']
    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    return {"file_path": path}

if __name__ == "__main__":
    # Note: debug=True can spawn a reloader process; if you see double loads, use use_reloader=False
    app.run(debug=True, threaded=True)
