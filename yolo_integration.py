# YOLO-based Parking/People Detection from Satellite Images
import torch
from PIL import Image
from ultralytics import YOLO

def load_yolo_model():
    model = YOLO("yolov8n.pt")  # Make sure this is available or downloaded
    return model

def extract_cv_features(image, model):
    results = model(image)
    boxes = results[0].boxes
    labels = results[0].names

    people_count = 0
    car_count = 0
    for box in boxes:
        cls = int(box.cls)
        label = labels[cls].lower()
        if "person" in label:
            people_count += 1
        elif "car" in label or "vehicle" in label or "truck" in label:
            car_count += 1

    percent_lot_filled = min(round(car_count / 20, 2), 1.0)  # estimate max 20 spaces
    return percent_lot_filled, people_count
