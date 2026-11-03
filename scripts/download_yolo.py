# download_yolo.py — run once
from ultralytics import YOLO
model = YOLO("yolov8n.pt")   # downloads automatically on first run
model.export(format="ncnn")   # export for RPi4 — saves as yolov8n_ncnn_model/