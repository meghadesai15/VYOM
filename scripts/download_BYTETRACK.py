from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.track(
    source="data/raw/aggressive_gesture/aggresive_gesture (17).mp4",
    tracker="bytetrack.yaml",
    persist=True,
    save = True
)

if results[0].boxes.id is not None:
    track_ids = results[0].boxes.id.int().cpu().tolist()
    boxes = results[0].boxes.xyxy.cpu().tolist()

    print(track_ids)