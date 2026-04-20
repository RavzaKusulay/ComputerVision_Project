from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(

    data="data.yaml",

    epochs=20,

    imgsz=640,

    batch=8,

    patience=10

)