import os
import cv2
from ultralytics import YOLO

# Eğer kendi modelin yoksa yolov8n.pt kullan
model = YOLO("yolov8n.pt")

image_folder = "C:/Users/user/Desktop/ComputerVision_Project/dataset/images"
label_folder = "C:/Users/user/Desktop/ComputerVision_Project/dataset/labels"

os.makedirs(label_folder, exist_ok=True)

for file in os.listdir(image_folder):

    if file.endswith(".jpg"):

        path = os.path.join(image_folder, file)

        img = cv2.imread(path)

        results = model(img, conf=0.25)

        h, w, _ = img.shape

        txt_name = file.replace(".jpg", ".txt")
        txt_path = os.path.join(label_folder, txt_name)

        with open(txt_path, "w") as f:

            if results[0].boxes is not None:

                for box in results[0].boxes:

                    cls = int(box.cls[0])

                    # sadece insan
                    if cls != 0:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0]

                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h

                    f.write(
                        f"0 {x_center} {y_center} {width} {height}\n"
                    )

print("Etiketleme tamamlandı.")