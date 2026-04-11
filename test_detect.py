from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # bilerek hazır model
cap = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3)

    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("TEST", frame)
    if cv2.waitKey(1) == 27:
        break