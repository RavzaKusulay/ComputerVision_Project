import cv2
import os

video_path = "main_test.mp4"

os.makedirs("images", exist_ok=True)

video = cv2.VideoCapture(video_path)

i = 0
count = 0

while True:

    ret, frame = video.read()

    if not ret:
        break

    # Her 15 karede bir görüntü al
    if i % 15 == 0:

        filename = f"images/frame_{count}.jpg"

        cv2.imwrite(filename, frame)

        count += 1

    i += 1

video.release()

print("Toplam görüntü:", count)