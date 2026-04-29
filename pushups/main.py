import cv2
import time
# from torch.xpu import device
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import numpy as np

def get_angle(a, b, c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0])
    ab = np.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle

model = YOLO("yolo26n-pose.pt")
# model.to("cuda")

counter = 0
stage = None
last_time = time.time()
timeout = 5.0
camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret,frame = camera.read()

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

    results = model(frame, verbose=False)

    if not results or not results[0].keypoints or not results[0].keypoints.xy.tolist()[0]:
        if time.time() - last_time > timeout:
            counter = 0
            stage = None

        cv2.putText(frame, f"Push-up: {counter}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0),3)
        cv2.imshow("Push-up", frame)
        continue

    result = results[0]
    keypoints = result.keypoints.xy.tolist()[0]
    last_time = time.time()

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0],
                   result.orig_shape, 5, True)
    annotated = annotator.result()

    left_shoulder = keypoints[5]
    left_elbow = keypoints[7]
    left_wrist = keypoints[9]
    # right_shoulder = keypoints[6]
    # right_elbow = keypoints[8]
    # right_wrist = keypoints[10]

    angle = get_angle(left_shoulder,left_elbow,left_wrist)

    if angle > 160:
        if stage == 'down':
            counter += 1
        stage = 'up'
    elif angle < 90:
        stage = 'down'

    cv2.putText(annotated, f"Push-up: {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Push-up", annotated)

camera.release()
cv2.destroyAllWindows()
