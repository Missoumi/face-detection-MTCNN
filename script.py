import mtcnn
from matplotlib import pyplot as plt
import cv2
from matplotlib.patches import Circle

detector = mtcnn.MTCNN()


def detect(frame):
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        for key, value in face['keypoints'].items():
            cv2.circle(frame, value, 2, (0, 255, 0))
    return frame


video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    canvas = detect(frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


