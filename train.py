from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("/home/cthi/UIT/IE212/params/yolov8s/pt_yolov8s.pt")  # load a pretrained model (recommended for training)

results = model('/home/cthi/UIT/IE212/dataset/test/images/M0101_000004.jpg')

image = cv2.imread('/home/cthi/UIT/IE212/dataset/test/images/M0101_000004.jpg', cv2.IMREAD_COLOR)

ret, buffer = cv2.imencode('.jpg', image)

print(buffer.tobytes())