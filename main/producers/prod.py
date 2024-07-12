import logging
import sys
import cv2
import os
from json import dumps, loads
from time import sleep
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO)
from sys import path

path.append('./')
from _constants import *


# Video Generation
def generate_video(image_folder):
    folder_name = image_folder.split('/')[-1]
    video_path = 'data/videos/' + folder_name + '.mp4'

    print(video_path)

    # Listed images
    images = [img for img in os.listdir(image_folder)]
    images = sorted(images)
    height = 540
    width = 1024

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # XVID codec
    video = cv2.VideoWriter(video_path, fourcc, 20, (width, height))

    # Appending the images to the video one by one 
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

    return video_path


def publish_camera(topic, imgs):
    video_path = generate_video(imgs)

    # Create producer
    producer = KafkaProducer(bootstrap_servers=kafka_server)
    video = cv2.VideoCapture(video_path)

    # Send frames of generated video
    print(f'Publishing {topic}')
    try:
        k = 1
        while (True):
            __, frame = video.read()
            __, buffer = cv2.imencode('.jpg', frame)
            producer.send(topic, key=k.to_bytes(4, byteorder='big'), value=buffer.tobytes())
            k += 1
    except KeyboardInterrupt:
        producer.flush()
        print("\nExiting...")
        sys.exit(1)
