import logging
import sys
import cv2
import os
from json import dumps, loads
from time import sleep
from kafka import KafkaProducer
logging.basicConfig(level=logging.INFO)


# Video Generation
def generate_video(image_folder):
    # image_folder = '/home/cthi/UIT/IE212/UAV-benchmark-M/M0101'
    folder_name = image_folder.split('/')[-1]
    video_path = 'videos/' + folder_name + '.mp4'

    print(video_path)

    # Listed images
    images = [img for img in os.listdir(image_folder)] 
    images = sorted(images)
    height = 540
    width = 1024

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # XVID codec
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    # Appending the images to the video one by one 
    for image in images:  
        video.write(cv2.imread(os.path.join(image_folder, image)))  

    video.release()

    return video_path

def publish_camera(topic, imgs):
    video_path = generate_video(imgs)

    # Create producer
    producer = KafkaProducer(bootstrap_servers='localhost:9092')
    video = cv2.VideoCapture(video_path)
    
    # Send frames of generated video
    try:
        print(f'Publishing {topic}')
        k = 1
        while(True):    
            __, frame = video.read()
            __, buffer = cv2.imencode('.jpg', frame)
            producer.send(topic, key=k.to_bytes(4, byteorder='big'), value=buffer.tobytes())
            k+=1
    except KeyboardInterrupt:
        producer.flush()
        print("\nExiting...")
        sys.exit(1)