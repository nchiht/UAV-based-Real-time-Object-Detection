from kafka import KafkaConsumer
from flask import Flask, Response, render_template, request, url_for
import numpy as np
# from _constants import *
from time import sleep


def get_video_stream(uavID):
    topic = uavID
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers = ['localhost:9092'],
    )
    sleep(5)
    for msg in consumer:
        try:
            yield (b"--frame\r\n" b"Content-Type: image/jpg\r\n\r\n" + msg.value + b'\r\n\r\n')
            sleep(0.1)
        except AttributeError:
            continue

app = Flask(__name__)

#Home
@app.route("/")
def home():
    return Response(get_video_stream('UAV_1'), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True, port=5001)