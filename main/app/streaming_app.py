from kafka import KafkaConsumer
from flask import Flask, Response, render_template, request, url_for
import numpy as np
from sys import path

path.append('./')
from _constants import *
from time import sleep


def get_video_stream(uavID):
    topic = uavID
    sleep(5)
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=[kafka_server],
        # auto_offset_reset = 'earliest'
    )
    for msg in consumer:
        try:
            yield (b"--frame\r\n" b"Content-Type: image/jpg\r\n\r\n" + msg.value + b'\r\n\r\n')
            sleep(0.1)
        except AttributeError:
            continue


app = Flask(__name__)


# Home
@app.route("/")
def home():
    return render_template('index.html')


# UAV 1
@app.route('/UAV1')
def UAV1():
    return Response(get_video_stream(topic_out_1), mimetype='multipart/x-mixed-replace; boundary=frame')


# UAV 2
@app.route("/UAV2")
def UAV2():
    return Response(get_video_stream(topic_out_2), mimetype='multipart/x-mixed-replace; boundary=frame')


# UAV 3
@app.route("/UAV3")
def UAV3():
    return Response(get_video_stream(topic_out_3), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Wait for the query writing data
    app.run(debug=True)
