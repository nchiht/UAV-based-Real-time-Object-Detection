# Standard library imports
import datetime
from time import sleep
import struct

# Third-party imports
import numpy as np
import pandas as pd
import cv2
from flask import Flask, Response, render_template
from IPython.display import display, clear_output
from kafka import KafkaConsumer
from ultralytics import YOLO
import matplotlib.pyplot as plt
import io

# PySpark imports
import pyspark
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql.functions import col, udf
from pyspark.sql.streaming import DataStreamReader
from pyspark.sql.types import ArrayType, FloatType, BinaryType


scala_version = '2.12'
spark_version = '3.5.1'
packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
    'org.apache.kafka:kafka-clients:3.7.0'
]
spark = SparkSession \
.builder \
.appName("BigSale") \
.master("local") \
.config("spark.executor.memory", "18g") \
.config("spark.driver.memory", "8g") \
.config("spark.python.worker.reuse", "true") \
.config("spark.sql.execution.arrow.pyspark.enabled", "true") \
.config("spark.sql.execution.arrow.maxRecordsPerBatch", "16") \
.config("spark.jars.packages", ",".join(packages)) \
.getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

conf=SparkConf()

print(spark)

# # # Load a model
model = YOLO("/home/cthi/UIT/IE212/params/pt_yolov5n.pt")  # pretrained YOLOv8n model

# consumer = KafkaConsumer(topic, bootstrap_servers=['localhost:9092'])
topic_name = 'demo_drone'
kafka_server = 'localhost:9092'
streamRawDf = spark.readStream.format("kafka").option("kafka.bootstrap.servers", kafka_server).option("subscribe", topic_name).option("startingOffsets","latest").load()

# df = streamRawDf.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
stream_writer = (streamRawDf.writeStream.queryName('Item').trigger(processingTime="2 second").outputMode("append").format("memory"))

query = stream_writer.start()

# Broadcast model
sc = SparkContext.getOrCreate()
broadcast_model = sc.broadcast(model)

def load_and_preprocess_frames(frame_bytes):
    frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    return frame

# Define UDF to predict
def predict(frame_bytes):
    model = broadcast_model.value
    image = load_and_preprocess_frames(frame_bytes)
    prediction = model.predict(image)
    ret, buffer = cv2.imencode('.jpg', prediction[0].plot())
    print(buffer)
    return buffer.tobytes()

predict_udf = udf(predict, BinaryType())

app = Flask(__name__)

@app.route('/')
def video_feed():
    sleep(10)
    return Response( get_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def bytes_to_int(byte_array):
    return int.from_bytes(byte_array, byteorder='big')

def get_video_stream():

    index = 1
    # Read the data from the memory sink
    while True:
        try:
            byte_array_key = struct.pack('>I', index)
            hex_key = ''.join(f'{byte:02X}' for byte in byte_array_key)
            query = f"SELECT key, value FROM Item WHERE key = X'{hex_key}'"
            row = spark.sql(query)
            row = row.withColumn('prediction', predict_udf(row['value']))
            value = (row.select('prediction').first()).prediction

            yield (b"--frame\r\n" b"Content-Type: image/jpg\r\n\r\n" + value + b'\r\n\r\n')
            # # Increase index
            index += 1

        except KeyboardInterrupt as e:
            query.stop()
            return e
        except ValueError or AttributeError:
            continue

if __name__ == '__main__':
    app.run(debug=True)