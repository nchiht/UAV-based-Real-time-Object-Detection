import findspark
findspark.init()

# Standard library imports
from time import sleep
# from _constants import *

# Third-party imports
import numpy as np
import cv2
from ultralytics import YOLO

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql.functions import col, udf
from pyspark.sql.streaming import DataStreamReader
from pyspark.sql.types import BinaryType


# Kafka's Information
kafka_server = 'localhost:9092'

topic_in_1 = 'drone_1'
topic_in_2 = 'drone_2'
topic_in_3 = 'drone_3'

topic_out_1 = 'UAV_1'
topic_out_2 = 'UAV_2'
topic_out_3 = 'UAV_3'

# Spark Configuration
scala_version = '2.12'
spark_version = '3.5.1'

packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
    'org.apache.kafka:kafka-clients:3.7.0'
]
spark = SparkSession \
.builder \
.appName("UAV Detection") \
.master("local") \
.config("spark.executor.memory", "16g") \
.config("spark.driver.memory", "16g") \
.config("spark.python.worker.reuse", "true") \
.config("spark.sql.execution.arrow.pyspark.enabled", "true") \
.config("spark.sql.execution.arrow.maxRecordsPerBatch", "5") \
.config("spark.scheduler.mode", "FAIR")  \
.config("spark.jars.packages", ",".join(packages)) \
.getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

conf=SparkConf()

print(spark)

yolo = YOLO("params/pt_yolov5n.pt")
# yolo = YOLO("../params/pt_yolov8s.pt") 
# yolo = YOLO("../params/pt_yolov8n.pt")  

# Broadcast model
sc = SparkContext.getOrCreate()
broadcast_model = sc.broadcast(yolo)

model = broadcast_model.value

def load_and_preprocess_frames(frame_bytes):
    frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    return frame

def predict(frame_bytes):
    image = load_and_preprocess_frames(frame_bytes)
    prediction = model.predict(image)
    ret, buffer = cv2.imencode('.jpg', prediction[0].plot())
    print(buffer)
    return buffer.tobytes()

predict_udf = udf(predict, BinaryType())

def queryWriter(topic_in, topic_out):
    streamRawDF = spark.readStream.format("kafka").option("kafka.bootstrap.servers", kafka_server).option("subscribe", topic_in).option("startingOffsets","latest").load()

    streamRawDF = streamRawDF.withColumn('value1', col('value'))
    streamRawDF = streamRawDF.drop('value')
    streamRawDF = streamRawDF.withColumn('value', predict_udf('value1'))

    query = streamRawDF.writeStream \
    .format("kafka") \
    .trigger(processingTime="2 seconds") \
    .option("kafka.bootstrap.servers", kafka_server) \
    .option('topic', topic_out) \
    .option("checkpointLocation", f'checkpoint/' + topic_out) \
    .start()

    return query

query_1 = queryWriter(topic_in_1, topic_out_1)
query_2 = queryWriter(topic_in_2, topic_out_2)
query_3 = queryWriter(topic_in_3, topic_out_3)

query_3.awaitTermination()