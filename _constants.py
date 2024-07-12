import os

kafka_server = 'localhost:9092'

set_producer_1 = 'data/UAV-benchmark-M/M0101'
set_producer_2 = 'data/UAV-benchmark-M/M0201'

# Folder for weather generation
set_weather = 'data/UAV-benchmark-M/M0101'
folder_weather = 'data/UAV-benchmark-M/M0101wth'

# Spark Configuration
scala_version = '2.12'
spark_version = '3.5.1'
kafka_version = '3.7.0'

# App spark
app_name = 'UAV Detection'

# Directory YOLO model
yolov5n = "params/pt_yolov5n.pt"
yolov8s = "params/pt_yolov8s.pt"
yolov8n = "params/pt_yolov8n.pt"

# Kafka's Topics
topic_in_1 = 'drone_1'
topic_in_2 = 'drone_2'
topic_in_3 = 'drone_3'

topic_out_1 = 'UAV_1'
topic_out_2 = 'UAV_2'
topic_out_3 = 'UAV_3'

checkpoint_1 = 'uav1'
checkpoint_2 = 'uav2'
checkpoint_3 = 'uav3'

# Weather_model
pretrained_model_rainy = "Weather_Effect_Generator/Cyclic_GAN/clear2rainy.pth"
pretrained_model_snowy = "Weather_Effect_Generator/Cyclic_GAN/clear2snowy.pth"
