# UAV-based Real-time Vehicles Detection



## I. Introduction

Within this project, we would like to deploy a pipeline to continuously detect vehicles (car, truck, bus) within video 
frames sent from simulated Unmanned Aerial Vehicles (UAVs). 

Each UAV, as producer, continuously captures and sends video frames to Kafka Streaming. As this concept requires to
process a huge amount of data and address real-time challenges, we utilize Spark Distributed Deep Learning for detecting
bounding boxes of vehicles and Spark Structured Streaming to read and write stream data. The end point of data is Kafka 
output sink, and we use consumers to load that stream data out.

## II. Dataset

We decided to use [UAVDT Benchmark](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5)
dataset for our simulation and training stage. The distribution of the dataset is shown below.

![](images/UAVDT_distribution.png)

## III. Model

We have chosen YOLO (You Only Look Once) as the primary solution for our approach due to the system's need for both real
-time constraints and high accuracy, with a higher priority on the former. Specifically, we are utilizing YOLOv5n for 
our detection tasks. Additionally, we have included other YOLO models, such as YOLOv8s and YOLOv8n, to enable a more 
comprehensive and accurate comparison.

![](images/prediction.jpg)

## IV. Pipeline Lineage

![](images/System.png)
