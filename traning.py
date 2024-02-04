from pathlib import Path 
from ultralytics import YOLO
import os

data_yaml= "/home/dcmdobot4/Documents/Training/Full_data_base_V2/data.yaml"
model=YOLO("yolov8s.pt")
model.train(
    imgsz=640,
    epochs=100,
    data=data_yaml,
    batch=8,
)
 



 
