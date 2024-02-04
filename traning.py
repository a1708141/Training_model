from pathlib import Path 
from ultralytics import YOLO
import os
#EDIT HERE
data_yaml= "SOURCE TO DATA.YAML FILE"
model=YOLO("yolov8s.pt")
model.train(
    imgsz=640,
    epochs=100,
    data=data_yaml,
    batch=8,
)
 



 
