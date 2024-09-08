import os

from ultralytics import YOLOv10


config_path = './config.yaml'

# Load a model
model = YOLOv10.from_pretrained("jameslahm/yolov10n")  # load pre trained model

# Use the model
model.train(data=config_path, epochs=200, batch=32)  # train the model
