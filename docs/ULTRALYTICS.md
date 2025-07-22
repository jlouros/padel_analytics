# Ultralytics

Ultralytics is a company that provides tools and services for computer vision. Their most popular product is the YOLO (You Only Look Once) series of object detection models.

## YOLO Models

YOLO is a state-of-the-art object detection model that is known for its speed and accuracy. It is a one-stage detector, which means that it predicts bounding boxes and class probabilities in a single pass. This makes it much faster than two-stage detectors like Faster R-CNN.

### YOLOv8

YOLOv8 is the latest version of the YOLO model. It is faster and more accurate than previous versions, and it is also easier to use. YOLOv8 is available in a variety of sizes, from the small `yolov8n.pt` to the large `yolov8x.pt`. The larger models are more accurate, but they are also slower.

## Ultralytics Library

The `ultralytics` library provides a simple and intuitive API for working with YOLO models. It allows you to train, validate, and deploy YOLO models with just a few lines of code.

### Installation

To install the `ultralytics` library, you can use pip:
```bash
pip install ultralytics
```

### Usage

Here is an example of how to use the `ultralytics` library to train a YOLOv8 model:
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from scratch
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Use the model
model.train(data='coco128.yaml', epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
path = model.export(format='onnx')  # export the model to ONNX format
```

## Licensing

The `ultralytics` library is licensed under the AGPL-3.0 license. This means that it is free to use for personal and academic purposes. However, if you want to use it for commercial purposes, you will need to purchase a commercial license.
