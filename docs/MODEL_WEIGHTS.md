# Model Weights

Model weights are a crucial part of any deep learning project. They are the parameters that the model learns during training, and they are what allow the model to make accurate predictions.

## How Model Weights Are Used

In this project, the model weights are used to initialize the YOLO models for object detection. The weights are loaded into the models before the analysis begins, and they are what allow the models to detect the ball, players, and key points on the court.

## How Model Weights Were Created

The model weights used in this project were created by training the YOLO models on a large dataset of padel images. The dataset was annotated with bounding boxes for the ball, players, and key points, and the models were trained to predict these bounding boxes.

## How to Create New Model Weights

To create new model weights, you will need to train the YOLO models on your own dataset. The process is as follows:

### Requirements

- **A large dataset of images:** The more images you have, the better your model will be.
- **A tool for annotating images:** You will need a tool to draw bounding boxes around the objects you want to detect. LabelImg is a popular choice.
- **A powerful computer with a GPU:** Training a deep learning model is a computationally intensive task, and a GPU will significantly speed up the process.
- **The `ultralytics` library:** This library provides the tools you need to train YOLO models.

### Step 1: Create a Dataset

1. **Collect Images:** Collect a large number of images of padel games. The images should be high-quality and representative of the conditions you expect to see in the real world.
2. **Annotate Images:** Use a tool like LabelImg to draw bounding boxes around the objects you want to detect (e.g., ball, players, key points). The annotations should be saved in the YOLO format.

#### How to Annotate Images with LabelImg

1. **Install LabelImg:**
   ```bash
   pip install labelImg
   ```
2. **Run LabelImg:**
   ```bash
   labelImg
   ```
3. **Open a Directory:** Click on the "Open Dir" button and select the directory where your images are located.
4. **Create Bounding Boxes:** Click on the "Create RectBox" button and draw a bounding box around the object you want to detect.
5. **Enter a Label:** Enter a label for the bounding box (e.g., "ball", "player", "keypoint").
6. **Save the Annotation:** Click on the "Save" button to save the annotation. The annotation will be saved as a `.xml` file in the same directory as the image.
7. **Convert to YOLO Format:** You will need to convert the `.xml` files to the YOLO format. You can use a script like this one to do the conversion:
   ```python
   import xml.etree.ElementTree as ET
   import glob
   import os

   def xml_to_yolo(box, size):
       dw = 1./size[0]
       dh = 1./size[1]
       x = (box[0] + box[1])/2.0
       y = (box[2] + box[3])/2.0
       w = box[1] - box[0]
       h = box[3] - box[2]
       x = x*dw
       w = w*dw
       y = y*dh
       h = h*dh
       return (x,y,w,h)

   classes = ["ball", "player", "keypoint"]

   def convert_annotation(image_id):
       in_file = open('Annotations/%s.xml'%(image_id))
       out_file = open('labels/%s.txt'%(image_id), 'w')
       tree=ET.parse(in_file)
       root = tree.getroot()
       size = root.find('size')
       w = int(size.find('width').text)
       h = int(size.find('height').text)

       for obj in root.iter('object'):
           difficult = obj.find('difficult').text
           cls = obj.find('name').text
           if cls not in classes or int(difficult) == 1:
               continue
           cls_id = classes.index(cls)
           xmlbox = obj.find('bndbox')
           b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
           bb = xml_to_yolo(b, (w,h))
           out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

   wd = os.getcwd()

   for image_path in glob.glob(os.path.join(wd, "images", "*.jpg")):
       image_id = os.path.basename(image_path).split('.')[0]
       convert_annotation(image_id)
   ```

### Step 2: Train the Models

1. **Install the `ultralytics` library:**
   ```bash
   pip install ultralytics
   ```
2. **Train the Models:** Use the `yolo` command-line tool to train the models. You will need to specify the path to your dataset, the model you want to use, and the number of epochs to train for.
   ```bash
   yolo train data=path/to/your/dataset.yaml model=yolov8n.pt epochs=100
   ```
3. **Save the Weights:** Once the models are trained, the weights will be saved to a file in the `runs/train/exp` directory.

### Step 3: Update the Config

1. **Copy the Weights:** Copy the weights file to the `models` directory.
2. **Update the Config:** Open the `config.py` file and update the paths to your new weights file.
