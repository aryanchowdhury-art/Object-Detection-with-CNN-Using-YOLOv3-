# Object-Detection-with-CNN-Using-YOLOv3-
Object Detection with CNN (Using YOLOv3)" teaches the implementation of YOLOv3 for real-time object detection. Learn CNN fundamentals, YOLO architecture, and practical techniques for training models to identify objects in images and videos efficiently. Ideal for advancing skills in computer vision and deep learning.


import cv2
import numpy as np
import tensorflow as tf

# Load YOLOv3 model and weights
yolo_model = tf.keras.models.load_model('path/to/yolov3_model.h5')

# Load the COCO class labels
labels_path = 'path/to/coco.names'
with open(labels_path) as f:
    labels = f.read().strip().split("\n")

# Load the image
image = cv2.imread('path/to/image.jpg')
(H, W) = image.shape[:2]

# Prepare the image for YOLOv3 model
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
yolo_model.setInput(blob)

# Get the bounding boxes and probabilities
layer_names = yolo_model.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo_model.getUnconnectedOutLayers()]
layer_outputs = yolo_model.forward(output_layers)

boxes = []
confidences = []
class_ids = []

for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

# Ensure at least one detection exists
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
