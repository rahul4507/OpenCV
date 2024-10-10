import cv2 as cv
import numpy as np

# Load the YOLO model (You need the config and weights files)
net = cv.dnn.readNet('../Resources/Yolo3/yolov3.weights', '../Resources/Yolo3/yolov3.cfg')

# Load the COCO class labels (YOLO is trained on this dataset)
with open('../Resources/Yolo3/coco.names', 'r') as f:
    classes = f.read().splitlines()

# Load the input image
img = cv.imread('../Resources/Photos/park.jpg')
height, width, _ = img.shape

# Convert the image to blob format for YOLO
blob = cv.dnn.blobFromImage(img, scalefactor=1/255, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)

# Get the output layer names for YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
layer_outputs = net.forward(output_layers)

# Variables to store detected objects
boxes = []
confidences = []
class_ids = []

# Loop through the layer outputs
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]  # Get confidence scores for all classes
        class_id = np.argmax(scores)  # Get the highest scoring class id
        confidence = scores[class_id]

        if confidence > 0.3:  # Lowered confidence threshold for better detection
            # Get the object's bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the top-left corner of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Save the detection
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Perform non-maxima suppression to avoid multiple boxes for the same object
indices = cv.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# Draw the bounding boxes for detected objects
if len(indices) > 0:
    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))

        # Print the detected label and confidence
        print(f'Detected label: {label}, Confidence: {confidence}')

        # Draw bounding box and label
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(img, f'{label} {confidence}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show the image
cv.imshow('Detected Objects', img)
cv.waitKey(0)
cv.destroyAllWindows()
