import cv2
import numpy as np

# Load the YOLOv3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the COCO class labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set the minimum confidence threshold for detection
conf_threshold = 0.5

# Set the non-maximum suppression threshold
nms_threshold = 0.4

# Open the video stream
cap = cv2.VideoCapture(0)  # set the index to 0 for the default camera

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input layer of the network to the blob
    net.setInput(blob)

    # Get the output layers of the network
    output_layers = net.getUnconnectedOutLayersNames()

    # Run the forward pass through the network
    outputs = net.forward(output_layers)

    # Create lists to store the bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each output from the network
    for output in outputs:
        # Loop over each detection in the output
        for detection in output:
            # Get the class ID and confidence of the current detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections by confidence threshold
            if confidence > conf_threshold and class_id == 0:  # class_id 0 is for person in the COCO dataset
                # Scale the bounding box coordinates to the input image size
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                # Add the bounding box, confidence, and class ID to the lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Loop over the indices of the bounding boxes to draw them on the frame
    for i in indices:
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        label = classes[class_id]
        confidence = confidences[i]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = "{}: {:.4f}".format(label, confidence)
        cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
