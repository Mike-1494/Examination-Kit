import cv2
import numpy as np

def detect_human(frame):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    conf_threshold = 0.5

    # Set the non-maximum suppression threshold
    nms_threshold = 0.4

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

    for output in outputs:
        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0: 

                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Check if the label is 'person' and the confidence is above the threshold
                if classes[class_id] == 'person' and confidence >= 0.6:
                    # Crop the frame to the bounding box of the person detection
                    frame_cut = frame[y:y+h, x:x+w] 
                    return frame_cut

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

    # Return None if no person detection above confidence threshold was found
    return None