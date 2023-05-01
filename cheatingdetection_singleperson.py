import cv2
import numpy as np
import mediapipe as mp
from utils import CheatingDetection


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
conf_threshold = 0.7
nms_threshold = 0.4

cap = cv2.VideoCapture(0)
frame_count = 0 
while True:
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    boxes = []
    confidences = []
    class_ids = []
    frame_cut = None

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
                if classes[class_id] == 'person' and confidence >= 0.6:
                    frame_cut = frame[y:y+h, x:x+w] 

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    if frame_cut is not None:
        img = cv2.cvtColor(frame_cut, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        mp.solutions.drawing_utils.draw_landmarks(frame_cut, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results.pose_landmarks is not None:
            for landmark in results.pose_landmarks.landmark:
                if(landmark.visibility <= 0.2):
                    landmark.x = 100
                    landmark.y = 100
            nose = results.pose_landmarks.landmark[0]
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_eye = results.pose_landmarks.landmark[1]
            right_eye = results.pose_landmarks.landmark[2]
            left_index = results.pose_landmarks.landmark[24]
            right_index = results.pose_landmarks.landmark[25]
            detector = CheatingDetection()
            detector.load_data('landmarks.csv')
            detector.train_with_data()
            print("fc: ", frame_count) 
            if frame_count % 2 == 0 :
                cheat = detector.predict(nose.x, nose.y, left_shoulder.x, left_shoulder.y, right_shoulder.x, right_shoulder.y, left_elbow.x, left_elbow.y, right_elbow.x, right_elbow.y, left_wrist.x, left_wrist.y, right_wrist.x, right_wrist.y, left_index.x, left_index.y, right_index.x, right_index.y, left_eye.x, left_eye.y, left_eye.z, right_eye.x, right_eye.y, right_eye.z)
                #print(right_index)
                print(cheat)
            frame_count += 1 

            for i in indices:
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                label = classes[class_id]
                confidence = confidences[i]
                color = (0, 255, 0)
                
                if cheat == 1 :
                    text = "CHEATING"
                    color = (0,0,255)
                elif cheat == 0.5:   
                    text = "WARNING"
                    color = (0,255,255)
                else:
                    text = "NOT CHEATING"
                    color = (0,255,0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text , (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()