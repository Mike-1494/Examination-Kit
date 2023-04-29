import os
import csv
import cv2
import mediapipe as mp
import numpy as np
import math
from boundbox import detect_human

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True ,min_detection_confidence=0.5, min_tracking_confidence=0.5)

data_path = 'D:\CODE\COMPUTER VISION\Vision Proj\DATASET'
csv_file = 'landmarks.csv'
labels = {'cheating': 1, 'not_cheating': 0}

with open(csv_file, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    header = ['file_name', 'label', 'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y', 'left_elbow_x', 'left_elbow_y', 'right_elbow_x', 'right_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x', 'right_wrist_y', 'left_index_finger_x', 'left_index_finger_y', 'right_index_finger_x', 'right_index_finger_y', 'dx', 'dy']
    landmark_names = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_index', 'right_index']

    csv_writer.writerow(header)

for label in labels:
    label_path = os.path.join(data_path, label)
    for filename in os.listdir(label_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):

            img = cv2.imread(os.path.join(label_path, filename))
            roi = detect_human(img)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            #print(roi)
            results = pose.process(roi_rgb)
            if results.pose_landmarks is not None:
                landmarks = [filename, labels[label]]

                # Extract landmark coordinates
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    landmark_name = mp_pose.PoseLandmark(i).name.lower()
                    if landmark.visibility >= 0.6 and landmark_name in landmark_names:
                        landmarks.append(round(landmark.x, 5))
                        landmarks.append(round(landmark.y, 5))
                #Angels
                nose = results.pose_landmarks.landmark[0]
                left_eye = results.pose_landmarks.landmark[1]
                right_eye = results.pose_landmarks.landmark[2]
                left_shoulder = results.pose_landmarks.landmark[11] 
                right_shoulder = results.pose_landmarks.landmark[12] 
                left_elbow = results.pose_landmarks.landmark[13] 
                right_elbow = results.pose_landmarks.landmark[14]
                left_wrist = results.pose_landmarks.landmark[15]
                right_wrist = results.pose_landmarks.landmark[16]

                eye_center = ((left_eye.x + right_eye.x) / 2, (left_eye.y + right_eye.y) / 2)

                dx = eye_center[0] - nose.x
                dy = -eye_center[1] + nose.y
                landmarks.append(round(dx, 5))
                landmarks.append(round(dy, 5))


                with open(csv_file, mode='a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(landmarks)
            
            cv2.imshow("ROI: "+filename, roi)
            cv2.waitKey(5000)

            cv2.destroyAllWindows()