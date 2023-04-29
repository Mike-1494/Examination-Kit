import cv2  
import numpy as np
from pyzbar.pyzbar import decode
import os
import face_recognition
from multiprocessing import Pool

cap = cv2.VideoCapture(1)

# Initialize face recognition model and known encodings
known_encodings = []
known_names = []
known_dir = 'C:/CODE/AI/Vision Proj/known_dir'

for file in os.listdir(known_dir):
    #print(file)
    img = face_recognition.load_image_file(os.path.join(known_dir, file))
    img_enc = face_recognition.face_encodings(img)[0]
    known_encodings.append(img_enc)
    known_names.append(file.split('.')[0])

while True: 
    ret, frame = cap.read() 

    # Convert the frame from BGR to RGB format (required for face_recognition library)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Use numpy slicing instead of cv2 ROI selection
    roi = frame[150:300, 200:480]

    # Use multiprocessing to decode barcodes in parallel
    barcodes = decode(frame)
    result = []
    for barcode in barcodes:
        # Extract barcode data and draw bounding box around it
        barcode_data = barcode.data.decode('utf-8')
        print(barcode_data) 
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x + 200, y + 150), (x + w + 200, y + h + 150), (0, 255, 0), 2)
        #cv2.putText(frame, barcode_data, (x + 200, y + 150 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)     

        file_to_open = str(barcode_data) +".jpg"     
        img = face_recognition.load_image_file(os.path.join(known_dir, file_to_open))
        img_enc = face_recognition.face_encodings(img)[0]
        known_encodings.append(img_enc)
        known_names.append(file.split('.')[0])

        matches = face_recognition.compare_faces(known_encodings, face_encodings, tolerance=0.7)
        if matches[0]:
            name = known_names[known_encodings.index(img_enc)]
            result.append(name)


    # Draw boxes and labels for the faces
    for (top, right, bottom, left), name in zip(face_locations, result):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    cv2.imshow('Frame', frame) 
    if(cv2.waitKey(1) ==  ord('q')):
        break
    

cap.release() 
cv2.destroyAllWindows() 
