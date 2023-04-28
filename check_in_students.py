import cv2
import numpy as np
from pyzbar.pyzbar import decode
import os
import face_recognition
from multiprocessing import Pool
import xepcho


cap2 = cv2.VideoCapture(1)
cap = cv2.VideoCapture(0)

# Initialize face recognition model
def load_known_faces(known_dir):
    known_encodings = []
    known_names = []

    for file in os.listdir(known_dir):
        img = face_recognition.load_image_file(os.path.join(known_dir, file))
        img_enc = face_recognition.face_encodings(img)[0]
        known_encodings.append(img_enc)
        known_names.append(file.split('.')[0])

    return known_encodings, known_names

seated = {}

while True:
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    
    barcodes = decode(frame2)
    valid_barcode = False

    for barcode in barcodes:
        barcode_data = barcode.data.decode('utf-8')
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame2, (x,y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame2, barcode_data, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)     

        if barcode_data.isdigit():  # Check if barcode is a valid number
            valid_barcode = True
            known_dir = f'D:\CODE\COMPUTER VISION\Vision Proj\known_dir\{barcode_data}'
            known_encodings, known_names = load_known_faces(known_dir)
            break

    if valid_barcode:
        # Convert the frame from BGR to RGB format (required for face_recognition library)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Compare faces to the known encodings
        results = []
        seat = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            seat_num = 0
            if True in matches:
                match_index = matches.index(True) 
                name = known_names[match_index]
                results.append(name)
                if(name not in seated):
                    seat_num = xepcho.seat_taker(num_seats= xepcho.num_seats)
                    seated.update({name:seat_num})
                    seat.append(seat_num)
                    
                else:
                    seat_num = seated[name]

        for (top, right, bottom, left), name in zip(face_locations, results):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
            cv2.putText(frame, "Cho ngoi cua ban la: " +str(seat_num), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame) 
    cv2.imshow("Cam2", frame2)

    if(cv2.waitKey(1) ==  ord('q')):
        break

cap.release()
cap2.release()
cv2.destroyAllWindows()