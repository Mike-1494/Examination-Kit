import cv2
from boundbox import detect_human

# Create a VideoCapture object to read from the webcam
cap = cv2.VideoCapture(0)

# Set the frame size to 600x600
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Detect humans in the frame using the detect_human function
    human_frame = detect_human(frame)

    # If a human is detected, draw a bounding box around them
    if human_frame is not None:
        cv2.imshow('Human detection', human_frame)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
