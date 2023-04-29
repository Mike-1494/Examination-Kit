import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    if not ret:
        break
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img)

    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_eye = results.pose_landmarks.landmark[1]
        right_eye = results.pose_landmarks.landmark[2]
        eye_center = ((left_eye.x + right_eye.x) / 2, (left_eye.y + right_eye.y) / 2)

        dx = eye_center[0] - nose.x
        dy = -eye_center[1] + nose.y
        print("dx: ", dx, " dy: ", dy)
        # Check if the student's hands are below their elbows and in front of their shoulders.
        if left_wrist.y < left_elbow.y and right_wrist.y < right_elbow.y and \
                left_wrist.x > left_shoulder.x and right_wrist.x < right_shoulder.x:
            print("Detected cheating")

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

    pose.close()

cap.release()
cv2.destroyAllWindows()
