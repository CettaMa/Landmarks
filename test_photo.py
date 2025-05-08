import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import math


# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    N1 = distance.euclidean(eye[1][0], eye[1][1])
    N2 = distance.euclidean(eye[2][0], eye[2][1])
    N3 = distance.euclidean(eye[3][0], eye[3][1])
    D = distance.euclidean(eye[0][0], eye[0][1])
    return (N1 + N2 + N3) / (3 * D)

# Function to calculate mouth aspect ratio
def mouth_aspect_ratio(mouth):
    N1 = distance.euclidean(mouth[1][0], mouth[1][1])
    N2 = distance.euclidean(mouth[2][0], mouth[2][1])
    N3 = distance.euclidean(mouth[3][0], mouth[3][1])
    D = distance.euclidean(mouth[0][0], mouth[0][1])
    return (N1 + N2 + N3) / (3 * D)

# Function to calculate pupil circularity
def pupil_circularity(eye):
    perimeter = distance.euclidean(eye[0][0], eye[1][0]) + \
                distance.euclidean(eye[1][0], eye[2][0]) + \
                distance.euclidean(eye[2][0], eye[3][0]) + \
                distance.euclidean(eye[3][0], eye[0][1]) + \
                distance.euclidean(eye[0][1], eye[3][1]) + \
                distance.euclidean(eye[3][1], eye[2][1]) + \
                distance.euclidean(eye[2][1], eye[1][1]) + \
                distance.euclidean(eye[1][1], eye[0][0])
    area = math.pi * ((distance.euclidean(eye[1][0], eye[3][1]) * 0.5) ** 2)
    return (4 * math.pi * area) / (perimeter ** 2)

# Load the image
image_path = r"Landmarks\serious-young-driver-behind-steering-wheel_jpg.png"
frame = cv2.imread(image_path)

# Resize frame to 480p
frame = cv2.resize(frame, (854, 480))
# Convert the frame to RGB
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# Process the frame and get the face landmarks
results = face_mesh.process(rgb_frame)

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
        landmarks[:, 0] *= frame.shape[1]
        landmarks[:, 1] *= frame.shape[0]

        # Define landmark indices for right eye, left eye, and mouth
        right_eye_indices = [[33, 133], [160, 144], [159, 145], [158, 153]]
        left_eye_indices = [[263, 362], [387, 373], [386, 374], [385, 380]]
        mouth_indices = [[61, 291], [39, 181], [0, 17], [269, 405]]

        # Extract the landmarks for right eye, left eye, and mouth
        right_eye = landmarks[right_eye_indices]
        left_eye = landmarks[left_eye_indices]
        mouth = landmarks[mouth_indices]

        right_ear = eye_aspect_ratio(right_eye)
        left_ear = eye_aspect_ratio(left_eye)
        ear = round(((right_ear + left_ear) / 2.0), 4)

        right_pupil_circularity = pupil_circularity(right_eye)
        left_pupil_circularity = pupil_circularity(left_eye)
        avg_pupil_circularity = round(((right_pupil_circularity + left_pupil_circularity) / 2.0), 4)

        mar = round((mouth_aspect_ratio(mouth)), 4)
        moe = round((mar / ear), 4)

        # Draw face landmarks
        for landmark in landmarks:
            cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1)

        # Overlay text on the frame
        cv2.putText(frame, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'MAR: {mar:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Pupil Circularity: {avg_pupil_circularity:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'MOE: {moe:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Predict the state
        state = 1 if moe > 3 else 0
        cv2.putText(frame, f'State: {"Drowsy" if state ==1 else "Awake"}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
else:
    cv2.putText(frame, 'No face detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Display the frame
cv2.imshow('Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
