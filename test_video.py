import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import distance
import math
import joblib
import time
import matplotlib.pyplot as plt
from playsound import playsound

class FaceMeshDetector:
    def __init__(self, model_path):
        self.knn_model = joblib.load(model_path)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            color=(128, 0, 128), thickness=2, circle_radius=1
        )
        self.inference_times = []
        self.state_change_counter = 0
        self.last_state_change_time = time.time()
        self.current_state = None
        self.alert_start_time = None  # New attribute for tracking alert time
        self.fps_start_time = time.time()
        self.fps_counter = 0
 

    def eye_aspect_ratio(self, eye):
        N1 = distance.euclidean(eye[1][0], eye[1][1])
        N2 = distance.euclidean(eye[2][0], eye[2][1])
        N3 = distance.euclidean(eye[3][0], eye[3][1])
        D = distance.euclidean(eye[0][0], eye[0][1])
        return (N1 + N2 + N3) / (3 * D)

    def mouth_aspect_ratio(self, mouth):
        N1 = distance.euclidean(mouth[1][0], mouth[1][1])
        N2 = distance.euclidean(mouth[2][0], mouth[2][1])
        N3 = distance.euclidean(mouth[3][0], mouth[3][1])
        D = distance.euclidean(mouth[0][0], mouth[0][1])
        return (N1 + N2 + N3) / (3 * D)

    def pupil_circularity(self, eye):
        perimeter = (
            distance.euclidean(eye[0][0], eye[1][0])
            + distance.euclidean(eye[1][0], eye[2][0])
            + distance.euclidean(eye[2][0], eye[3][0])
            + distance.euclidean(eye[3][0], eye[0][1])
        )
        area = math.pi * ((distance.euclidean(eye[1][0], eye[3][1]) * 0.5) ** 2)
        return (4 * math.pi * area) / (perimeter**2)

    def head_pose(self, face_2d, face_3d, img_w, img_h):
        focal_length = 1 * img_w
        cam_matrix = np.array(
            [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
        )
        distortion_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vec, translation_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, distortion_matrix
        )
        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles

    def process_frame(self, frame, frame_count, frame_rate):
        start_time = time.time()
        timestamp = round((frame_count / frame_rate), 2)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        text_color = (0, 0, 0)  # Default black text color
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        img_h, img_w, _ = frame.shape
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
                landmarks[:, 0] *= frame.shape[1]
                landmarks[:, 1] *= frame.shape[0]

                right_eye_indices = [[33, 133], [160, 144], [159, 145], [158, 153]]
                left_eye_indices = [[263, 362], [387, 373], [386, 374], [385, 380]]
                mouth_indices = [[61, 291], [39, 181], [0, 17], [269, 405]]
                head_pose_indices = [1, 33, 263, 61, 291, 199]

                right_eye = landmarks[right_eye_indices]
                left_eye = landmarks[left_eye_indices]
                mouth = landmarks[mouth_indices]

                right_ear = self.eye_aspect_ratio(right_eye)
                left_ear = self.eye_aspect_ratio(left_eye)
                ear = round(((right_ear + left_ear) / 2.0), 4)

                right_pupil_circularity = self.pupil_circularity(right_eye)
                left_pupil_circularity = self.pupil_circularity(left_eye)
                avg_pupil_circularity = round(
                    ((right_pupil_circularity + left_pupil_circularity) / 2.0), 4
                )

                mar = round((self.mouth_aspect_ratio(mouth)), 4)
                moe = round((mar / ear), 4)

                face_2d = []
                face_3d = []

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in head_pose_indices:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                angles = self.head_pose(face_2d, face_3d, img_w, img_h)

                x, y, z = (
                    angles[0] * (180 / math.pi),
                    angles[1] * (180 / math.pi),
                    angles[2] * (180 / math.pi),
                )
                if y < -15:
                    head_text = "Menoleh Kanan"
                elif y > 15:
                    head_text = "Menoleh Kiri"
                elif x < -10:
                    head_text = "Menunduk"
                elif x > 10:
                    head_text = "Menadah"
                else:
                    head_text = "Kedepan"

                input_data = np.array([ear, mar, avg_pupil_circularity, moe]).reshape(
                    1, -1
                )
                state = self.knn_model.predict(input_data)[0]
                current_time = time.time()
                if self.current_state is None:
                    self.current_state = state
                    self.last_state_change_time = current_time
                elif state != self.current_state:
                    if state == 1 and (current_time - self.last_state_change_time > 2):
                        self.state_change_counter += 1
                        self.current_state = state
                        self.last_state_change_time = current_time
                        if self.state_change_counter >= 5:
                            self.alert_start_time = current_time
                    elif state == 0:
                        self.current_state = 0
                else :
                    self.last_state_change_time = current_time

                
                if self.alert_start_time and (current_time - self.alert_start_time <= 5):
                    text_color = (0, 0, 255)  # Red
                    playsound("asset/alerts.mp3")
                elif self.alert_start_time and (current_time - self.alert_start_time > 5):
                    self.alert_start_time = None  # Reset alert
                    self.state_change_counter = 0  # Reset counter
                    
                for landmark in landmarks:
                    cv2.circle(
                        frame, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1
                    )

                cv2.putText(
                    frame,
                    f"Time: {timestamp:.2f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"MAR: {mar:.2f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"Pupil Circularity: {avg_pupil_circularity:.2f}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"MOE: {moe:.2f}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"Direction:  {head_text}",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}",
                    (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2,
                )

                
                cv2.putText(
                    frame,
                    f"State: {(state>0.5).astype(int)}",
                    (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"State Changes: {self.state_change_counter}",
                    (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    text_color,
                    2,
                )

        else:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
            )

        cv2.putText(
            frame,
            f"Current Time: {current_time}",
            (10, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )

        self.fps_counter += 1
        fps_end_time = time.time()
        fps = self.fps_counter / (fps_end_time - self.fps_start_time)

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )

        end_time = time.time()
        inference_time = end_time - start_time
        self.inference_times.append(inference_time)

        return frame

def main():
    model_path = (
        r"Landmarks/model/xgb_model.pkl"
    )
    detector = FaceMeshDetector(model_path)

    cap = cv2.VideoCapture(0)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (854, 480))
        frame = detector.process_frame(frame, frame_count, frame_rate)

        cv2.imshow("Driver Safety System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
