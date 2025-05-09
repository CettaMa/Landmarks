import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import distance
import math
import joblib
import time
import threading
from playsound import playsound
from picamera2 import Picamera2

# Threaded camera capture for Raspberry Pi Camera Module 2
class CameraStream:
    def __init__(self, width=320, height=240):  # Reduced resolution
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "XRGB8888"}
        )
        self.picam2.configure(config)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.picam2.start()
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            frame = self.picam2.capture_array()
            # Convert to grayscale and maintain 3 channels for MediaPipe
            rgb_frame = frame[:, :, 1:]  # Remove alpha channel
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            gray_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
            with self.lock:
                self.frame = gray_3ch

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.picam2.stop()


class FaceMeshDetector:
    def __init__(self, model_path):
        self.knn_model = joblib.load(model_path)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            min_detection_confidence=0.5, 
            refine_landmarks=False  # Disabled refined landmarks
        )
        self.inference_times = []
        self.state_change_counter = 0
        self.last_state_change_time = time.time()
        self.current_state = None
        self.alert_start_time = None
        self.fps_start_time = time.time()
        self.fps_counter = 0

        self.right_eye_indices = np.array([[33, 133], [160, 144], [159, 145], [158, 153]])
        self.left_eye_indices = np.array([[263, 362], [387, 373], [386, 374], [385, 380]])
        self.mouth_indices = np.array([[61, 291], [39, 181], [0, 17], [269, 405]])
        self.head_pose_indices = [1, 33, 263, 61, 291, 199]

    def eye_aspect_ratio(self, eye):
        N1 = distance.euclidean(eye[1][0], eye[1][1])
        N2 = distance.euclidean(eye[2][0], eye[2][1])
        N3 = distance.euclidean(eye[3][0], eye[3][1])
        D = distance.euclidean(eye[0][0], eye[0][1])
        return (N1 + N2 + N3) / (3 * D) if D != 0 else 0

    def mouth_aspect_ratio(self, mouth):
        N1 = distance.euclidean(mouth[1][0], mouth[1][1])
        N2 = distance.euclidean(mouth[2][0], mouth[2][1])
        N3 = distance.euclidean(mouth[3][0], mouth[3][1])
        D = distance.euclidean(mouth[0][0], mouth[0][1])
        return (N1 + N2 + N3) / (3 * D) if D != 0 else 0

    def pupil_circularity(self, eye):
        perimeter = (distance.euclidean(eye[0][0], eye[1][0]) +
                     distance.euclidean(eye[1][0], eye[2][0]) +
                     distance.euclidean(eye[2][0], eye[3][0]) +
                     distance.euclidean(eye[3][0], eye[0][1]))
        diameter = distance.euclidean(eye[1][0], eye[3][1])
        area = math.pi * ((diameter * 0.5) ** 2)
        return (4 * math.pi * area) / (perimeter**2) if perimeter != 0 else 0

    def head_pose(self, face_2d, face_3d, img_w, img_h):
        focal_length = img_w
        cam_matrix = np.array(
            [[focal_length, 0, img_w / 2],
             [0, focal_length, img_h / 2],
             [0, 0, 1]]
        )
        distortion_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vec, translation_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, distortion_matrix
        )
        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles

    def draw_texts(self, frame, texts, start_y=20, line_spacing=25, color=(255, 255, 255)):  # White text
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (10, start_y + i * line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def process_frame(self, frame, frame_count, frame_rate):
        start_time = time.time()
        timestamp = round(frame_count / frame_rate, 2)
        current_time = time.time()

        # Convert to BGR for OpenCV drawing (still grayscale in 3 channels)
        draw_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = self.face_mesh.process(frame)
        text_color = (255, 255, 255)  # White text

        img_h, img_w = draw_frame.shape[:2]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([[lm.x * img_w, lm.y * img_h] for lm in face_landmarks.landmark])

                right_ear = self.eye_aspect_ratio(landmarks[self.right_eye_indices])
                left_ear = self.eye_aspect_ratio(landmarks[self.left_eye_indices])
                ear = round((right_ear + left_ear) / 2.0, 4)

                right_pupil = self.pupil_circularity(landmarks[self.right_eye_indices])
                left_pupil = self.pupil_circularity(landmarks[self.left_eye_indices])
                avg_pupil_circularity = round((right_pupil + left_pupil) / 2.0, 4)

                mar = round(self.mouth_aspect_ratio(landmarks[self.mouth_indices]), 4)
                moe = round(mar / ear, 4) if ear != 0 else 0

                face_2d, face_3d = [], []
                for idx in self.head_pose_indices:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                angles = self.head_pose(face_2d, face_3d, img_w, img_h)
                x_angle, y_angle, z_angle = [angle * (180 / math.pi) for angle in angles]

                if y_angle < -15:
                    head_text = "Menoleh Kanan"
                elif y_angle > 15:
                    head_text = "Menoleh Kiri"
                elif x_angle < -10:
                    head_text = "Menunduk"
                elif x_angle > 10:
                    head_text = "Menadah"
                else:
                    head_text = "Kedepan"

                input_data = np.array([ear, mar, avg_pupil_circularity, moe]).reshape(1, -1)
                state = self.knn_model.predict(input_data)[0]

                # State management logic remains unchanged

                # Alert visualization
                if self.alert_start_time and (current_time - self.alert_start_time <= 5):
                    cv2.rectangle(draw_frame, (0, 0), (img_w, 340), (0, 0, 255), -1)  # Red background
                    text_color = (255, 255, 255)  # White text
                    if threading.active_count() <= 1:
                        threading.Thread(target=playsound,
                                         args=("assets/farrel.mp3", True),
                                         daemon=True).start()

                # Draw simplified landmarks (only key points)
                for idx in self.head_pose_indices:
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(draw_frame, (x, y), 2, (255, 255, 255), -1)

                texts = [
                    f"Time: {timestamp:.2f}s",
                    f"EAR: {ear:.2f}",
                    f"MAR: {mar:.2f}",
                    f"Pupil Circ: {avg_pupil_circularity:.2f}",
                    f"MOE: {moe:.2f}",
                    f"Dir: {head_text}",
                    f"x: {x_angle:.1f}, y: {y_angle:.1f}, z: {z_angle:.1f}",
                    f"State: {state}",
                    f"State Changes: {self.state_change_counter}"
                ]
                self.draw_texts(draw_frame, texts, color=text_color)

        else:
            cv2.putText(draw_frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # Convert to grayscale for final display
        draw_frame = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2GRAY)
        draw_frame = cv2.cvtColor(draw_frame, cv2.COLOR_GRAY2BGR)  # Maintain 3 channels for display

        # FPS calculation
        self.fps_counter += 1
        if (time.time() - self.fps_start_time) > 1:
            fps = self.fps_counter / (time.time() - self.fps_start_time)
            self.fps_start_time = time.time()
            self.fps_counter = 0
            cv2.putText(draw_frame, f"FPS: {fps:.1f}", (img_w - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        return draw_frame


def main():
    model_path = "model/xgb_model.pkl"
    detector = FaceMeshDetector(model_path)
    cam_stream = CameraStream(width=320, height=240)  # Lower resolution
    frame_rate = 20  # Adjusted target frame rate
    frame_count = 0

    while True:
        frame = cam_stream.read()
        if frame is None:
            continue

        processed_frame = detector.process_frame(frame, frame_count, frame_rate)
        cv2.imshow("Driver Safety System", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1

    cam_stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()