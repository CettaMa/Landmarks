import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import distance
import math
import joblib
import time
from playsound import playsound
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from picamera2 import Picamera2  # Ensure Picamera2 is installed

# Constants for landmark indices
RIGHT_EYE_INDICES = [[33, 133], [160, 144], [159, 145], [158, 153]]
LEFT_EYE_INDICES = [[263, 362], [387, 373], [386, 374], [385, 380]]
MOUTH_INDICES = [[61, 291], [39, 181], [0, 17], [269, 405]]
HEAD_POSE_INDICES = [1, 33, 263, 61, 291, 199]

# Updated colors for better visibility on grayscale
TEXT_COLOR_DEFAULT = (255, 255, 255)  # White
TEXT_COLOR_ALERT = (0, 255, 255)      # Yellow

class FaceMeshDetector:
    def __init__(self, model_path):
        try:
            self.knn_model = joblib.load(model_path)
        except FileNotFoundError:
            raise Exception(f"Model file not found at {model_path}")

        # Initialize Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
        )

        # Initialize Object Detector
        base_options = python.BaseOptions(model_asset_path='model/objek.tflite')
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5
        )
        self.object_detector = vision.ObjectDetector.create_from_options(options)

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            color=(128, 0, 128), thickness=2, circle_radius=1
        )

        # Initialize performance tracking variables
        self.inference_times = []
        self.state_change_counter = 0
        self.last_state_change_time = time.time()
        self.current_state = None
        self.alert_start_time = None
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.blink_counter = 0
        self.blink_start_time = time.time()
        self.last_blink_time = 0 

    @staticmethod
    def eye_aspect_ratio(eye):
        """Calculate the Eye Aspect Ratio (EAR)."""
        N1 = distance.euclidean(eye[1][0], eye[1][1])
        N2 = distance.euclidean(eye[2][0], eye[2][1])
        N3 = distance.euclidean(eye[3][0], eye[3][1])
        D = distance.euclidean(eye[0][0], eye[0][1])
        return (N1 + N2 + N3) / (3 * D)

    @staticmethod
    def mouth_aspect_ratio(mouth):
        """Calculate the Mouth Aspect Ratio (MAR)."""
        N1 = distance.euclidean(mouth[1][0], mouth[1][1])
        N2 = distance.euclidean(mouth[2][0], mouth[2][1])
        N3 = distance.euclidean(mouth[3][0], mouth[3][1])
        D = distance.euclidean(mouth[0][0], mouth[0][1])
        return (N1 + N2 + N3) / (3 * D)

    @staticmethod
    def pupil_circularity(eye):
        """Calculate the Pupil Circularity."""
        perimeter = (
            distance.euclidean(eye[0][0], eye[1][0])
            + distance.euclidean(eye[1][0], eye[2][0])
            + distance.euclidean(eye[2][0], eye[3][0])
            + distance.euclidean(eye[3][0], eye[0][1])
        )
        area = math.pi * ((distance.euclidean(eye[1][0], eye[3][1]) * 0.5) ** 2)
        return (4 * math.pi * area) / (perimeter**2)

    def calculate_blink_rate(self):
        """Calculate the average blinking rate per minute."""
        elapsed_time = time.time() - self.blink_start_time
        if elapsed_time > 0:
            return (self.blink_counter / elapsed_time) * 60  # Blinks per minute
        return 0

    @staticmethod
    def head_pose(face_2d, face_3d, img_w, img_h):
        """Calculate head pose angles."""
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

    def play_alert(self):
        """Play alert sound in a separate thread."""
        if threading.active_count() <= 1:  # Avoid multiple threads
            threading.Thread(target=playsound, args=("assets/alerts.mp3", True), daemon=True).start()

    def draw_text(self, frame, text, position, color=TEXT_COLOR_DEFAULT):
        """Utility function to draw text on the frame."""
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    def process_frame(self, frame, timestamp):
        """Process a single video frame."""
        start_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create grayscale version for display
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
        # Process face mesh
        face_results = self.face_mesh.process(rgb_frame)
        img_h, img_w, _ = frame.shape
        text_color = TEXT_COLOR_DEFAULT
        current_time = time.time()

        # Process object detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.object_detector.detect(mp_image)
        
        # Draw object detection results on grayscale frame
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            
            # Draw bounding box in green
            cv2.rectangle(gray_frame_bgr, start_point, end_point, (0, 255, 0), 2)
            
            # Draw label and score
            category = detection.categories[0]
            class_name = category.category_name
            score = round(category.score, 2)
            label = f"{class_name}: {score}"
            
            cv2.putText(gray_frame_bgr, label, 
                        (bbox.origin_x, bbox.origin_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
                landmarks[:, 0] *= img_w
                landmarks[:, 1] *= img_h

                right_eye = landmarks[RIGHT_EYE_INDICES]
                left_eye = landmarks[LEFT_EYE_INDICES]
                mouth = landmarks[MOUTH_INDICES]
                pupil = round(
                    (self.pupil_circularity(right_eye) + self.pupil_circularity(left_eye)) / 2.0, 
                    4
                )
                ear = round(
                    (self.eye_aspect_ratio(right_eye) + self.eye_aspect_ratio(left_eye)) / 2.0,
                    4
                )
                mar = round(self.mouth_aspect_ratio(mouth), 4)
                moe = round(mar / ear, 4)
                
                if ear < 0.16 and current_time - self.last_blink_time > 0.2:
                    self.blink_counter += 1
                    self.last_blink_time = current_time

                blink_rate = self.calculate_blink_rate()

                # Head pose estimation
                face_2d = []
                face_3d = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in HEAD_POSE_INDICES:
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
                head_text = (
                    "Menoleh Kanan" if y < -2 else
                    "Menoleh Kiri" if y > 2 else
                    "Menunduk" if x < -2 else
                    "Menadah" if x > 2 else
                    "Kedepan"
                )

                # Predict state
                input_data = np.array([ear, mar, pupil, moe]).reshape(1, -1)
                state = self.knn_model.predict(input_data)[0]

                # Handle state changes
                if self.current_state is None or state != self.current_state:
                    if state == 1 and (current_time - self.last_state_change_time > 2):
                        self.state_change_counter += 1
                        self.current_state = state
                        self.last_state_change_time = current_time
                        if self.state_change_counter >= 2:
                            self.alert_start_time = current_time
                    elif state == 0:
                        self.current_state = 0
                else:
                    self.last_state_change_time = current_time

                # Trigger alert
                if self.alert_start_time and (current_time - self.alert_start_time <= 5):
                    text_color = TEXT_COLOR_ALERT
                    self.play_alert()
                elif self.alert_start_time and (current_time - self.alert_start_time > 5):
                    self.alert_start_time = None
                    self.state_change_counter = 0

                # Draw landmarks and text on grayscale frame
                for landmark in landmarks:
                    cv2.circle(
                        gray_frame_bgr, (int(landmark[0]), int(landmark[1])), 
                        1, (0, 255, 0), -1
                    )

                self.draw_text(gray_frame_bgr, f"Time: {timestamp:.2f}s", (10, 30), text_color)
                self.draw_text(gray_frame_bgr, f"EAR: {ear:.2f}", (10, 60), text_color)
                self.draw_text(gray_frame_bgr, f"MAR: {mar:.2f}", (10, 90), text_color)
                self.draw_text(gray_frame_bgr, f"MOE: {moe:.2f}", (10, 120), text_color)
                self.draw_text(gray_frame_bgr, f"Direction: {head_text}", (10, 150), text_color)
                self.draw_text(gray_frame_bgr, f"State: {state}", (10, 180), text_color)
                self.draw_text(gray_frame_bgr, f"State Changes: {self.state_change_counter}", (10, 210), text_color)
                self.draw_text(gray_frame_bgr, f"X: {round(x,3)} Y: {round(y,3)}", (10, 240), text_color)
                self.draw_text(gray_frame_bgr, f"Blink Rate: {blink_rate:.2f} BPM", (10, 300), text_color)
                self.draw_text(gray_frame_bgr, f"Blink Count: {self.blink_counter}", (10, 330), text_color)
        else:
            self.draw_text(gray_frame_bgr, "No face detected", (10, 30), text_color)

        # Calculate FPS
        self.fps_counter += 1
        fps = self.fps_counter / (current_time - self.fps_start_time)
        self.draw_text(gray_frame_bgr, f"FPS: {fps:.2f}", (10, 270), text_color)

        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        return gray_frame_bgr

class CameraStream:
    def __init__(self, width=854, height=480):
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
            # Convert XRGB to RGB (remove alpha channel)
            frame = frame[:, :, 1:]
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            # Convert RGB to BGR for OpenCV compatibility
            return cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

    def release(self):
        self.stopped = True
        self.picam2.stop()

def main():
    model_path = r"model/xgb_model.pkl"
    detector = FaceMeshDetector(model_path)

    # Initialize camera stream with desired resolution
    camera = CameraStream(width=854, height=480)
    start_time = time.time()

    while True:
        frame = camera.read()
        if frame is None:
            continue

        # Calculate current timestamp
        timestamp = time.time() - start_time

        # Process the frame
        processed_frame = detector.process_frame(frame, timestamp)

        cv2.imshow("Driver Safety System", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()