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
import pygame

# Constants for landmark indices
RIGHT_EYE_INDICES = [[33, 133], [160, 144], [159, 145], [158, 153]]
LEFT_EYE_INDICES = [[263, 362], [387, 373], [386, 374], [385, 380]]
MOUTH_INDICES = [[61, 291], [39, 181], [0, 17], [269, 405]]
HEAD_POSE_INDICES = [1, 33, 263, 61, 291, 199]

# Thresholds and parameters
BLINK_THRESHOLD = 0.15
BLINK_COOLDOWN = 0.2
STATE_CHANGE_THRESHOLD = 2
ALERT_DURATION = 3
HEAD_ANGLE_THRESHOLD = 2

# Colors for display
TEXT_COLOR_DEFAULT = (255, 255, 255)  # White
TEXT_COLOR_ALERT = (0, 255, 255)      # Yellow
LANDMARK_COLOR = (0, 255, 0)          # Green

class FaceMeshDetector:
    def __init__(self, model_path):
        pygame.mixer.init()
        self._load_model(model_path)
        self._setup_detectors()
        self._initialize_tracking_variables()

    def _load_model(self, model_path):
        """Load the machine learning model."""
        try:
            self.knn_model = joblib.load(model_path)
        except FileNotFoundError:
            raise Exception(f"Model file not found at {model_path}")

    def _setup_detectors(self):
        """Initialize face mesh and object detectors."""
        # Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            min_detection_confidence=0.5,
            refine_landmarks=True,
        )
        
        # Object detector setup
        base_options = python.BaseOptions(model_asset_path='model/model (5).tflite')
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.6
        )
        self.object_detector = vision.ObjectDetector.create_from_options(options)
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            color=(128, 0, 128), thickness=2, circle_radius=1
        )

    def _initialize_tracking_variables(self):
        """Initialize performance and state tracking variables."""
        current_time = time.time()
        self.inference_times = []
        self.state_change_counter = 0
        self.last_state_change_time = current_time
        self.current_state = None
        self.alert_start_time = None
        self.fps_start_time = current_time
        self.fps_counter = 0
        self.blink_counter = 0
        self.blink_start_time = current_time
        self.last_blink_time = 0
        self.blink_state = False
        self.alert_sound = pygame.mixer.Sound("assets/farrel.mp3")

    @staticmethod
    def calc_aspect_ratio(points):
        """Calculate aspect ratio (used for both eyes and mouth)."""
        N1 = distance.euclidean(points[1][0], points[1][1])
        N2 = distance.euclidean(points[2][0], points[2][1])
        N3 = distance.euclidean(points[3][0], points[3][1])
        D = distance.euclidean(points[0][0], points[0][1])
        return (N1 + N2 + N3) / (3 * D)

    eye_aspect_ratio = calc_aspect_ratio  # Alias for backward compatibility
    mouth_aspect_ratio = calc_aspect_ratio  # Alias for backward compatibility

    @staticmethod
    def pupil_circularity(eye):
        """Calculate the Pupil Circularity."""
        # Calculate perimeter of eye
        perimeter = (
            distance.euclidean(eye[0][0], eye[1][0])
            + distance.euclidean(eye[1][0], eye[2][0])
            + distance.euclidean(eye[2][0], eye[3][0])
            + distance.euclidean(eye[3][0], eye[0][1])
        )
        # Calculate area assuming elliptical shape
        radius = distance.euclidean(eye[1][0], eye[3][1]) * 0.5
        area = math.pi * (radius ** 2)
        # Perfect circle has circularity of 1
        return (4 * math.pi * area) / (perimeter**2)

    def calculate_blink_rate(self):
        """Calculate blinking rate per minute."""
        elapsed_time = max(0.001, time.time() - self.blink_start_time)  # Avoid division by zero
        return (self.blink_counter / elapsed_time) * 60

    @staticmethod
    def head_pose(face_2d, face_3d, img_w, img_h):
        """Calculate head pose angles using PnP algorithm."""
        # Camera matrix approximation
        focal_length = img_w
        center = (img_h / 2, img_w / 2)
        cam_matrix = np.array(
            [[focal_length, 0, center[0]], 
             [0, focal_length, center[1]], 
             [0, 0, 1]]
        )
        # Solve PnP to get rotation vector
        distortion_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vec, _ = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, distortion_matrix
        )
        # Convert rotation vector to Euler angles
        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles

    def play_alert(self):
        """Play alert sound without blocking the main thread."""
        if threading.active_count() <= 1:  # Prevent multiple alert sounds
            threading.Thread(
                target=playsound, 
                args=("assets/farrel.mp3", True), 
                daemon=True
            ).start()

    def draw_text(self, frame, text, position, color=TEXT_COLOR_DEFAULT):
        """Draw text on frame with consistent styling."""
        cv2.putText(
            frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2
        )

    def _detect_objects(self, rgb_frame, output_frame):
        """Detect objects in the frame."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.object_detector.detect(mp_image)
        
        # Draw detection results
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            
            # Draw bounding box
            cv2.rectangle(output_frame, start_point, end_point, LANDMARK_COLOR, 2)
            
            # Draw label
            category = detection.categories[0]
            label = f"{category.category_name}: {round(category.score, 2)}"
            cv2.putText(
                output_frame, label, 
                (bbox.origin_x, bbox.origin_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, LANDMARK_COLOR, 2
            )
        
        return detection_result

    def _extract_face_features(self, face_landmarks, img_w, img_h):
        """Extract features from detected face landmarks."""
        # Convert landmarks to numpy array
        landmarks = np.array([(lm.x * img_w, lm.y * img_h) for lm in face_landmarks.landmark])
        
        # Get eye and mouth landmarks
        right_eye = landmarks[RIGHT_EYE_INDICES]
        left_eye = landmarks[LEFT_EYE_INDICES]
        mouth = landmarks[MOUTH_INDICES]
        
        # Calculate facial metrics
        pupil = round((self.pupil_circularity(right_eye) + self.pupil_circularity(left_eye)) / 2.0, 4)
        ear = round((self.eye_aspect_ratio(right_eye) + self.eye_aspect_ratio(left_eye)) / 2.0, 4)
        mar = round(self.mouth_aspect_ratio(mouth), 4)
        moe = round(mar / ear, 4)
        
        return landmarks, right_eye, left_eye, mouth, ear, mar, moe, pupil

    def _calculate_head_pose(self, face_landmarks, img_w, img_h):
        """Calculate head pose angles."""
        # Extract points for head pose estimation
        face_2d = []
        face_3d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx in HEAD_POSE_INDICES:
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        # Convert to numpy arrays
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        # Calculate angles
        angles = self.head_pose(face_2d, face_3d, img_w, img_h)
        
        # Convert to degrees
        x, y, z = [angle * (180 / math.pi) for angle in angles]
        
        # Determine head direction
        if y < -HEAD_ANGLE_THRESHOLD:
            head_text = "Menoleh Kanan"
        elif y > HEAD_ANGLE_THRESHOLD:
            head_text = "Menoleh Kiri"
        elif x < -HEAD_ANGLE_THRESHOLD:
            head_text = "Menunduk"
        elif x > HEAD_ANGLE_THRESHOLD:
            head_text = "Menadah"
        else:
            head_text = "Kedepan"
            
        return x, y, z, head_text

    def _detect_blinks(self, ear, current_time):
        """Detect eye blinks and update counters."""
        if ear < BLINK_THRESHOLD:
            # Check if enough time has passed since last blink
            if self.last_blink_time == 0 or (current_time - self.last_blink_time > BLINK_COOLDOWN):
                self.blink_counter += 1
                self.last_blink_time = current_time

    def _handle_state_transitions(self, state, current_time):
        """Handle transitions between alertness states."""
        # Check for state change
        if self.current_state is None or state != self.current_state:
            # Transition to drowsy state (1)
            if state == 1 and (current_time - self.last_state_change_time > STATE_CHANGE_THRESHOLD):
                self.state_change_counter += 1
                self.current_state = state
                self.last_state_change_time = current_time
                # Trigger alert after multiple drowsy detections
                if self.state_change_counter >= 2:
                    self.alert_start_time = current_time
            # Transition to alert state (0)
            elif state == 0:
                self.current_state = 0
        else:
            # Update last state change time
            self.last_state_change_time = current_time

        # Clear alert after specified duration
        if self.alert_start_time and (current_time - self.alert_start_time > ALERT_DURATION):
            self.alert_start_time = None
            self.state_change_counter = 0

    def process_frame(self, frame, frame_count, frame_rate):
        """Process a single video frame and analyze driver state."""
        start_time = time.time()
        current_time = start_time
        timestamp = round(frame_count / frame_rate, 2)
        
        # Convert frame to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create grayscale version for display
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
        # Get frame dimensions
        img_h, img_w, _ = frame.shape
        
        # Default text color
        text_color = TEXT_COLOR_DEFAULT
        
        # Detect objects in frame
        detection_result = self._detect_objects(rgb_frame, output_frame)
        
        # Process face mesh
        face_results = self.face_mesh.process(rgb_frame)
        
        # Check if any faces were detected
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Extract facial features
                landmarks, right_eye, left_eye, mouth, ear, mar, moe, pupil = self._extract_face_features(
                    face_landmarks, img_w, img_h
                )
                
                # Detect blinks
                self._detect_blinks(ear, current_time)
                blink_rate = self.calculate_blink_rate()
                
                # Calculate head pose
                x, y, z, head_text = self._calculate_head_pose(face_landmarks, img_w, img_h)
                
                # Predict driver state using model
                input_data = np.array([ear, mar, pupil, moe]).reshape(1, -1)
                state = self.knn_model.predict(input_data)[0]
                
                # Handle state transitions
                self._handle_state_transitions(state, current_time)
                
                # Trigger alert if needed
                if self.alert_start_time and (current_time - self.alert_start_time <= ALERT_DURATION):
                    text_color = TEXT_COLOR_ALERT
                    self.play_alert()
                
                # Draw face landmarks
                for landmark in landmarks:
                    cv2.circle(
                        output_frame, 
                        (int(landmark[0]), int(landmark[1])), 
                        1, LANDMARK_COLOR, -1
                    )
                
                # Display metrics and state information
                self._display_metrics(
                    output_frame, timestamp, ear, mar, moe, head_text, 
                    state, x, y, blink_rate, text_color
                )
        else:
            # No face detected
            self.draw_text(output_frame, "No face detected", (10, 30), text_color)
        
        # Calculate and display FPS
        self.fps_counter += 1
        fps = self.fps_counter / (current_time - self.fps_start_time)
        self.draw_text(output_frame, f"FPS: {fps:.2f}", (10, 270), text_color)
        
        # Track inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return output_frame

    def _display_metrics(self, frame, timestamp, ear, mar, moe, head_text, 
                        state, x, y, blink_rate, text_color):
        """Display all metrics and state information on the frame."""
        metrics = [
            (f"Time: {timestamp:.2f}s", 30),
            (f"EAR: {ear:.2f}", 60),
            (f"MAR: {mar:.2f}", 90),
            (f"MOE: {moe:.2f}", 120),
            (f"Direction: {head_text}", 150),
            (f"State: {state}", 180),
            (f"State Changes: {self.state_change_counter}", 210),
            (f"X: {round(x,3)} Y: {round(y,3)}", 240),
            (f"FPS: {self.fps_counter / max(0.001, time.time() - self.fps_start_time):.2f}", 270),
            (f"Blink Rate: {blink_rate:.2f} BPM", 300),
            (f"Blink Count: {self.blink_counter}", 330)
        ]
        
        for text, y_pos in metrics:
            self.draw_text(frame, text, (10, y_pos), text_color)


def main():
    """Main function to run the driver monitoring system."""
    model_path = r"model/svm_model.pkl"
    detector = FaceMeshDetector(model_path)

    # Open webcam
    cap = cv2.VideoCapture(0)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and process frame
        frame = cv2.resize(frame, (854, 480))
        processed_frame = detector.process_frame(frame, frame_count, frame_rate)

        # Display results
        cv2.imshow("Driver Safety System", processed_frame)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()