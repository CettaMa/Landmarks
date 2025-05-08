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
from picamera2 import Picamera2, Preview
import os

# Configure for Raspberry Pi performance
# Add these at the top of your file
os.environ["MP_NUM_THREADS"] = "4"          # Use all 4 CPU cores
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # Optimize BLAS threading
os.environ["OMP_NUM_THREADS"] = "1"

# Constants
OPTIMIZED_RESOLUTION = (640, 480)  # Reduced resolution
LANDMARK_INDICES = {
    'RIGHT_EYE': [[33, 133], [160, 144], [159, 145], [158, 153]],
    'LEFT_EYE': [[263, 362], [387, 373], [386, 374], [385, 380]],
    'MOUTH': [[61, 291], [39, 181], [0, 17], [269, 405]],
    'HEAD_POSE': [1, 33, 263, 61, 291, 199]
}
TEXT_COLOR_DEFAULT = (255, 255, 255)
TEXT_COLOR_ALERT = (0, 255, 255)

class FaceMeshDetector:
    def __init__(self, model_path):
        # Load model with memory mapping for efficiency
        self.knn_model = joblib.load(model_path, mmap_mode='r')
        if hasattr(self.knn_model, 'n_jobs'):
            self.knn_model.n_jobs = 1  # Limit CPU threads for model

        # Optimized Face Mesh configuration
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,  # Disable iris detection
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Hardware-accelerated Object Detector
        base_options = python.BaseOptions(
            model_asset_path='model/objek.tflite',
        )
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5,
            running_mode=vision.RunningMode.LIVE_STREAM,
        )
        self.object_detector = vision.ObjectDetector.create_from_options(options)

        # Pre-allocate arrays for head pose calculation
        self.face_3d = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0]
        ], dtype=np.float64)
        
        # State management
        self.last_state_change = time.monotonic()  # Better for time measurements
        self.performance_stats = {
            'fps': 0,
            'inference_time': 0,
            'blink_rate': 0
        }

    def calculate_ratios(self, landmarks, img_size):
        """Optimized ratio calculations using numpy vectorization"""
        img_w, img_h = img_size
        landmarks = landmarks * np.array([img_w, img_h])
        
        # Eye calculations
        right_eye = landmarks[LANDMARK_INDICES['RIGHT_EYE']]
        left_eye = landmarks[LANDMARK_INDICES['LEFT_EYE']]
        
        ear = (self.eye_aspect_ratio(right_eye) + 
               self.eye_aspect_ratio(left_eye)) / 2
        
        # Mouth calculations
        mouth = landmarks[LANDMARK_INDICES['MOUTH']]
        mar = self.mouth_aspect_ratio(mouth)
        
        return round(ear, 4), round(mar, 4)

    @staticmethod
    def eye_aspect_ratio(eye):
        """Vectorized EAR calculation"""
        distances = np.linalg.norm(eye[:, 0] - eye[:, 1], axis=1)
        return np.mean(distances[1:4]) / (3 * distances[0])

    @staticmethod
    def mouth_aspect_ratio(mouth):
        """Vectorized MAR calculation"""
        distances = np.linalg.norm(mouth[:, 0] - mouth[:, 1], axis=1)
        return np.mean(distances[1:4]) / (3 * distances[0])

    def process_frame(self, frame):
        """Optimized frame processing pipeline"""
        start_time = time.monotonic()
        img_h, img_w = frame.shape[:2]
        
        # Convert once and reuse
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Parallel processing where possible
        face_results = self.face_mesh.process(rgb_frame)
        detection_result = self.object_detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        )
        # Process results
        self._process_face(face_results, (img_w, img_h), gray_frame)
        self._process_objects(detection_result, gray_frame)
        
        # Update performance stats
        self.performance_stats['inference_time'] = time.monotonic() - start_time
        return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    def _process_face(self, results, img_size, frame):
        """Optimized face processing with early exits"""
        if not results.multi_face_landmarks:
            self._draw_text(frame, "No face detected", (10, 30))
            return

        landmarks = np.array([[lm.x, lm.y] for lm in 
                            results.multi_face_landmarks[0].landmark])
        ear, mar = self.calculate_ratios(landmarks, img_size)
        
        # State prediction
        state = self.knn_model.predict([[ear, mar]])[0]
        self._update_state(state)
        
        # Head pose estimation
        angles = self._estimate_head_pose(landmarks[LANDMARK_INDICES['HEAD_POSE']], img_size)
        self._update_display(frame, ear, mar, angles, state)

    def _estimate_head_pose(self, landmarks, img_size):
        """Optimized head pose estimation with pre-allocated arrays"""
        img_w, img_h = img_size
        focal_length = img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h/2],
            [0, focal_length, img_w/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        _, rotation_vec, _ = cv2.solvePnP(
            self.face_3d,
            landmarks.astype(np.float64),
            cam_matrix,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        rmat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return np.degrees(angles)

    def _update_display(self, frame, ear, mar, angles, state):
        """Optimized display updates with limited text operations"""
        text_color = TEXT_COLOR_ALERT if self.alert_active() else TEXT_COLOR_DEFAULT
        direction = self._get_head_direction(angles)
        
        # Batch text operations
        texts = [
            f"FPS: {self.performance_stats['fps']:.1f}",
            f"EAR: {ear:.2f} MAR: {mar:.2f}",
            f"Head: {direction}",
            f"State: {state}"
        ]
        
        for i, text in enumerate(texts):
            self._draw_text(frame, text, (10, 30 + 30*i), text_color)

    def _draw_text(self, frame, text, position, color):
        """Optimized text rendering"""
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 1, cv2.LINE_AA)

class PiCameraStream:
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={
                "size": (640, 480),  # Lower resolution
                "format": "YUV420"   # Hardware-accelerated format
            },
            controls={
                "FrameRate": 20,      # Reduced frame rate
                "AwbMode": "Auto",    # Auto white balance
                "ExposureTime": 10000  # Fixed exposure
            }
        )
        self.picam2.configure(config)
        self.picam2.start_preview(Preview.NULL)  # Disable preview overhead
        self.picam2.start()

def main():
    detector = FaceMeshDetector("model/xgb_model.pkl")
    camera = PiCameraStream()
    
    # Warmup detectors
    _ = detector.process_frame(camera.get_frame())
    
    fps_counter = 0
    fps_start = time.monotonic()
    
    try:
        while True:
            frame = camera.get_frame()
            processed = detector.process_frame(frame)
            
            # Update FPS counter
            fps_counter += 1
            if time.monotonic() - fps_start >= 1:
                detector.performance_stats['fps'] = fps_counter
                fps_counter = 0
                fps_start = time.monotonic()
            
            cv2.imshow("Driver Monitor", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()