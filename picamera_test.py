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
from picamera2 import Picamera2
from collections import deque
import os

# Environment optimizations for Raspberry Pi
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Configuration constants
OPTIMAL_RESOLUTION = (640, 480)
MAX_QUEUE_SIZE = 2  # Maintain low memory usage
LANDMARK_INDICES = {
    'RIGHT_EYE': [[33, 133], [160, 144], [159, 145], [158, 153]],
    'LEFT_EYE': [[263, 362], [387, 373], [386, 374], [385, 380]],
    'MOUTH': [[61, 291], [39, 181], [0, 17], [269, 405]],
    'HEAD_POSE': [1, 33, 263, 61, 291, 199]
}

class FaceMeshDetector:
    def __init__(self, model_path):
        self.result_queue = deque(maxlen=MAX_QUEUE_SIZE)
        self.last_timestamp = 0
        
        # Load drowsiness detection model
        self.knn_model = joblib.load(model_path)
        
        # Initialize Face Mesh with optimized parameters
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Configure object detector with live stream callback
        base_options = python.BaseOptions(
            model_asset_path='model/objek.tflite'
        )
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5,
            category_allowlist=["phone", "bottle"],
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.detection_callback
        )
        self.object_detector = vision.ObjectDetector.create_from_options(options)

    def detection_callback(self, result, output_image, timestamp_ms):
        """Thread-safe result handling for object detection"""
        with threading.Lock():
            self.result_queue.append((timestamp_ms, result))

    def process_frame(self, frame):
        """Main processing pipeline with timestamp management"""
        timestamp_ms = int(time.monotonic() * 1000)
        
        # Convert frame once for multiple uses
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Async object detection
        self.object_detector.detect_async(mp_image, timestamp_ms)
        
        # Process face mesh synchronously
        face_results = self.face_mesh.process(rgb_frame)
        
        # Get latest detection results
        current_detections = self.get_latest_detections(timestamp_ms)
        
        # Generate output frame
        return self.generate_output(frame, face_results, current_detections)

    def get_latest_detections(self, current_timestamp):
        """Get most recent detections with timestamp validation"""
        with threading.Lock():
            while self.result_queue:
                ts, result = self.result_queue[0]
                if ts <= current_timestamp:
                    self.result_queue.popleft()
                    return result
            return None

    def generate_output(self, frame, face_results, detections):
        """Generate final output frame with overlays"""
        output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)
        
        # Process and draw face landmarks
        if face_results.multi_face_landmarks:
            self.process_face_landmarks(face_results, output_frame)
        
        # Draw object detections
        if detections:
            self.draw_detections(detections, output_frame)
        
        return output_frame

    def process_face_landmarks(self, results, frame):
        """Optimized face landmark processing"""
        landmarks = np.array([[lm.x, lm.y] 
                            for lm in results.multi_face_landmarks[0].landmark])
        img_h, img_w = frame.shape[:2]
        landmarks *= [img_w, img_h]
        
        # Calculate eye metrics
        right_eye = landmarks[LANDMARK_INDICES['RIGHT_EYE']]
        left_eye = landmarks[LANDMARK_INDICES['LEFT_EYE']]
        ear = (self.eye_aspect_ratio(right_eye) + 
               self.eye_aspect_ratio(left_eye)) / 2
        
        # Calculate mouth metrics
        mouth = landmarks[LANDMARK_INDICES['MOUTH']]
        mar = self.mouth_aspect_ratio(mouth)
        
        # Predict driver state
        state = self.knn_model.predict([[ear, mar]])[0]
        self.update_display(frame, ear, mar, state)

    @staticmethod
    def eye_aspect_ratio(eye):
        distances = np.linalg.norm(eye[:,0] - eye[:,1], axis=1)
        return (distances[1] + distances[2] + distances[3]) / (3 * distances[0])

    @staticmethod
    def mouth_aspect_ratio(mouth):
        distances = np.linalg.norm(mouth[:,0] - mouth[:,1], axis=1)
        return (distances[1] + distances[2] + distances[3]) / (3 * distances[0])

    def update_display(self, frame, ear, mar, state):
        """Efficient display updates"""
        text_color = (255, 255, 255) if state == 0 else (0, 255, 255)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        cv2.putText(frame, f"State: {'Alert' if state == 0 else 'Drowsy'}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

class PiCameraController:
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": OPTIMAL_RESOLUTION, "format": "YUV420"},
            controls={"FrameRate": 20}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # Camera warmup

    def get_frame(self):
        yuv420 = self.picam2.capture_array("main")
        return cv2.cvtColor(yuv420, cv2.COLOR_YUV420p2BGR)

    def release(self):
        self.picam2.stop()

def main():
    camera = PiCameraController()
    detector = FaceMeshDetector("model/xgb_model.pkl")
    
    fps_counter = 0
    fps_last = time.monotonic()
    
    try:
        while True:
            start_time = time.monotonic()
            frame = camera.get_frame()
            processed = detector.process_frame(frame)
            
            # Calculate FPS
            fps_counter += 1
            if (time.monotonic() - fps_last) >= 1.0:
                print(f"FPS: {fps_counter}")
                fps_counter = 0
                fps_last = time.monotonic()
            
            cv2.imshow("Driver Monitor", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()