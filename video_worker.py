import cv2
import time
import threading

class OpticalBuffer:
    def __init__(self, camera_index=0, sample_interval=0.5):
        self.cap = cv2.VideoCapture(camera_index)
        self.sample_interval = sample_interval
        self.latest_keyframe = None
        self.lock = threading.Lock()
        self.running = True

        # Start non-blocking capture thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        last_sample_time = 0
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            current_time = time.time()
            # Enforce Delta_t = 0.5s sampling (2 FPS lossy decimation)
            if current_time - last_sample_time >= self.sample_interval:
                with self.lock:
                    self.latest_keyframe = frame
                last_sample_time = current_time

    def get_frame(self):
        with self.lock:
            return self.latest_keyframe.copy() if self.latest_keyframe is not None else None

    def release(self):
        self.running = False
        self.cap.release()

if __name__ == "__main__":
    print("Testing Optical Buffer Ingestion B_v at 2 FPS...")
    cam = OpticalBuffer()
    time.sleep(1.0)
    
    for i in range(5):
        frame = cam.get_frame()
        if frame is not None:
            print(f"Captured Keyframe #{i+1} - Shape: {frame.shape}")
        time.sleep(0.5)
        
    cam.release()
    print("Optical buffer test complete.")