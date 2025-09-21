# This file will:
# Open webcam
# Run MediaPipe Pose
# Return keypoints (x, y, confidence)
# Draw skeleton overlay (for debugging)

#--------------------------------------------------------------------------

# layers/vision_input.py

import cv2
import mediapipe as mp

class VisionInput:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def get_frame_and_keypoints(self):
        """Capture frame, run pose detection, return keypoints list."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        keypoints = []
        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                keypoints.append({
                    "id": idx,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

            # Draw skeleton on frame
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
            )

        return frame, keypoints

    def release(self):
        """Release camera resources."""
        self.cap.release()
        cv2.destroyAllWindows()