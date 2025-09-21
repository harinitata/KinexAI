# main.py
from layers.vision_input import VisionInput
from layers.biomechanics import Biomechanics
from layers.form_validation import FormValidator
from layers.serializer import Serializer
import cv2
import time
import json

def main():
    vision = VisionInput()
    bio = Biomechanics()
    validator = FormValidator()
    serializer = Serializer()
    start_time = time.time()
    
    while True:
        frame, keypoints = vision.get_frame_and_keypoints()
        if frame is None:
            break

        cv2.imshow("KinexAI - Pose Detection", frame)

        # Process every ~5 seconds as per your plan
        if time.time() - start_time >= 5.0:
            if keypoints:
                angles = bio.compute_angles(keypoints)
                if angles:
                    # Step 1: Run form validation
                    feedback = validator.validate_squat(angles)

                    # Step 2: Create JSON payload
                    current_time_ms = int(time.time() * 1000)
                    payload = serializer.create_payload(current_time_ms, angles, feedback)

                    # For now, just print the payload
                    # In a future step, you'd send this to your AI integration layer
                    print(json.dumps(payload, indent=2))
            
            start_time = time.time() # Reset timer

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vision.release()

if __name__ == "__main__":
    main()