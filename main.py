# main.py
from layers.vision_input import VisionInput
from layers.biomechanics import Biomechanics
from layers.form_validation import FormValidator
from layers.serializer import Serializer
from layers.ai_integration import AIIntegration # <-- Import the new class
import cv2
import time

def main():
    vision = VisionInput()
    bio = Biomechanics()
    validator = FormValidator()
    serializer = Serializer()
    ai_coach = AIIntegration() # <-- Create an instance of the AI coach
    start_time = time.time()
    
    print("\nStarting KinexAI... Press 'q' to quit.")

    while True:
        frame, keypoints = vision.get_frame_and_keypoints()
        if frame is None:
            break
        
        cv2.imshow("KinexAI - Pose Detection", frame)

        if time.time() - start_time >= 5.0:
            if keypoints:
                angles = bio.compute_angles(keypoints)
                if angles:
                    feedback = validator.validate_squat(angles, keypoints, bio)
                    payload = serializer.create_payload(int(time.time() * 1000), angles, feedback)
                    
                    # --- GET AND PRINT AI FEEDBACK ---
                    ai_response = ai_coach.get_ai_feedback(payload)
                    print(f"\nAI Coach: {ai_response}")

            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vision.release()

if __name__ == "__main__":
    main()