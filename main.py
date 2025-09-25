# main.py
from layers.vision_input import VisionInput
from layers.biomechanics import Biomechanics
from layers.form_validation import FormValidator
from layers.ai_integration import AIIntegration
import cv2
import numpy as np
import threading

# --- SETUP THE AI COACH ---
ai_coach = AIIntegration()
ai_feedback_text = "Start your first set!"

def call_ai_in_background(feedback_to_rephrase):
    """Run AI rephrasing in a separate thread."""
    global ai_feedback_text
    print("\n[AI] Rephrasing feedback...")
    ai_feedback_text = ai_coach.rephrase_feedback(feedback_to_rephrase)
    print(f"[AI] New feedback: {ai_feedback_text}")

def main():
    vision = VisionInput()
    bio = Biomechanics()
    validator = FormValidator()

    rep_counter = 0
    stage = "up"
    form_feedback_text = ""

    # Thresholds for detecting squat phases
    UP_THRESHOLD = 160
    DOWN_THRESHOLD = 110

    # Ideal range for bottom position
    IDEAL_DOWN_MIN = 80
    IDEAL_DOWN_MAX = 100

    min_angle = 180  # Track lowest angle per rep

    print("Starting KinexAI... Press 'q' to quit.")

    while True:
        frame, keypoints = vision.get_frame_and_keypoints()
        if frame is None:
            break

        if keypoints:
            angles = bio.compute_angles(keypoints)
            if angles:
                avg_knee_angle = np.mean([
                    angles.get('knee_left', 180),
                    angles.get('knee_right', 180)
                ])

                # Track min knee angle when going down
                if avg_knee_angle < min_angle:
                    min_angle = avg_knee_angle

                # Phase detection
                if avg_knee_angle < DOWN_THRESHOLD:
                    if stage == 'up':
                        form_feedback_text = ""  # clear previous feedback
                    stage = "down"

                # Rep completed when we go back up
                if avg_knee_angle > UP_THRESHOLD and stage == 'down':
                    stage = "up"
                    rep_counter += 1

                    # --- Evaluate the rep based on min_angle ---
                    if min_angle > IDEAL_DOWN_MAX:
                        form_feedback_text = "Try going a bit deeper next rep!"
                    elif min_angle < IDEAL_DOWN_MIN:
                        form_feedback_text = "Careful — you went too deep!"
                    else:
                        form_feedback_text = "Good rep! Keep it up!"

                    # Call AI in a new thread to rephrase feedback
                    ai_thread = threading.Thread(
                        target=call_ai_in_background,
                        args=(form_feedback_text,)
                    )
                    ai_thread.start()

                    # Reset for next rep
                    min_angle = 180

                # Validate form only during downward movement (real-time micro-checks)
                if stage == "down":
                    feedback = validator.validate_squat(angles, keypoints, bio)
                    if feedback:
                        # Collect all validator messages into a single string
                        form_feedback_text = " | ".join(feedback.values())

        # --- Display on screen ---
        cv2.putText(frame, "AI COACH:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, ai_feedback_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 191, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, f"REPS: {rep_counter}",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"STAGE: {stage.upper()}",
                    (200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("KinexAI - Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vision.release()

if __name__ == "__main__":
    main()
