# layers/form_validation.py
import json

class FormValidator:
    def __init__(self, config_file='config/angles_config.json'):
        self.angle_thresholds = self._load_config(config_file)
        self.feedback = {}

    def _load_config(self, config_file):
        """Loads angle thresholds from a JSON file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Config file '{config_file}' not found.")
            return {}

    def validate_squat(self, angles, keypoints, bio_mechanics_instance):
        """
        Checks angles and CoG balance against predefined squat thresholds.
        """
        self.feedback = {} # Reset feedback for each new validation
        thresholds = self.angle_thresholds.get('squat_parallel', {})
        
        knee_min, knee_max = thresholds.get('knee_angle_target_range', [90, 110])
        hip_max = thresholds.get('hip_angle_target_range', [70, 105])[1] # We only need the max for this check
        torso_max = thresholds.get('torso_max_tilt', 30)
        cog_max_offset = thresholds.get('cog_max_horizontal_offset', 0.07)

        # -- ANGLE VALIDATION --
        avg_knee_angle = (angles.get('knee_left', 180) + angles.get('knee_right', 180)) / 2
        avg_hip_angle = (angles.get('hip_left', 180) + angles.get('hip_right', 180)) / 2

        if avg_knee_angle > knee_max:
            self.feedback['depth'] = "Squat deeper to bring your thighs parallel to the floor."
        
        if avg_hip_angle > hip_max:
            self.feedback['hip_hinge'] = "Not low enough. Hinge more at your hips."

        if angles.get('torso_tilt', 0) > torso_max:
            self.feedback['torso_form'] = "Keep your chest up. You are leaning too far forward."
            
        # -- CENTER OF GRAVITY (BALANCE) VALIDATION --
        center_of_gravity, base_of_support = bio_mechanics_instance.compute_center_of_gravity(keypoints)

        if center_of_gravity and base_of_support:
            horizontal_offset = abs(center_of_gravity['x'] - base_of_support['x'])
            
            if horizontal_offset > cog_max_offset:
                self.feedback['balance'] = "You are leaning too far forward or backward. Keep your weight centered over your feet."

        return self.feedback