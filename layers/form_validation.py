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

    def validate_squat(self, angles):
        """Checks angles against predefined squat thresholds."""
        self.feedback = {}
        thresholds = self.angle_thresholds.get('squat', {})

        # Validate knee angles (should be close to a straight line at the top)
        if angles.get('knee_left') and angles['knee_left'] < thresholds.get('knee_min_angle', 90):
            self.feedback['knee_left_form'] = "Left knee too bent."
        if angles.get('knee_right') and angles['knee_right'] < thresholds.get('knee_min_angle', 90):
            self.feedback['knee_right_form'] = "Right knee too bent."

        # Validate hip angles (should be less than 90 degrees at the bottom)
        if angles.get('hip_left') and angles['hip_left'] > thresholds.get('hip_max_angle', 100):
            self.feedback['hip_left_form'] = "Left hip not low enough."
        if angles.get('hip_right') and angles['hip_right'] > thresholds.get('hip_max_angle', 100):
            self.feedback['hip_right_form'] = "Right hip not low enough."

        # Validate torso tilt (should not be too far forward)
        if angles.get('torso_tilt') and angles['torso_tilt'] > thresholds.get('torso_max_tilt', 30):
            self.feedback['torso_form'] = "Torso leaning too far forward."
            
        return self.feedback