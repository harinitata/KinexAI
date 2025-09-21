# layers/serializer.py
import json

class Serializer:
    def __init__(self):
        pass

    def create_payload(self, frame_timestamp, angles, validation_feedback):
        """
        Creates a JSON-serializable dictionary with all data.
        """
        payload = {
            "timestamp": frame_timestamp,
            "angles": angles,
            "form_feedback": validation_feedback
        }
        return payload