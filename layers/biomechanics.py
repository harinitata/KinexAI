# layers/biomechanics.py
import math

class Biomechanics:
    def __init__(self):
        pass

    @staticmethod
    def _angle_3pts(a, b, c):
        """
        Calculate angle (in degrees) at point b given 3 points a, b, c.
        Points should be dicts with x, y keys (normalized coords).
        """
        try:
            # Convert to vectors
            v1 = (a['x'] - b['x'], a['y'] - b['y'])
            v2 = (c['x'] - b['x'], c['y'] - b['y'])

            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

            if mag1 == 0 or mag2 == 0:
                return None

            angle_rad = math.acos(max(min(dot / (mag1*mag2), 1.0), -1.0))
            return math.degrees(angle_rad)
        except:
            return None

    def compute_angles(self, keypoints):
        """
        Compute knee, hip, torso angles from keypoints.
        keypoints: list of dicts from vision_input
        Returns dictionary of angles.
        """
        # MediaPipe Pose indices
        # https://google.github.io/mediapipe/solutions/pose.html
        L_HIP, R_HIP = 23, 24
        L_KNEE, R_KNEE = 25, 26
        L_ANKLE, R_ANKLE = 27, 28
        L_SHOULDER, R_SHOULDER = 11, 12

        # Helper function to fetch point safely
        def kp(idx):
            return {"x": keypoints[idx]['x'], "y": keypoints[idx]['y']} if keypoints[idx]['visibility'] > 0.5 else None

        lh, rh = kp(L_HIP), kp(R_HIP)
        lk, rk = kp(L_KNEE), kp(R_KNEE)
        la, ra = kp(L_ANKLE), kp(R_ANKLE)
        ls, rs = kp(L_SHOULDER), kp(R_SHOULDER)

        results = {}

        # Knee Angles
        if lh and lk and la:
            results['knee_left'] = self._angle_3pts(lh, lk, la)
        if rh and rk and ra:
            results['knee_right'] = self._angle_3pts(rh, rk, ra)

        # Hip Angles
        if ls and lh and lk:
            results['hip_left'] = self._angle_3pts(ls, lh, lk)
        if rs and rh and rk:
            results['hip_right'] = self._angle_3pts(rs, rh, rk)

        # Torso tilt (average shoulders-hips vector vs vertical)
        if ls and rs and lh and rh:
            mid_shoulder = {"x": (ls['x']+rs['x'])/2, "y": (ls['y']+rs['y'])/2}
            mid_hip = {"x": (lh['x']+rh['x'])/2, "y": (lh['y']+rh['y'])/2}

            dx = mid_shoulder['x'] - mid_hip['x']
            dy = mid_shoulder['y'] - mid_hip['y']

            vertical_vec = (0, -1)  # reference vertical
            body_vec = (dx, dy)

            dot = vertical_vec[0]*body_vec[0] + vertical_vec[1]*body_vec[1]
            mag1 = math.sqrt(vertical_vec[0]**2 + vertical_vec[1]**2)
            mag2 = math.sqrt(body_vec[0]**2 + body_vec[1]**2)
            if mag2 != 0:
                angle_rad = math.acos(max(min(dot / (mag1*mag2), 1.0), -1.0))
                results['torso_tilt'] = math.degrees(angle_rad)

        return results
