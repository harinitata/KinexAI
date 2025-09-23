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
        """
        L_HIP, R_HIP = 23, 24
        L_KNEE, R_KNEE = 25, 26
        L_ANKLE, R_ANKLE = 27, 28
        L_SHOULDER, R_SHOULDER = 11, 12

        def kp(idx):
            return {"x": keypoints[idx]['x'], "y": keypoints[idx]['y']} if keypoints[idx]['visibility'] > 0.2 else None

        lh, rh = kp(L_HIP), kp(R_HIP)
        lk, rk = kp(L_KNEE), kp(R_KNEE)
        la, ra = kp(L_ANKLE), kp(R_ANKLE)
        ls, rs = kp(L_SHOULDER), kp(R_SHOULDER)

        results = {}
        if lh and lk and la: results['knee_left'] = self._angle_3pts(lh, lk, la)
        if rh and rk and ra: results['knee_right'] = self._angle_3pts(rh, rk, ra)
        if ls and lh and lk: results['hip_left'] = self._angle_3pts(ls, lh, lk)
        if rs and rh and rk: results['hip_right'] = self._angle_3pts(rs, rh, rk)
        
        if ls and rs and lh and rh:
            mid_shoulder = {"x": (ls['x']+rs['x'])/2, "y": (ls['y']+rs['y'])/2}
            mid_hip = {"x": (lh['x']+rh['x'])/2, "y": (lh['y']+rh['y'])/2}
            dx = mid_shoulder['x'] - mid_hip['x']
            dy = mid_shoulder['y'] - mid_hip['y']
            body_vec = (dx, dy)
            vertical_vec = (0, -1)
            dot = vertical_vec[0]*body_vec[0] + vertical_vec[1]*body_vec[1]
            mag1 = math.sqrt(vertical_vec[0]**2 + vertical_vec[1]**2)
            mag2 = math.sqrt(body_vec[0]**2 + body_vec[1]**2)
            if mag2 != 0:
                angle_rad = math.acos(max(min(dot / (mag1*mag2), 1.0), -1.0))
                results['torso_tilt'] = math.degrees(angle_rad)
        
        return results

    def compute_center_of_gravity(self, keypoints):
        """
        Estimates the 2D Center of Gravity (CoG) using a weighted average of keypoints.
        """
        segment_weights = { "hips": 25, "shoulders": 25, "knees": 15, "ankles": 5 }
        L_HIP, R_HIP = 23, 24
        L_SHOULDER, R_SHOULDER = 11, 12
        L_KNEE, R_KNEE = 25, 26
        L_ANKLE, R_ANKLE = 27, 28

        def get_midpoint(kp_left_idx, kp_right_idx):
            if keypoints[kp_left_idx]['visibility'] > 0.5 and keypoints[kp_right_idx]['visibility'] > 0.5:
                return {"x": (keypoints[kp_left_idx]['x'] + keypoints[kp_right_idx]['x']) / 2,
                        "y": (keypoints[kp_left_idx]['y'] + keypoints[kp_right_idx]['y']) / 2}
            return None

        mid_hip = get_midpoint(L_HIP, R_HIP)
        mid_shoulder = get_midpoint(L_SHOULDER, R_SHOULDER)
        mid_knee = get_midpoint(L_KNEE, R_KNEE)
        mid_ankle = get_midpoint(L_ANKLE, R_ANKLE)
        
        points = {"hips": mid_hip, "shoulders": mid_shoulder, "knees": mid_knee, "ankles": mid_ankle}
        weighted_x_sum, weighted_y_sum, total_weight = 0, 0, 0

        for part, point in points.items():
            if point:
                weight = segment_weights[part]
                weighted_x_sum += point['x'] * weight
                weighted_y_sum += point['y'] * weight
                total_weight += weight

        if total_weight == 0:
            return None, None
        
        cog = {"x": weighted_x_sum / total_weight, "y": weighted_y_sum / total_weight}
        return cog, mid_ankle

    def print_joint_visibility(self, keypoints):
        """Prints the visibility score of key joints for debugging."""
        L_HIP, R_HIP = 23, 24
        L_KNEE, R_KNEE = 25, 26
        L_ANKLE, R_ANKLE = 27, 28
        L_SHOULDER, R_SHOULDER = 11, 12

        print("\n--- Joint Visibility Scores ---")
        try:
            print(f"  Left Shoulder:  {keypoints[L_SHOULDER]['visibility']:.2f}")
            print(f"  Left Hip:       {keypoints[L_HIP]['visibility']:.2f}")
            print(f"  Left Knee:      {keypoints[L_KNEE]['visibility']:.2f}")
            print(f"  Left Ankle:     {keypoints[L_ANKLE]['visibility']:.2f}") # <-- Fixed L_ANLE typo here
            print("-----------------------------")
        except IndexError:
            print("  Could not read all joint visibilities (keypoints list might be short).")