"""
Improved homography calculation for 14 keypoints including height information.
"""

def enhanced_homography_matrix(self, keypoints_detection: Keypoints) -> np.ndarray:
    """
    Calculate homography matrix supporting both 12 and 14 keypoints.
    For 14 keypoints, use ground plane keypoints and validate with height keypoints.
    """
    if len(keypoints_detection) == 12:
        return self.homography_matrix(keypoints_detection)
    
    elif len(keypoints_detection) == 14:
        # Use ground plane keypoints (first 12) for homography
        ground_keypoints = keypoints_detection[:12]
        height_keypoints = keypoints_detection[12:14]  # k13, k14
        
        # Calculate homography using ground plane
        H = self.homography_matrix(ground_keypoints)
        
        # Validate homography using height keypoints
        expected_height_points = self.court_keypoints.keypoints(number_keypoints=14)[12:14]
        
        for i, (height_kp, expected_kp) in enumerate(zip(height_keypoints, expected_height_points)):
            projected = self.project_point(expected_kp.xy, H)
            distance = np.linalg.norm(np.array(projected) - np.array(height_kp.xy))
            
            if distance > 50:  # pixels - threshold for validation
                print(f"Warning: Height keypoint {i+12} projection error: {distance:.2f} pixels")
        
        return H
    
    else:
        raise ValueError(f"Unsupported number of keypoints: {len(keypoints_detection)}")
