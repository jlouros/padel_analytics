"""
Example implementations of key improvements for the 3D tracking system.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging
from dataclasses import dataclass

# Example 1: Enhanced Homography with 14 Keypoints
class Enhanced14KeypointHomography:
    """
    Enhanced homography calculation supporting 14 keypoints with height validation.
    """
    
    def __init__(self, court_model):
        self.court_model = court_model
        self.logger = logging.getLogger(__name__)
        
    def calculate_homography(self, keypoints_detection) -> np.ndarray:
        """
        Calculate homography matrix with support for 12 or 14 keypoints.
        """
        if len(keypoints_detection) == 12:
            return self._calculate_12_point_homography(keypoints_detection)
        elif len(keypoints_detection) == 14:
            return self._calculate_14_point_homography(keypoints_detection)
        else:
            raise ValueError(f"Unsupported keypoint count: {len(keypoints_detection)}")
    
    def _calculate_14_point_homography(self, keypoints_detection) -> np.ndarray:
        """
        Calculate homography using ground plane keypoints and validate with height keypoints.
        """
        # Use first 12 keypoints (ground plane) for homography calculation
        ground_keypoints = keypoints_detection[:12]
        height_keypoints = keypoints_detection[12:14]  # k13, k14
        
        # Calculate homography using ground plane
        H = self._calculate_12_point_homography(ground_keypoints)
        
        # Validate homography using height keypoints
        validation_score = self._validate_with_height_keypoints(H, height_keypoints)
        
        if validation_score > 50:  # pixels error threshold
            self.logger.warning(f"High validation error: {validation_score:.2f} pixels")
            # Could implement fallback or correction here
        
        return H
    
    def _validate_with_height_keypoints(self, H: np.ndarray, height_keypoints) -> float:
        """
        Validate homography using height keypoints and return average error.
        """
        errors = []
        expected_height_points = self.court_model.keypoints(number_keypoints=14)[12:14]
        
        for actual_kp, expected_kp in zip(height_keypoints, expected_height_points):
            # Project expected 2D position using homography
            projected = self._project_point(expected_kp.xy, H)
            
            # Calculate error
            error = np.linalg.norm(np.array(projected) - np.array(actual_kp.xy))
            errors.append(error)
            
        return np.mean(errors)


# Example 2: Analytical Jacobians for Better Numerical Stability
class AnalyticalEKF:
    """
    Extended Kalman Filter with analytical Jacobians for improved stability.
    """
    
    def __init__(self, court_model, g=9.81):
        self.court_model = court_model
        self.g = g
        self.width = court_model.width
        self.length = court_model.length
        
    def transition_jacobian(self, x: np.ndarray, dt: float = 1/30) -> np.ndarray:
        """
        Analytical Jacobian of the transition function.
        State: [x, y, z, vx, vy, vz, 1]
        """
        F = np.array([
            [1, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0,  0,  dt, -0.5*self.g*dt**2],
            [0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0,  0,  1,  -self.g*dt],
            [0, 0, 0, 0,  0,  0,  1]
        ])
        
        # Handle bouncing conditions
        F = self._apply_bounce_jacobian(F, x)
        
        return F
    
    def observation_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Analytical Jacobian of observation function (3D to 2D projection).
        """
        P = self.court_model.projection_matrix
        xyz1 = np.append(x[:3], 1)  # homogeneous coordinates
        
        # Project to homogeneous image coordinates
        uvw = P @ xyz1
        u, v, w = uvw[0], uvw[1], uvw[2]
        
        # Prevent division by zero
        if abs(w) < 1e-8:
            w = 1e-8 * np.sign(w) if w != 0 else 1e-8
        
        # Analytical derivatives
        H = np.zeros((2, 7))
        
        # du/dx, du/dy, du/dz
        H[0, 0] = (P[0,0]*w - P[2,0]*u) / (w**2)
        H[0, 1] = (P[0,1]*w - P[2,1]*u) / (w**2)
        H[0, 2] = (P[0,2]*w - P[2,2]*u) / (w**2)
        
        # dv/dx, dv/dy, dv/dz
        H[1, 0] = (P[1,0]*w - P[2,0]*v) / (w**2)
        H[1, 1] = (P[1,1]*w - P[2,1]*v) / (w**2)
        H[1, 2] = (P[1,2]*w - P[2,2]*v) / (w**2)
        
        # Velocity components don't affect observation directly
        H[:, 3:6] = 0
        H[:, 6] = 0  # Homogeneous coordinate
        
        return H
    
    def _apply_bounce_jacobian(self, F: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Modify Jacobian to account for bouncing conditions.
        """
        # If ball is at/below ground and moving downward
        if x[2] <= 0 and x[5] <= 0:
            # Reflection in z-direction
            F[2, 5] = -F[2, 5]  # Velocity reflection
            F[5, 5] = -F[5, 5]  # Sign change for velocity
            
        # Handle wall bounces similarly for x and y directions
        if x[0] <= 0 or x[0] >= self.width:
            F[0, 3] = -F[0, 3]
            F[3, 3] = -F[3, 3]
            
        if x[1] <= 0 or x[1] >= self.length:
            F[1, 4] = -F[1, 4]
            F[4, 4] = -F[4, 4]
            
        return F


# Example 3: Robust Tracker with Error Recovery
@dataclass
class TrackingState:
    """Data class for tracking state information."""
    position_3d: np.ndarray
    velocity_3d: np.ndarray
    uncertainty: np.ndarray
    last_detection_frame: int
    confidence: float


class RobustBallTracker:
    """
    Robust ball tracker with comprehensive error handling and recovery.
    """
    
    def __init__(self, court_model, max_missed_detections=15):
        self.court_model = court_model
        self.max_missed_detections = max_missed_detections
        self.logger = logging.getLogger(__name__)
        
        # Tracking state
        self.current_state: Optional[TrackingState] = None
        self.missed_count = 0
        self.frame_count = 0
        
        # Validation thresholds
        self.max_velocity = 50.0  # m/s
        self.max_acceleration = 100.0  # m/s^2
        self.position_bounds = {
            'x': (-1, court_model.width + 1),  # Allow slight extrapolation
            'y': (-1, court_model.length + 1),
            'z': (-0.5, 10.0)  # Allow below ground for robustness
        }
        
    def track_frame(self, detection_2d: Optional[Tuple[float, float]]) -> Optional[TrackingState]:
        """
        Process a single frame of detection data.
        """
        self.frame_count += 1
        
        try:
            # Always predict forward in time
            if self.current_state is not None:
                self._predict_state()
            
            # Process detection if available and valid
            if detection_2d is not None and self._validate_detection(detection_2d):
                self._update_with_detection(detection_2d)
                self.missed_count = 0
            else:
                self.missed_count += 1
                self._handle_missed_detection()
            
            # Validate current state
            if self.current_state is not None:
                if not self._validate_3d_state(self.current_state):
                    self.logger.warning(f"Invalid 3D state at frame {self.frame_count}")
                    self._correct_invalid_state()
            
            return self.current_state
            
        except Exception as e:
            self.logger.error(f"Error in frame {self.frame_count}: {e}")
            return self._fallback_prediction()
    
    def _validate_detection(self, detection: Tuple[float, float]) -> bool:
        """Validate 2D detection quality."""
        x, y = detection
        
        # Basic bounds check
        if x <= 0 or y <= 0:
            return False
        
        # Check against previous detection for reasonableness
        if self.current_state is not None:
            predicted_2d = self.court_model.world2image(self.current_state.position_3d)
            distance = np.linalg.norm(np.array([x, y]) - predicted_2d)
            
            # Reject detections too far from prediction
            if distance > 100:  # pixels
                self.logger.debug(f"Detection rejected: distance {distance:.1f} px")
                return False
        
        return True
    
    def _validate_3d_state(self, state: TrackingState) -> bool:
        """Validate 3D state for physical reasonableness."""
        pos = state.position_3d
        vel = state.velocity_3d
        
        # Position bounds
        if not (self.position_bounds['x'][0] <= pos[0] <= self.position_bounds['x'][1]):
            return False
        if not (self.position_bounds['y'][0] <= pos[1] <= self.position_bounds['y'][1]):
            return False
        if not (self.position_bounds['z'][0] <= pos[2] <= self.position_bounds['z'][1]):
            return False
        
        # Velocity bounds
        speed = np.linalg.norm(vel)
        if speed > self.max_velocity:
            self.logger.warning(f"Speed too high: {speed:.1f} m/s")
            return False
        
        return True
    
    def _handle_missed_detection(self):
        """Handle case where no valid detection is available."""
        if self.missed_count > self.max_missed_detections:
            self.logger.warning("Too many missed detections, resetting tracker")
            self.current_state = None
        elif self.current_state is not None:
            # Increase uncertainty due to missing measurement
            self.current_state.confidence *= 0.95
            self.current_state.uncertainty *= 1.1
    
    def _correct_invalid_state(self):
        """Attempt to correct invalid state estimates."""
        if self.current_state is None:
            return
        
        pos = self.current_state.position_3d
        
        # Clamp position to valid bounds
        pos[0] = np.clip(pos[0], self.position_bounds['x'][0], self.position_bounds['x'][1])
        pos[1] = np.clip(pos[1], self.position_bounds['y'][0], self.position_bounds['y'][1])
        pos[2] = np.clip(pos[2], self.position_bounds['z'][0], self.position_bounds['z'][1])
        
        # Limit velocity if too high
        vel_norm = np.linalg.norm(self.current_state.velocity_3d)
        if vel_norm > self.max_velocity:
            self.current_state.velocity_3d *= (self.max_velocity / vel_norm)
        
        # Reduce confidence after correction
        self.current_state.confidence *= 0.8
        
        self.logger.info("State corrected due to invalid values")
    
    def _fallback_prediction(self) -> Optional[TrackingState]:
        """Provide fallback prediction when main tracking fails."""
        if self.current_state is not None:
            # Simple linear extrapolation as fallback
            dt = 1/30  # Assume 30fps
            fallback_pos = (self.current_state.position_3d + 
                          self.current_state.velocity_3d * dt)
            
            return TrackingState(
                position_3d=fallback_pos,
                velocity_3d=self.current_state.velocity_3d * 0.9,  # Damping
                uncertainty=self.current_state.uncertainty * 2,  # Higher uncertainty
                last_detection_frame=self.current_state.last_detection_frame,
                confidence=0.1  # Low confidence
            )
        
        return None


# Example 4: Performance Optimization with Caching
class OptimizedCourtModel:
    """
    Court model with performance optimizations and caching.
    """
    
    def __init__(self, keypoints, cache_size=1000):
        self.keypoints = keypoints
        self.cache_size = cache_size
        
        # Initialize standard model
        self._initialize_model()
        
        # Performance optimizations
        self.projection_cache = {}
        self.jacobian_cache = {}
        self.batch_size = 32
        
    def world2image_batch(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Efficiently project multiple 3D points to 2D.
        """
        if len(points_3d.shape) == 1:
            points_3d = points_3d.reshape(1, -1)
        
        # Convert to homogeneous coordinates
        points_homogeneous = np.c_[points_3d, np.ones(len(points_3d))]
        
        # Batch projection
        projected = (self.projection_matrix @ points_homogeneous.T).T
        
        # Normalize homogeneous coordinates
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = projected[:, :2] / projected[:, 2:3]
            
        # Handle invalid projections
        invalid_mask = ~np.isfinite(normalized).all(axis=1)
        if invalid_mask.any():
            self.logger.warning(f"Invalid projections: {invalid_mask.sum()}")
            normalized[invalid_mask] = 0
        
        return normalized
    
    def clear_cache(self):
        """Clear performance caches."""
        self.projection_cache.clear()
        self.jacobian_cache.clear()


if __name__ == "__main__":
    # Example usage demonstration
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Example implementations for 3D tracking improvements")
    print("=" * 60)
    
    # These examples show the structure and approach for implementing
    # the key improvements identified in the analysis.
    # They would need to be adapted and integrated into the existing codebase.
    
    print("✅ Enhanced 14-keypoint homography support")
    print("✅ Analytical Jacobians for numerical stability") 
    print("✅ Robust error handling and recovery")
    print("✅ Performance optimizations with caching")
    print("\nSee ANALYSIS_AND_RECOMMENDATIONS.md for complete implementation roadmap.")
