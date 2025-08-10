"""
Comprehensive unit tests for the 3D ball tracking system.
"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import Mock, patch
import torch

from trackers.ball_tracker.court_3d_model import Court3DModel, compute_regression_line, find_intersection
from trackers.ball_tracker.kalman3d_tracking import KalmanFilter3DTracking
from trackers.ball_tracker.ekf import ExtendedKalmanFilter
from trackers.keypoints_tracker.keypoints_tracker import Keypoints, Keypoint


class TestCourt3DModelAdvanced:
    """Advanced tests for the 3D court model."""
    
    def test_keypoint_correspondence_completeness(self, court_keypoints):
        """Test that all expected keypoints are mapped correctly."""
        court_model = Court3DModel(keypoints=court_keypoints)
        
        # Check that all critical keypoints are present
        expected_keypoints = [0, 1, 5, 6, 10, 11, 12, 13]
        for kp_id in expected_keypoints:
            assert kp_id in court_model.keypoint_correspondence
            
        # Check that 3D coordinates are reasonable
        for kp_id, xyz in court_model.keypoint_correspondence.items():
            assert len(xyz) == 3
            assert 0 <= xyz[0] <= court_model.width
            assert 0 <= xyz[1] <= court_model.length
            
    def test_projection_matrix_properties(self, court_keypoints):
        """Test mathematical properties of the projection matrix."""
        court_model = Court3DModel(keypoints=court_keypoints)
        P = court_model.projection_matrix
        
        # Should be 3x4
        assert P.shape == (3, 4)
        
        # Should be full rank (rank 3)
        assert np.linalg.matrix_rank(P) == 3
        
        # Test projection consistency
        for kp_id, world_coords in court_model.keypoint_correspondence.items():
            projected = court_model.world2image(world_coords)
            expected = court_model.keypoints.keypoints_by_id[kp_id].xy
            
            # Allow for reasonable projection error
            error = np.linalg.norm(projected - np.array(expected))
            assert error < 50, f"Keypoint {kp_id} projection error too large: {error}"
            
    def test_vanishing_point_geometry(self):
        """Test vanishing point calculation with known geometry."""
        # Create keypoints representing a rectangular court viewed in perspective
        keypoints = Keypoints([
            Keypoint(id=0, xy=(100, 400)),   # k1 - bottom left
            Keypoint(id=1, xy=(500, 400)),   # k2 - bottom right  
            Keypoint(id=5, xy=(150, 300)),   # k6 - middle left
            Keypoint(id=6, xy=(450, 300)),   # k7 - middle right
            Keypoint(id=10, xy=(200, 200)),  # k11 - top left
            Keypoint(id=11, xy=(400, 200)),  # k12 - top right
            Keypoint(id=12, xy=(120, 380)),  # k13 - bottom left height
            Keypoint(id=13, xy=(480, 380)),  # k14 - bottom right height
        ])
        
        court_model = Court3DModel(keypoints=keypoints)
        
        # Vanishing points should be finite for well-conditioned points
        assert np.isfinite(court_model.depth_vanishing_point[0])
        assert np.isfinite(court_model.depth_vanishing_point[1])
        assert np.isfinite(court_model.height_vanishing_point[0])
        assert np.isfinite(court_model.height_vanishing_point[1])
        
    def test_world2image_consistency(self, court_keypoints):
        """Test world-to-image projection consistency."""
        court_model = Court3DModel(keypoints=court_keypoints)
        
        # Test with various 3D points
        test_points = [
            [0, 0, 0],                    # court corner
            [court_model.width/2, court_model.length/2, 0],  # center
            [court_model.width, court_model.length, 0],      # opposite corner
            [court_model.width/2, court_model.length/2, 2],  # center elevated
        ]
        
        for point_3d in test_points:
            point_2d = court_model.world2image(point_3d)
            assert len(point_2d) == 2
            assert np.isfinite(point_2d[0])
            assert np.isfinite(point_2d[1])
            
    def test_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal valid keypoints
        minimal_keypoints = Keypoints([
            Keypoint(id=i, xy=(i*10, i*10))
            for i in [0, 1, 5, 6, 10, 11, 12, 13]
        ])
        
        # Should not raise exception
        court_model = Court3DModel(keypoints=minimal_keypoints)
        assert court_model.projection_matrix is not None


class TestKalmanFilter3DTrackingAdvanced:
    """Advanced tests for 3D Kalman filter tracking."""
    
    def test_state_initialization(self, court_model):
        """Test various initialization scenarios."""
        # Test default initialization
        filter_default = KalmanFilter3DTracking(court_model=court_model)
        assert len(filter_default.x) == 7
        assert filter_default.x[6] == 1  # homogeneous coordinate
        
        # Test custom initialization
        custom_x0 = [1, 2, 3, 0.5, 1.0, 1.5, 1]
        filter_custom = KalmanFilter3DTracking(court_model=court_model, x0=custom_x0)
        npt.assert_array_equal(filter_custom.x, custom_x0)
        
    def test_physics_simulation(self, court_model):
        """Test physics simulation accuracy."""
        # Initialize with known state
        x0 = [5, 10, 2, 0, 0, 5, 1]  # ball going up
        filter = KalmanFilter3DTracking(court_model=court_model, x0=x0)
        
        dt = 1/30
        
        # Simulate several steps
        states = []
        for _ in range(10):
            filter.predict()
            states.append(filter.x.copy())
            
        states = np.array(states)
        
        # Check that gravity affects vertical motion
        z_positions = states[:, 2]
        z_velocities = states[:, 5]
        
        # Z should first increase then decrease (parabolic trajectory)
        assert z_positions[0] < z_positions[1]  # initial upward motion
        
        # Velocity should decrease due to gravity
        assert z_velocities[-1] < z_velocities[0]
        
    def test_bouncing_physics(self, court_model):
        """Test ball bouncing behavior."""
        # Initialize ball just above ground with downward velocity
        x0 = [5, 10, 0.1, 0, 0, -2, 1]
        filter = KalmanFilter3DTracking(court_model=court_model, x0=x0)
        
        # Predict several steps
        for _ in range(5):
            filter.predict()
            
        # Ball should have bounced (positive z and upward velocity)
        assert filter.x[2] > 0  # above ground
        assert filter.x[5] > 0  # upward velocity after bounce
        
    def test_observation_function(self, court_model):
        """Test observation function accuracy."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        # Test with known 3D position
        test_state = [5, 10, 1, 0, 0, 0, 1]
        filter.x = test_state
        
        observation = filter.observation_function(test_state)
        assert len(observation) == 2
        assert np.isfinite(observation[0])
        assert np.isfinite(observation[1])
        
        # Observation should be consistent with court model
        expected = court_model.world2image(test_state[:3])
        npt.assert_almost_equal(observation, expected, decimal=6)
        
    def test_update_step(self, court_model):
        """Test filter update with measurements."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        initial_uncertainty = np.trace(filter.P)
        
        # Provide a measurement
        measurement = [320, 240]  # image coordinates
        filter.update(measurement)
        
        # Uncertainty should decrease after measurement
        final_uncertainty = np.trace(filter.P)
        assert final_uncertainty < initial_uncertainty
        
    def test_invalid_measurements(self, court_model):
        """Test handling of invalid measurements.""" 
        filter = KalmanFilter3DTracking(court_model=court_model)
        initial_state = filter.x.copy()
        
        # Test with (0, 0) measurement (invalid detection)
        result = filter.update((0, 0))
        assert result is None  # Should skip update
        
        # State should be unchanged
        npt.assert_array_equal(filter.x, initial_state)
        
    def test_initial_state_estimation(self, court_model):
        """Test initial state estimation from observations."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        # Create synthetic observations representing a parabolic trajectory
        observations = []
        for t in range(20):
            time = t * (1/30)
            # Simulate ball trajectory
            x_3d = 5 + 2*time
            y_3d = 10 + 3*time  
            z_3d = max(0, 2 + 1*time - 0.5*9.81*time**2)
            
            # Project to 2D
            observation_2d = court_model.world2image([x_3d, y_3d, z_3d])
            observations.append([observation_2d[0], observation_2d[1], time])
            
        estimated_state = filter.estimate_initial_state(observations)
        
        assert len(estimated_state) == 7
        # Position should be reasonable
        assert 0 <= estimated_state[0] <= court_model.width
        assert 0 <= estimated_state[1] <= court_model.length
        assert estimated_state[2] >= 0  # height non-negative


class TestBallTrackerIntegration:
    """Integration tests for ball tracker with 3D filtering."""
    
    @pytest.fixture
    def mock_ball_tracker(self, court_model):
        """Create a mock ball tracker for testing."""
        from trackers.ball_tracker.ball_tracker import BallTracker
        
        # Mock the tracker to avoid loading actual models
        with patch.object(BallTracker, '__init__', return_value=None):
            tracker = BallTracker.__new__(BallTracker)
            tracker.kalman_tracker = KalmanFilter3DTracking(court_model=court_model)
            tracker.court_model = court_model
            return tracker
            
    def test_3d_tracking_integration(self, mock_ball_tracker):
        """Test integration of 2D detection with 3D tracking."""
        tracker = mock_ball_tracker
        
        # Simulate detection sequence
        detections_2d = [
            (320, 240),
            (325, 235), 
            (330, 230),
            (335, 225),
        ]
        
        results = []
        for detection in detections_2d:
            tracker.kalman_tracker.predict()
            tracker.kalman_tracker.update(detection)
            state = tracker.kalman_tracker.get_state()
            results.append(state[:3])  # 3D position
            
        # Check that 3D positions change smoothly
        results = np.array(results)
        
        # Position should change smoothly (no large jumps)
        for i in range(1, len(results)):
            distance = np.linalg.norm(results[i] - results[i-1])
            assert distance < 2.0  # reasonable movement between frames
            
    def test_missing_detection_handling(self, mock_ball_tracker):
        """Test handling of missing detections."""
        tracker = mock_ball_tracker
        
        # Mix of valid and invalid detections
        detections = [
            (320, 240),
            (0, 0),      # invalid
            (330, 230),
            (0, 0),      # invalid
            (340, 220),
        ]
        
        results = []
        for detection in detections:
            tracker.kalman_tracker.predict()
            if detection != (0, 0):
                tracker.kalman_tracker.update(detection)
            state = tracker.kalman_tracker.get_state()
            results.append(state[:3])
            
        # Should still produce reasonable trajectory despite missing detections
        results = np.array(results)
        assert len(results) == len(detections)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_degenerate_keypoints(self):
        """Test handling of degenerate keypoint configurations."""
        # All keypoints in a line (degenerate case)
        degenerate_keypoints = Keypoints([
            Keypoint(id=i, xy=(i*10, 100)) 
            for i in range(14)
        ])
        
        # Should handle gracefully (might raise exception or return None)
        try:
            court_model = Court3DModel(keypoints=degenerate_keypoints)
            # If no exception, check that projection matrix has low rank
            if court_model.projection_matrix is not None:
                rank = np.linalg.matrix_rank(court_model.projection_matrix)
                assert rank < 3  # Should be rank deficient
        except (np.linalg.LinAlgError, ValueError):
            # Expected for degenerate case
            pass
            
    def test_extreme_parameter_values(self, court_model):
        """Test filter with extreme parameter values."""
        # Very high process noise
        filter_noisy = KalmanFilter3DTracking(
            court_model=court_model, 
            q=1000  # very high process noise
        )
        
        # Should still initialize without error
        assert filter_noisy.x is not None
        
        # Very low measurement noise
        filter_precise = KalmanFilter3DTracking(
            court_model=court_model,
            r=0.001  # very low measurement noise
        )
        
        # Should still work
        filter_precise.predict()
        filter_precise.update([320, 240])
        
    def test_numerical_stability(self, court_model):
        """Test numerical stability over many iterations."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        # Run many prediction steps
        for i in range(1000):
            filter.predict()
            
            # Add occasional measurements
            if i % 10 == 0:
                filter.update([320 + np.sin(i/10)*10, 240 + np.cos(i/10)*10])
                
        # State should remain finite
        assert np.all(np.isfinite(filter.x))
        assert np.all(np.isfinite(filter.P))
        
        # Covariance should remain positive definite
        eigenvals = np.linalg.eigvals(filter.P)
        assert np.all(eigenvals > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
