"""
Integration tests for the complete 3D tracking pipeline.
"""

import pytest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
import cv2

from trackers.ball_tracker.ball_tracker import BallTracker, Ball
from trackers.ball_tracker.court_3d_model import Court3DModel
from trackers.ball_tracker.kalman3d_tracking import KalmanFilter3DTracking
from trackers.keypoints_tracker.keypoints_tracker import Keypoints, Keypoint
from trackers.runner import TrackingRunner
from analytics.projected_court import ProjectedCourt


class TestIntegrationPipeline:
    """Integration tests for the complete tracking pipeline."""
    
    @pytest.fixture
    def mock_video_frames(self):
        """Generate mock video frames for testing."""
        frames = []
        for i in range(30):  # 1 second at 30fps
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        return frames
        
    @pytest.fixture
    def mock_video_info(self):
        """Mock video info object."""
        import supervision as sv
        video_info = Mock(spec=sv.VideoInfo)
        video_info.fps = 30
        video_info.width = 640
        video_info.height = 480
        video_info.total_frames = 30
        video_info.resolution_wh = (640, 480)
        return video_info
        
    def test_end_to_end_ball_tracking(self, court_keypoints, mock_video_frames, mock_video_info):
        """Test complete end-to-end ball tracking pipeline."""
        # Setup court model
        court_model = Court3DModel(keypoints=court_keypoints)
        
        # Mock ball tracker components
        with patch('trackers.ball_tracker.ball_tracker.torch.load') as mock_load, \
             patch('trackers.ball_tracker.ball_tracker.get_model') as mock_get_model:
            
            # Configure mocks
            mock_load.return_value = {
                'param_dict': {'seq_len': 8, 'bg_mode': 'concat'},
                'model': {}
            }
            
            mock_tracknet = Mock()
            mock_tracknet.return_value = torch.zeros((1, 1, 288, 512))
            mock_get_model.return_value = mock_tracknet
            
            # Create ball tracker
            ball_tracker = BallTracker(
                tracking_model_path="dummy_path.pt",
                inpainting_model_path="dummy_inpaint.pt",
                batch_size=1,
                median_max_sample_num=10,
                court_model=court_model
            )
            
            ball_tracker.video_info_post_init(mock_video_info)
            
            # Test prediction on frames
            ball_detections = ball_tracker.predict_frames(
                frame_generator=iter(mock_video_frames),
                total_frames=len(mock_video_frames)
            )
            
            # Verify results
            assert len(ball_detections) == len(mock_video_frames)
            
            for detection in ball_detections:
                assert isinstance(detection, Ball)
                assert detection.frame >= 0
                assert len(detection.xy) == 2
                assert detection.visibility in [0, 1]
                
                # If 3D tracking is enabled, should have xyz coordinates
                if hasattr(detection, 'xyz') and detection.xyz:
                    assert len(detection.xyz) == 3
                    
    def test_projected_court_integration(self, court_keypoints):
        """Test integration with projected court visualization."""
        # Create mock ball with 3D coordinates
        ball_detection = Ball(
            frame=0,
            xy=(320, 240),
            xyz=(5, 10, 1),
            visibility=1
        )
        
        court_model = Court3DModel(keypoints=court_keypoints)
        projected_court = ProjectedCourt()
        
        # Mock homography matrix
        H = np.eye(3)
        
        # Test ball projection
        projected_ball = projected_court.project_ball(
            ball_detection=ball_detection,
            homography_matrix=H,
            court_model=court_model
        )
        
        assert hasattr(projected_ball, 'projection')
        assert len(projected_ball.projection) == 2
        
    def test_tracking_runner_integration(self, court_keypoints, mock_video_info):
        """Test integration with tracking runner."""
        court_model = Court3DModel(keypoints=court_keypoints)
        
        # Mock trackers
        mock_ball_tracker = Mock()
        mock_ball_tracker.results = Mock()
        mock_ball_tracker.results.predictions = [
            Ball(frame=i, xy=(320+i, 240+i), xyz=(5+i*0.1, 10+i*0.1, 1), visibility=1)
            for i in range(30)
        ]
        mock_ball_tracker.object.return_value = Ball
        mock_ball_tracker.draw_kwargs.return_value = {}
        mock_ball_tracker.__str__ = lambda: "ball_tracker"
        
        mock_keypoints_tracker = Mock()
        mock_keypoints_tracker.results = Mock()
        mock_keypoints_tracker.results.predictions = [court_keypoints] * 30
        mock_keypoints_tracker.object.return_value = Keypoints
        mock_keypoints_tracker.draw_kwargs.return_value = {}
        mock_keypoints_tracker.__str__ = lambda: "keypoints_tracker"
        
        # Create runner
        with patch('trackers.runner.sv.VideoInfo.from_video_path', return_value=mock_video_info):
            runner = TrackingRunner(
                trackers=[mock_ball_tracker, mock_keypoints_tracker],
                video_path="dummy.mp4",
                inference_path="output.mp4",
                court_model=court_model,
                start=0,
                end=30,
                collect_data=False
            )
            
        # Verify court model is integrated
        assert runner.court_model == court_model
        
    def test_configuration_consistency(self):
        """Test that configuration changes are consistent."""
        from config import (
            COLLECT_DATA, BALL_TRACKER_LOAD_PATH, PLAYERS_TRACKER_LOAD_PATH,
            PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH, FIXED_COURT_KEYPOINTS_SAVE_PATH
        )
        
        # Verify configuration changes match branch expectations
        assert COLLECT_DATA == False
        assert BALL_TRACKER_LOAD_PATH is None
        assert PLAYERS_TRACKER_LOAD_PATH is None  
        assert PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH is None
        assert FIXED_COURT_KEYPOINTS_SAVE_PATH == "./cache/fixed_keypoints_detection.json"
        
    def test_requirements_consistency(self):
        """Test that requirements are properly specified."""
        import requirements
        
        # Read requirements.txt
        with open('/home/jlouros/code/padel_analytics/requirements.txt', 'r') as f:
            req_lines = f.readlines()
            
        # Check for key new dependencies
        req_text = ''.join(req_lines)
        assert 'numdifftools' in req_text
        assert 'scipy' in req_text
        
        # Check version pinning is used
        version_pinned = [line for line in req_lines if '~=' in line or '==' in line]
        assert len(version_pinned) > 15  # Most packages should be version pinned


class TestErrorRecoveryAndFailover:
    """Test error recovery and failover scenarios."""
    
    def test_kalman_filter_numerical_instability_recovery(self, court_model):
        """Test recovery from numerical instability in Kalman filter."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        # Force numerical instability by corrupting covariance matrix
        filter.P = np.array([[np.inf, 0], [0, np.inf]])  # Invalid covariance
        
        # Filter should detect and recover from this
        try:
            filter.predict()
            filter.update([320, 240])
            
            # If it doesn't crash, check that state is reasonable
            assert np.all(np.isfinite(filter.x))
            
        except Exception as e:
            # If it does crash, that's expected for this extreme case
            assert "numerical" in str(e).lower() or "singular" in str(e).lower()
            
    def test_court_model_degenerate_keypoints_handling(self):
        """Test handling of degenerate keypoint configurations."""
        # Create degenerate keypoints (all collinear)
        degenerate_keypoints = Keypoints([
            Keypoint(id=0, xy=(0, 0)),
            Keypoint(id=1, xy=(100, 0)), 
            Keypoint(id=5, xy=(200, 0)),
            Keypoint(id=6, xy=(300, 0)),
            Keypoint(id=10, xy=(400, 0)),
            Keypoint(id=11, xy=(500, 0)),
            Keypoint(id=12, xy=(50, 0)),
            Keypoint(id=13, xy=(150, 0)),
        ])
        
        # Should handle gracefully
        try:
            court_model = Court3DModel(keypoints=degenerate_keypoints)
            # If successful, projection matrix should be rank-deficient
            rank = np.linalg.matrix_rank(court_model.projection_matrix)
            assert rank < 3
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # Expected for truly degenerate cases
            assert "singular" in str(e).lower() or "rank" in str(e).lower()
            
    def test_missing_detection_recovery(self, court_model):
        """Test recovery from extended periods of missing detections."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        # Provide initial detection
        filter.update([320, 240])
        
        # Simulate extended period of missing detections
        for _ in range(100):  # 100 frames without detection
            filter.predict()
            filter.update((0, 0))  # Invalid detection
            
        # State should remain finite even without measurements
        assert np.all(np.isfinite(filter.x))
        assert np.all(np.isfinite(filter.P))
        
        # Uncertainty should have increased
        final_uncertainty = np.trace(filter.P)
        assert final_uncertainty > 10  # Should be quite uncertain after 100 missing frames
        
    def test_extreme_measurement_outlier_rejection(self, court_model):
        """Test rejection of extreme measurement outliers."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        # Establish normal tracking
        normal_measurements = [[320, 240], [322, 242], [324, 244]]
        for measurement in normal_measurements:
            filter.predict()
            filter.update(measurement)
            
        state_before_outlier = filter.x.copy()
        
        # Introduce extreme outlier
        filter.predict()
        filter.update([10000, 10000])  # Extreme outlier
        
        state_after_outlier = filter.x.copy()
        
        # State should not change dramatically due to outlier
        position_change = np.linalg.norm(state_after_outlier[:3] - state_before_outlier[:3])
        assert position_change < 5.0  # Should not jump more than 5 meters due to outlier


class TestDataFlowAndConsistency:
    """Test data flow and consistency throughout the pipeline."""
    
    def test_coordinate_system_consistency(self, court_keypoints):
        """Test that coordinate systems are consistent throughout pipeline."""
        court_model = Court3DModel(keypoints=court_keypoints)
        
        # Test round-trip consistency: 3D -> 2D -> validation
        test_3d_points = [
            [0, 0, 0],                    # corner
            [court_model.width, 0, 0],   # opposite corner
            [court_model.width/2, court_model.length/2, 1]  # center elevated
        ]
        
        for point_3d in test_3d_points:
            # Project to 2D
            point_2d = court_model.world2image(point_3d)
            
            # Verify 2D point is reasonable (within image bounds approximately)
            assert 0 <= point_2d[0] <= 2000  # reasonable image width range
            assert 0 <= point_2d[1] <= 2000  # reasonable image height range
            
    def test_timestamp_consistency(self):
        """Test that timestamps are consistent across different components."""
        # Create ball detections with frame numbers
        ball_detections = [
            Ball(frame=i, xy=(320+i, 240+i), visibility=1)
            for i in range(10)
        ]
        
        # Verify frame numbers are sequential
        for i, detection in enumerate(ball_detections):
            assert detection.frame == i
            
    def test_state_vector_consistency(self, court_model):
        """Test consistency of state vector representation."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        # State should always be 7-dimensional
        assert len(filter.x) == 7
        
        # After prediction and update, should remain 7-dimensional
        filter.predict()
        filter.update([320, 240])
        assert len(filter.x) == 7
        
        # Homogeneous coordinate should remain 1
        assert filter.x[6] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
