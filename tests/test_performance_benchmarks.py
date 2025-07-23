"""
Performance and benchmarking tests for the 3D tracking system.
"""

import pytest
import time
import numpy as np
import psutil
import threading
from unittest.mock import Mock

from trackers.ball_tracker.court_3d_model import Court3DModel
from trackers.ball_tracker.kalman3d_tracking import KalmanFilter3DTracking


class TestPerformanceAndBenchmarks:
    """Performance tests and benchmarks for the tracking system."""
    
    def test_court_model_initialization_performance(self, court_keypoints):
        """Test performance of court model initialization."""
        start_time = time.time()
        
        for _ in range(100):
            court_model = Court3DModel(keypoints=court_keypoints)
            
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        # Should initialize quickly (less than 10ms on average)
        assert avg_time < 0.01, f"Court model initialization too slow: {avg_time:.4f}s"
        
    def test_kalman_filter_prediction_performance(self, court_model):
        """Test performance of Kalman filter prediction steps."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        start_time = time.time()
        
        for _ in range(1000):
            filter.predict()
            
        end_time = time.time()
        avg_time = (end_time - start_time) / 1000
        
        # Should predict quickly (less than 1ms per prediction)
        assert avg_time < 0.001, f"Kalman prediction too slow: {avg_time:.6f}s"
        
    def test_kalman_filter_update_performance(self, court_model):
        """Test performance of Kalman filter update steps."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        start_time = time.time()
        
        for i in range(1000):
            measurement = [320 + np.sin(i/10), 240 + np.cos(i/10)]
            filter.update(measurement)
            
        end_time = time.time()
        avg_time = (end_time - start_time) / 1000
        
        # Should update quickly (less than 2ms per update due to Jacobian computation)
        assert avg_time < 0.002, f"Kalman update too slow: {avg_time:.6f}s"
        
    def test_projection_performance(self, court_model):
        """Test performance of world-to-image projection."""
        # Generate random 3D points
        points_3d = np.random.rand(1000, 3)
        points_3d[:, 0] *= court_model.width
        points_3d[:, 1] *= court_model.length
        points_3d[:, 2] *= 5  # height up to 5m
        
        start_time = time.time()
        
        for point in points_3d:
            court_model.world2image(point)
            
        end_time = time.time()
        avg_time = (end_time - start_time) / 1000
        
        # Should project quickly (less than 0.1ms per projection)
        assert avg_time < 0.0001, f"Projection too slow: {avg_time:.6f}s"
        
    def test_memory_usage(self, court_model):
        """Test memory usage of the tracking system."""
        import gc
        
        # Measure initial memory
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple filters and run tracking
        filters = []
        for i in range(10):
            filter = KalmanFilter3DTracking(court_model=court_model)
            
            # Run some tracking steps
            for j in range(100):
                filter.predict()
                if j % 5 == 0:
                    filter.update([320 + i, 240 + j])
                    
            filters.append(filter)
            
        # Measure final memory
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB for 10 filters)
        assert memory_increase < 50, f"Excessive memory usage: {memory_increase:.2f}MB"
        
    def test_concurrent_tracking(self, court_model):
        """Test thread safety and concurrent tracking performance."""
        num_threads = 4
        num_iterations = 100
        
        def track_ball(thread_id, results):
            filter = KalmanFilter3DTracking(court_model=court_model)
            
            start_time = time.time()
            
            for i in range(num_iterations):
                filter.predict()
                measurement = [320 + thread_id*10, 240 + i]
                filter.update(measurement)
                
            end_time = time.time()
            results[thread_id] = end_time - start_time
            
        # Run concurrent tracking
        threads = []
        results = {}
        
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=track_ball, args=(i, results))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        total_time = time.time() - start_time
        
        # All threads should complete
        assert len(results) == num_threads
        
        # Average per-thread time should be reasonable
        avg_thread_time = sum(results.values()) / num_threads
        assert avg_thread_time < 1.0, f"Concurrent tracking too slow: {avg_thread_time:.3f}s"
        
    @pytest.mark.parametrize("noise_level", [0.1, 1.0, 10.0, 100.0])
    def test_performance_vs_noise(self, court_model, noise_level):
        """Test how performance scales with measurement noise."""
        filter = KalmanFilter3DTracking(court_model=court_model, r=noise_level)
        
        start_time = time.time()
        
        for i in range(500):
            filter.predict()
            # Add noise to measurement
            clean_measurement = [320, 240]
            noisy_measurement = [
                clean_measurement[0] + np.random.normal(0, noise_level),
                clean_measurement[1] + np.random.normal(0, noise_level)
            ]
            filter.update(noisy_measurement)
            
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance should not degrade significantly with noise
        assert total_time < 2.0, f"Performance degraded with noise {noise_level}: {total_time:.3f}s"


class TestScalabilityAndStressTests:
    """Scalability and stress tests."""
    
    def test_long_sequence_tracking(self, court_model):
        """Test tracking over very long sequences."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        # Simulate tracking for 10 minutes at 30fps
        num_frames = 10 * 60 * 30  # 18,000 frames
        
        start_time = time.time()
        
        for i in range(0, num_frames, 100):  # Sample every 100th frame for speed
            filter.predict()
            
            # Simulate reasonable ball movement
            t = i / 30.0  # time in seconds
            measurement = [
                320 + 50 * np.sin(t * 0.1),
                240 + 30 * np.cos(t * 0.1)
            ]
            filter.update(measurement)
            
            # Verify state remains stable
            assert np.all(np.isfinite(filter.x))
            assert np.all(np.isfinite(filter.P))
            
        end_time = time.time()
        
        # Should handle long sequences efficiently
        assert end_time - start_time < 10.0, "Long sequence tracking too slow"
        
    def test_rapid_state_changes(self, court_model):
        """Test handling of rapid ball state changes."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        # Simulate rapid direction changes (like ball hits)
        measurements = []
        for i in range(1000):
            if i % 50 == 0:  # Direction change every 50 frames
                direction = 1 if (i // 50) % 2 == 0 else -1
                
            x = 320 + direction * (i % 50) * 2
            y = 240 + np.sin(i * 0.1) * 20
            measurements.append([x, y])
            
        start_time = time.time()
        
        for measurement in measurements:
            filter.predict()
            filter.update(measurement)
            
        end_time = time.time()
        
        # Should handle rapid changes without performance degradation
        assert end_time - start_time < 2.0, "Rapid state changes handling too slow"
        
        # Final state should be reasonable
        assert np.all(np.isfinite(filter.x))
        
    def test_extreme_measurements(self, court_model):
        """Test handling of extreme/outlier measurements."""
        filter = KalmanFilter3DTracking(court_model=court_model)
        
        # Mix normal and extreme measurements
        measurements = [
            [320, 240],    # normal
            [320, 240],    # normal
            [1000, 1000],  # extreme outlier
            [325, 245],    # normal
            [-100, -100],  # extreme outlier
            [330, 250],    # normal
        ]
        
        states = []
        for measurement in measurements:
            filter.predict()
            filter.update(measurement)
            states.append(filter.x.copy())
            
        # Filter should remain stable despite outliers
        for state in states:
            assert np.all(np.isfinite(state))
            # 3D position should remain within reasonable bounds
            assert 0 <= state[0] <= court_model.width * 2  # allow some extrapolation
            assert 0 <= state[1] <= court_model.length * 2
            assert -1 <= state[2] <= 10  # reasonable height bounds


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
