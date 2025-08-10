# 🧪 Testing Guide for Padel Analytics

This comprehensive testing guide helps you validate the Padel Analytics system across different scenarios, from basic functionality to edge cases. Whether you're a developer or user, this guide ensures your analysis results are reliable and accurate.

## 🎯 What You'll Learn

- How to systematically test the entire system
- Different testing scenarios and their importance
- How to validate results and detect issues
- Performance testing and optimization
- Creating your own test cases

## 📋 Prerequisites

- **Completed setup**: Working installation following the Beginner's Guide
- **Sample videos**: Access to different types of padel videos
- **Basic understanding**: Familiarity with the analysis process
- **Validation skills**: Ability to assess whether results look correct

---

## 🚀 Quick Verification Test

Before running extensive tests, verify basic functionality:

### 30-Second System Check

```bash
# 1. Activate environment
source padel_analytics_env/bin/activate

# 2. Quick dependency check
python -c "import cv2, ultralytics, supervision, streamlit; print('✅ All packages imported successfully')"

# 3. Model weight verification
python -c "
import os
from config import *
weights = [BALL_TRACKER_MODEL, PLAYERS_TRACKER_MODEL, KEYPOINTS_TRACKER_MODEL, PLAYERS_KEYPOINTS_TRACKER_MODEL]
for w in weights:
    print(f'✅ {w}' if os.path.exists(w) else f'❌ {w} NOT FOUND')
"

# 4. Test with sample video
python main.py  # Should open keypoint selection window
```

**Expected outcome**: Keypoint selection window appears, no error messages in terminal.

---

## 🎬 Video-Based Testing Scenarios

### Test 1: Standard Rally Analysis

**Purpose**: Verify core functionality with ideal conditions

**Test Video Requirements**:

- **Duration**: 30-60 seconds
- **Resolution**: 1080p or higher
- **Camera angle**: Side view of court
- **Lighting**: Good, even lighting
- **Content**: Clear rally with ball and player movement

**Test Steps**:

1. **Setup**:

   ```python
   # Update config.py
   INPUT_VIDEO_PATH = "examples/videos/rally.mp4"
   OUTPUT_DIRECTORY = "test_outputs/standard_rally/"
   ```

2. **Run Analysis**:

   ```bash
   python main.py
   ```

3. **Select Keypoints**: Choose 12 court keypoints carefully

4. **Validate Results**:

   ```markdown
   ✅ Validation Checklist:
   - [ ] Ball is tracked consistently throughout rally
   - [ ] Both players are detected and tracked
   - [ ] No major tracking jumps or loss
   - [ ] Court projection looks accurate
   - [ ] Speed calculations seem reasonable (5-30 km/h for ball)
   - [ ] Heatmaps show logical player positioning
   ```

**Expected Performance**:

- **Processing time**: 2-5 minutes for 30-second video
- **Ball detection accuracy**: >90% of frames
- **Player detection accuracy**: >95% of frames
- **No system crashes or memory errors**

### Test 2: Different Camera Angles

**Purpose**: Test system robustness with various perspectives

**Test Variations**:

#### Test 2A: High Camera Angle

```python
# Expected challenges:
- Players may appear smaller
- Court lines might be more visible
- Different perspective transformation needed

# Validation points:
- Court keypoint selection accuracy
- Player detection at smaller sizes
- Geometric calculations correctness
```

#### Test 2B: Low Camera Angle

```python
# Expected challenges:
- Court lines might be partially obscured
- Players may overlap more frequently
- Net might obstruct view

# Validation points:
- Handling of occlusions
- Player ID consistency when crossing
- Ball tracking through net area
```

#### Test 2C: Angled Side View

```python
# Expected challenges:
- Perspective distortion
- Asymmetric court appearance
- Different keypoint visibility

# Validation points:
- Homography transformation accuracy
- Court dimension calculations
- Speed/distance measurements
```

### Test 3: Challenging Video Conditions

#### Test 3A: Low-Quality Video

**Video specs**: 720p or lower, compressed, potential artifacts

**Test Setup**:

```python
# Adjust settings for low quality
BALL_DETECTION_CONFIDENCE = 0.3  # Lower threshold
PLAYER_DETECTION_CONFIDENCE = 0.4
VIDEO_RESOLUTION_SCALE = 1.0  # Don't scale down further
```

**Validation Focus**:

```markdown
- [ ] System handles low resolution gracefully
- [ ] Detection still works despite video compression
- [ ] No increase in false positives
- [ ] Processing doesn't fail with poor quality input
```

#### Test 3B: Poor Lighting Conditions

**Scenarios**: Shadows, glare, uneven lighting, night games

**Test Variations**:

```python
# Heavy shadows
- Court partially in shadow
- Players moving in/out of shadows
- Ball visibility affected by lighting

# Bright glare
- Overexposed areas of court
- Sun glare affecting camera
- Washed out court lines

# Night/indoor low light
- Artificial lighting
- Color temperature differences
- Potential motion blur
```

**Validation Strategy**:

```python
# Check detection confidence scores
def analyze_detection_quality(results_file):
    with open(results_file) as f:
        data = json.load(f)
    
    ball_confidences = [frame['ball']['confidence'] 
                       for frame in data if 'ball' in frame]
    
    avg_confidence = sum(ball_confidences) / len(ball_confidences)
    print(f"Average ball detection confidence: {avg_confidence:.3f}")
    
    # Expect: >0.6 for good conditions, >0.4 for challenging
    return avg_confidence > 0.4
```

#### Test 3C: Partial Court Visibility

**Scenarios**: Court edges cut off, obstacles blocking view

**Test Setup**:

```python
# Expected challenges:
- Not all 12 keypoints visible
- Some court areas never seen
- Players entering/exiting frame

# Adaptation strategies:
- Use visible keypoints only
- Extrapolate court geometry
- Handle partial tracking data
```

**Handling Partial Keypoints**:

```python
# In ui.py, modify keypoint selection
visible_keypoints = [1, 2, 3, 4, 6, 7]  # Only these are visible
missing_keypoints = [5, 8, 9, 10, 11, 12]

# System should:
- Accept partial keypoint sets
- Warn user about reduced accuracy
- Still provide meaningful analysis
```

### Test 4: Specialized Scenarios

#### Test 4A: Wide-Angle (Fisheye) Lens

**Challenge**: Distortion correction needed

**Pre-processing**:

```python
# Add lens distortion correction
import cv2

def undistort_fisheye(frame, camera_matrix, dist_coeffs):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, 
                               None, new_camera_matrix)
    return undistorted

# Apply before analysis
corrected_video = undistort_video(original_video)
```

#### Test 4B: Multiple Camera Angles (Same Game)

**Purpose**: Test consistency across different viewpoints

**Test Process**:

```python
# Analyze same rally from 2+ camera angles
camera_1_results = analyze_video("rally_camera1.mp4")
camera_2_results = analyze_video("rally_camera2.mp4")

# Compare results
def compare_multi_camera_results(results1, results2):
    # Ball speed should be similar
    speed1 = results1['average_ball_speed']
    speed2 = results2['average_ball_speed']
    speed_diff = abs(speed1 - speed2) / max(speed1, speed2)
    
    # Player distances should be consistent
    dist1 = results1['total_player_distance']
    dist2 = results2['total_player_distance']
    dist_diff = abs(dist1 - dist2) / max(dist1, dist2)
    
    return speed_diff < 0.2, dist_diff < 0.3  # 20% and 30% tolerance
```

#### Test 4C: Different Frame Rates

**Purpose**: Ensure system works with various fps

**Test Configurations**:

```python
# Test with different frame rates
test_frame_rates = [30, 60, 120, 240]

for fps in test_frame_rates:
    # Convert video to target fps
    convert_video_fps(original_video, f"test_{fps}fps.mp4", fps)
    
    # Run analysis
    results = analyze_video(f"test_{fps}fps.mp4")
    
    # Validate temporal accuracy
    validate_temporal_results(results, fps)
```

---

## ⚡ Performance Testing

### Benchmark Standard Videos

**Create Performance Baseline**:

```python
# performance_test.py
import time
import psutil
import torch

def benchmark_analysis(video_path, iterations=3):
    results = []
    
    for i in range(iterations):
        # Clear caches
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Monitor resources
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        # Run analysis
        run_analysis(video_path)
        
        # Collect metrics
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        results.append({
            'processing_time': end_time - start_time,
            'memory_delta': (end_memory - start_memory) / 1024**2,  # MB
            'peak_gpu_memory': torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        })
    
    return results

# Expected benchmarks (varies by hardware)
EXPECTED_PERFORMANCE = {
    'processing_time_per_second': 2.0,  # seconds of processing per second of video
    'memory_per_frame': 5.0,            # MB per video frame
    'gpu_memory_total': 2000             # MB total GPU usage
}
```

### Stress Testing

#### Test 5A: Long Video Processing

**Purpose**: Test memory management and stability

```python
# Test with progressively longer videos
test_durations = [1, 5, 15, 30, 60]  # minutes

for duration in test_durations:
    print(f"Testing {duration}-minute video...")
    
    # Monitor memory usage over time
    memory_usage = []
    start_time = time.time()
    
    def memory_monitor():
        while processing:
            memory_usage.append(psutil.virtual_memory().percent)
            time.sleep(1)
    
    # Start monitoring in background
    monitor_thread = threading.Thread(target=memory_monitor)
    monitor_thread.start()
    
    # Run analysis
    try:
        analyze_video(f"long_video_{duration}min.mp4")
        print(f"✅ {duration}-minute video completed successfully")
    except MemoryError:
        print(f"❌ {duration}-minute video failed due to memory")
    
    processing = False
    monitor_thread.join()
    
    # Analyze memory patterns
    max_memory = max(memory_usage)
    memory_growth = memory_usage[-1] - memory_usage[0]
    print(f"Peak memory: {max_memory}%, Growth: {memory_growth}%")
```

#### Test 5B: Batch Processing

**Purpose**: Test system stability across multiple videos

```python
# Process multiple videos in sequence
test_videos = [
    "rally1.mp4", "rally2.mp4", "rally3.mp4", 
    "rally4.mp4", "rally5.mp4"
]

for i, video in enumerate(test_videos):
    print(f"Processing video {i+1}/{len(test_videos)}: {video}")
    
    # Memory before processing
    memory_before = psutil.virtual_memory().used
    
    try:
        analyze_video(video)
        
        # Memory after processing
        memory_after = psutil.virtual_memory().used
        memory_delta = (memory_after - memory_before) / 1024**2
        
        print(f"✅ Completed. Memory delta: {memory_delta:.1f}MB")
        
        # Check for memory leaks
        if memory_delta > 500:  # 500MB threshold
            print(f"⚠️ Potential memory leak detected")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
```

---

## 🎯 Accuracy Validation Testing

### Ground Truth Comparison

#### Test 6A: Manual Annotation Validation

**Purpose**: Compare AI results with human annotation

```python
# Create ground truth annotations
def create_ground_truth(video_path, output_path):
    """Manually annotate key frames for accuracy comparison"""
    
    cap = cv2.VideoCapture(video_path)
    frame_annotations = {}
    
    # Annotate every 30th frame
    for frame_num in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Manual annotation interface
            ball_pos = manual_ball_annotation(frame)
            player_boxes = manual_player_annotation(frame)
            
            frame_annotations[frame_num] = {
                'ball_position': ball_pos,
                'player_bounding_boxes': player_boxes
            }
    
    # Save ground truth
    with open(output_path, 'w') as f:
        json.dump(frame_annotations, f)

def compare_with_ground_truth(ai_results, ground_truth):
    """Compare AI predictions with manually annotated ground truth"""
    
    ball_errors = []
    player_errors = []
    
    for frame_num in ground_truth:
        if frame_num in ai_results:
            # Ball position error
            gt_ball = ground_truth[frame_num]['ball_position']
            ai_ball = ai_results[frame_num]['ball_position']
            
            if gt_ball and ai_ball:
                error = distance(gt_ball, ai_ball)
                ball_errors.append(error)
            
            # Player detection accuracy
            gt_players = ground_truth[frame_num]['player_bounding_boxes']
            ai_players = ai_results[frame_num]['player_detections']
            
            # Calculate IoU for player detections
            player_ious = calculate_player_ious(gt_players, ai_players)
            player_errors.extend(player_ious)
    
    return {
        'mean_ball_error_pixels': np.mean(ball_errors),
        'mean_player_iou': np.mean(player_errors),
        'ball_detection_rate': len(ball_errors) / len(ground_truth),
        'player_detection_rate': len(player_errors) / len(ground_truth)
    }
```

#### Test 6B: Physics-Based Validation

**Purpose**: Check if calculated physics make sense

```python
def validate_physics(results):
    """Check if calculated speeds and trajectories are physically reasonable"""
    
    # Ball speed validation
    ball_speeds = [frame['ball_speed'] for frame in results if 'ball_speed' in frame]
    
    # Padel ball speeds typically 20-180 km/h
    unrealistic_speeds = [s for s in ball_speeds if s < 5 or s > 200]
    speed_validation = len(unrealistic_speeds) / len(ball_speeds) < 0.05  # <5% unrealistic
    
    # Player speed validation  
    player_speeds = [frame['player_speed'] for frame in results if 'player_speed' in frame]
    
    # Human running speeds typically 0-25 km/h
    unrealistic_player_speeds = [s for s in player_speeds if s > 30]
    player_speed_validation = len(unrealistic_player_speeds) / len(player_speeds) < 0.02
    
    # Ball trajectory validation
    ball_positions = [(frame['ball_x'], frame['ball_y']) for frame in results if 'ball_x' in frame]
    
    # Check for impossible jumps (teleportation)
    max_reasonable_jump = 100  # pixels between frames
    large_jumps = 0
    for i in range(1, len(ball_positions)):
        distance = np.sqrt((ball_positions[i][0] - ball_positions[i-1][0])**2 + 
                          (ball_positions[i][1] - ball_positions[i-1][1])**2)
        if distance > max_reasonable_jump:
            large_jumps += 1
    
    trajectory_validation = large_jumps / len(ball_positions) < 0.1  # <10% large jumps
    
    return {
        'speed_validation': speed_validation,
        'player_speed_validation': player_speed_validation,
        'trajectory_validation': trajectory_validation,
        'overall_physics_score': (speed_validation + player_speed_validation + trajectory_validation) / 3
    }
```

---

## 🔧 Edge Case Testing

### Test 7: Unusual Scenarios

#### Test 7A: Empty Court (No Players)

**Purpose**: Test system behavior when no players are present

```python
# Create test video with only ball (practice shots, etc.)
def test_no_players():
    # Expected behavior:
    # - Ball tracking should continue working
    # - Player analytics should gracefully handle empty data
    # - No system crashes
    
    results = analyze_video("ball_only_practice.mp4")
    
    assert 'ball_tracking' in results
    assert results['player_count'] == 0
    assert len(results['player_analytics']) == 0
```

#### Test 7B: No Ball Visible

**Purpose**: Test with warm-up videos where ball isn't used

```python
def test_no_ball():
    # Players moving around court without ball
    results = analyze_video("warmup_no_ball.mp4")
    
    # Should still track players
    assert 'player_tracking' in results
    assert results['ball_detections'] == 0
    # Should not crash or produce false ball detections
```

#### Test 7C: Multiple Balls

**Purpose**: Test with practice scenarios involving multiple balls

```python
def test_multiple_balls():
    # Practice session with ball bucket
    results = analyze_video("practice_multiple_balls.mp4")
    
    # System should either:
    # 1. Track the active ball in play
    # 2. Provide warning about multiple ball detections
    # 3. Allow user to select which ball to track
```

#### Test 7D: Non-Padel Content

**Purpose**: Test robustness with unrelated videos

```python
def test_non_padel_content():
    # Test with tennis, squash, random videos
    test_videos = ["tennis.mp4", "squash.mp4", "random_sports.mp4"]
    
    for video in test_videos:
        # Should either:
        # 1. Gracefully detect no padel content
        # 2. Provide reasonable analysis if similar sport
        # 3. Not crash or produce nonsensical results
        pass
```

---

## 📊 Automated Testing Framework

### Create Test Suite

```python
# test_suite.py
import unittest
import json
import os
from pathlib import Path

class PadelAnalyticsTestSuite(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_videos_dir = Path("test_data/videos")
        cls.test_outputs_dir = Path("test_outputs")
        cls.test_outputs_dir.mkdir(exist_ok=True)
    
    def test_basic_functionality(self):
        """Test basic analysis with standard video"""
        video_path = self.test_videos_dir / "standard_rally.mp4"
        output_dir = self.test_outputs_dir / "basic_test"
        
        # Run analysis
        results = run_analysis(video_path, output_dir)
        
        # Assertions
        self.assertIsNotNone(results)
        self.assertIn('ball_tracking', results)
        self.assertIn('player_tracking', results)
        self.assertGreater(len(results['ball_tracking']), 0)
        
    def test_performance_requirements(self):
        """Test that processing meets performance requirements"""
        video_path = self.test_videos_dir / "30_second_rally.mp4"
        
        start_time = time.time()
        results = run_analysis(video_path)
        processing_time = time.time() - start_time
        
        # Should process 30-second video in under 5 minutes
        self.assertLess(processing_time, 300)
        
    def test_accuracy_requirements(self):
        """Test that analysis meets accuracy requirements"""
        video_path = self.test_videos_dir / "ground_truth_video.mp4"
        ground_truth_path = self.test_videos_dir / "ground_truth_annotations.json"
        
        results = run_analysis(video_path)
        accuracy = compare_with_ground_truth(results, ground_truth_path)
        
        # Accuracy requirements
        self.assertGreater(accuracy['ball_detection_rate'], 0.85)
        self.assertGreater(accuracy['mean_player_iou'], 0.75)
        
    def test_memory_management(self):
        """Test that memory usage stays within bounds"""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process several videos
        for i in range(5):
            video_path = self.test_videos_dir / f"test_video_{i}.mp4"
            run_analysis(video_path)
        
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024**2  # MB
        
        # Memory growth should be less than 1GB
        self.assertLess(memory_growth, 1000)
        
    def test_error_handling(self):
        """Test system behavior with invalid inputs"""
        
        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            run_analysis("non_existent_video.mp4")
        
        # Corrupted video file
        with self.assertRaises(Exception):
            run_analysis(self.test_videos_dir / "corrupted_video.mp4")
        
        # Invalid format
        with self.assertRaises(Exception):
            run_analysis(self.test_videos_dir / "text_file.txt")

if __name__ == '__main__':
    unittest.main()
```

### Continuous Testing

```python
# continuous_test.py
def run_continuous_tests():
    """Run tests continuously to catch regressions"""
    
    while True:
        try:
            # Run full test suite
            result = unittest.TextTestRunner(verbosity=2).run(
                unittest.TestLoader().loadTestsFromTestCase(PadelAnalyticsTestSuite)
            )
            
            # Log results
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
            }
            
            with open('test_log.json', 'a') as f:
                f.write(json.dumps(log_entry) + '
')
            
            # Wait before next test cycle
            time.sleep(3600)  # 1 hour
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Test suite error: {e}")
            time.sleep(300)  # 5 minutes before retry
```

---

## 📈 Results Analysis and Reporting

### Test Report Generation

```python
# test_reporter.py
def generate_test_report(test_results):
    """Generate comprehensive test report"""
    
    report = {
        'summary': {
            'total_tests': len(test_results),
            'passed': len([t for t in test_results if t['status'] == 'passed']),
            'failed': len([t for t in test_results if t['status'] == 'failed']),
            'success_rate': len([t for t in test_results if t['status'] == 'passed']) / len(test_results)
        },
        'performance_metrics': {
            'average_processing_time': np.mean([t['processing_time'] for t in test_results]),
            'average_memory_usage': np.mean([t['memory_usage'] for t in test_results]),
            'average_accuracy': np.mean([t['accuracy'] for t in test_results if 'accuracy' in t])
        },
        'detailed_results': test_results
    }
    
    # Generate HTML report
    html_report = f"""
    <html>
    <head><title>Padel Analytics Test Report</title></head>
    <body>
        <h1>Test Report</h1>
        <h2>Summary</h2>
        <p>Total Tests: {report['summary']['total_tests']}</p>
        <p>Passed: {report['summary']['passed']}</p>
        <p>Failed: {report['summary']['failed']}</p>
        <p>Success Rate: {report['summary']['success_rate']:.2%}</p>
        
        <h2>Performance Metrics</h2>
        <p>Average Processing Time: {report['performance_metrics']['average_processing_time']:.2f}s</p>
        <p>Average Memory Usage: {report['performance_metrics']['average_memory_usage']:.1f}MB</p>
        <p>Average Accuracy: {report['performance_metrics']['average_accuracy']:.2%}</p>
    </body>
    </html>
    """
    
    with open('test_report.html', 'w') as f:
        f.write(html_report)
    
    return report
```

### Performance Tracking

```python
# Track performance over time
def track_performance_trends():
    """Monitor performance trends across test runs"""
    
    # Load historical test data
    with open('test_history.json', 'r') as f:
        history = [json.loads(line) for line in f]
    
    # Analyze trends
    processing_times = [h['avg_processing_time'] for h in history]
    accuracy_scores = [h['avg_accuracy'] for h in history]
    
    # Detect performance regressions
    recent_avg = np.mean(processing_times[-10:])  # Last 10 runs
    historical_avg = np.mean(processing_times[:-10])
    
    if recent_avg > historical_avg * 1.2:  # 20% slower
        print("⚠️ Performance regression detected!")
        print(f"Recent average: {recent_avg:.2f}s")
        print(f"Historical average: {historical_avg:.2f}s")
    
    # Plot trends
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(processing_times)
    plt.title('Processing Time Trend')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Test Run')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_scores)
    plt.title('Accuracy Trend')
    plt.ylabel('Accuracy')
    plt.xlabel('Test Run')
    
    plt.tight_layout()
    plt.savefig('performance_trends.png')
```

---

## 🎓 Best Practices for Testing

### 1. **Systematic Approach**

- Start with basic functionality tests
- Progress to edge cases and stress tests
- Always validate results make physical sense
- Document all test cases and expected outcomes

### 2. **Test Data Management**

```python
# Organize test data systematically
test_data/
├── videos/
│   ├── standard/          # Good quality, typical scenarios
│   ├── challenging/       # Poor lighting, low quality, etc.
│   ├── edge_cases/        # Unusual scenarios
│   └── ground_truth/      # Manually annotated videos
├── expected_outputs/      # Known good results
└── configurations/        # Different config files for testing
```

### 3. **Automated Validation**

```python
def automated_result_validation(results):
    """Automatically check if results are reasonable"""
    
    validations = {
        'ball_speed_range': validate_ball_speeds(results),
        'player_speed_range': validate_player_speeds(results), 
        'court_dimensions': validate_court_measurements(results),
        'tracking_continuity': validate_tracking_continuity(results),
        'detection_confidence': validate_detection_confidence(results)
    }
    
    overall_valid = all(validations.values())
    return overall_valid, validations
```

### 4. **Performance Baselines**

```python
# Establish and maintain performance baselines
PERFORMANCE_BASELINES = {
    'processing_speed': {
        'min_fps': 30,  # Process at least 30 video frames per second
        'max_time_per_frame': 0.5  # Maximum 0.5 seconds per frame
    },
    'memory_usage': {
        'max_ram_gb': 8,  # Maximum 8GB RAM usage
        'max_gpu_gb': 4   # Maximum 4GB GPU memory
    },
    'accuracy': {
        'min_ball_detection_rate': 0.85,
        'min_player_detection_rate': 0.90,
        'max_false_positive_rate': 0.05
    }
}
```

### 5. **Regression Testing**

```python
def regression_test_suite():
    """Run tests to ensure new changes don't break existing functionality"""
    
    # Test with known good videos
    reference_videos = [
        ("rally_1.mp4", "expected_rally_1_results.json"),
        ("rally_2.mp4", "expected_rally_2_results.json"),
        ("practice.mp4", "expected_practice_results.json")
    ]
    
    for video, expected_file in reference_videos:
        current_results = run_analysis(video)
        
        with open(expected_file) as f:
            expected_results = json.load(f)
        
        # Compare results within tolerance
        comparison = compare_results(current_results, expected_results, tolerance=0.1)
        
        if not comparison['match']:
            print(f"⚠️ Regression detected in {video}")
            print(f"Differences: {comparison['differences']}")
```

---

## 🚀 Ready to Test

This comprehensive testing guide provides everything you need to thoroughly validate the Padel Analytics system. Remember:

- **Start simple**: Begin with basic functionality tests
- **Be systematic**: Follow the test scenarios in order
- **Document everything**: Keep records of test results and issues
- **Validate results**: Always check that outputs make sense
- **Monitor performance**: Track speed and accuracy over time

**Next Steps**:

1. Run the Quick Verification Test
2. Work through the video-based scenarios
3. Set up automated testing for regular validation
4. Create your own test cases for specific needs

Happy testing! 🧪🎾
