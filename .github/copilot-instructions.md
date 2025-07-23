# Padel Analytics - AI Coding Agent Instructions

## Project Overview
This is a computer vision system for analyzing padel game recordings using PyTorch models. The system tracks players, ball positions, and court keypoints to generate analytics and visualizations. The architecture is built around a modular tracking pipeline with configurable components.

## Architecture & Core Components

### Entry Points
- `main.py`: CLI entry point with interactive court keypoint selection UI
- `app.py`: Streamlit dashboard for data visualization and analysis
- Both use `config.py` for centralized configuration management

### Data Flow Pattern
1. **Video Input** → **Court Keypoint Selection** (manual UI selection of 12 keypoints)
2. **TrackingRunner** orchestrates multiple tracker instances in parallel
3. **Analytics modules** process tracking data into metrics
4. **Visualization components** render court projections and player-centric graphs

### Key Architectural Patterns

#### Tracker Abstraction (`trackers/tracker.py`)
All trackers inherit from `Tracker` base class with standardized methods:
- `predict_frame()` and `predict_sample()` for different inference modes
- JSON serialization via `from_json()` and `serialize()` 
- Caching support with `load_path` and `save_path` configuration

#### Memory-Efficient Pipeline (`trackers/runner.py`)
`TrackingRunner` implements frame-by-frame processing to handle large videos:
```python
# Example usage pattern:
runner = TrackingRunner(
    trackers=[player_tracker, ball_tracker, keypoints_tracker],
    video_path="input.mp4",
    court_model=court_3d_model
)
runner.run()
```

#### Configuration-Driven Design (`config.py`)
All model paths, batch sizes, and processing parameters are centralized:
- Model weights paths: `PLAYERS_TRACKER_MODEL`, `BALL_TRACKER_MODEL`, etc.
- Batch sizes for GPU memory management
- Cache paths for intermediate results

## Critical Developer Workflows

### Model Setup
1. Download pre-trained weights from Google Drive link in README
2. Update paths in `config.py` to point to downloaded weights
3. Models expect specific input formats (e.g., `PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE = 1280`)

### Court Keypoint Selection
**Critical**: The system requires manual selection of 12 court keypoints in specific order:
```
k11--------------------k12
|                       |
k8-----------k9--------k10
|            |          |
k6----------------------k7
|            |          |
k3-----------k4---------k5
|                       |
k1----------------------k2
```
- UI appears automatically when running `main.py`
- Keypoints are saved to `FIXED_COURT_KEYPOINTS_SAVE_PATH` for reuse
- See `examples/videos/select_keypoints.mp4` for demonstration

### Testing & Validation
No formal test suite - validation is done through:
1. Running `python main.py` with sample videos in `examples/videos/`
2. Checking output visualizations for correctness
3. Streamlit dashboard (`python -m streamlit run app.py`) for interactive analysis

### Performance Optimization
- Adjust batch sizes in `config.py` based on available VRAM (minimum 8GB recommended)
- Use caching extensively - set `LOAD_PATH` to skip re-computation
- `MAX_FRAMES` config limits processing for testing

## Project-Specific Conventions

### File Organization
- `trackers/`: Each tracker type has its own subdirectory with models, datasets, and prediction logic
- `constants/`: Physical measurements (`court_dimensions.py`, `player_heights.py`)
- `analytics/`: Data processing and metric calculation
- `visualizations/`: Plotting and court projection rendering
- `improvements/`: Active development features (EKF, robust tracking, 14-keypoint homography)

### Data Handling Patterns
- All tracking objects implement `Object` base class with serialization
- Use `DataPoint` and `PlayerPosition` dataclasses for structured analytics data
- Coordinates are in meters for real-world measurements, pixels for image space

### External Dependencies
- **PyTorch/Ultralytics**: YOLOv8 models for object detection
- **OpenCV**: Video processing and image manipulation  
- **Supervision**: Detection post-processing and visualization
- **Streamlit**: Interactive dashboard framework
- **Plotly**: Advanced plotting for analytics

## Current Development Areas
Based on `improvements/` directory:
- Extended Kalman Filter (EKF) for better tracking
- 14-keypoint court detection (currently uses 12)
- Device management for multi-GPU setups
- Robust tracking algorithms for occlusion handling

## Key Files for Understanding System
- `trackers/runner.py`: Main orchestration logic
- `config.py`: All configurable parameters
- `main.py` lines 24-38: Court keypoint numbering diagram
- `analytics/data_analytics.py`: Core metrics calculation
- `docs/ARCHITECTURE.md`: System overview with visual diagrams
