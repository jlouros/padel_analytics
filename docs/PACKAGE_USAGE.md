# Package Usage

- **kaleav:** This package is not used in the codebase.

- **ffmpeg:** This package is not used in the codebase.

- **matplotlib:** This package is not used in the codebase.

- **opencv-python:** Used for image and video processing. It is imported as `cv2` and used in the following files:
    - `analytics/projected_court.py`
    - `main.py`
    - `trackers/ball_tracker/ball_tracker.py`
    - `trackers/ball_tracker/iterable.py`
    - `trackers/ball_tracker/predict.py`
    - `trackers/keypoints_tracker/iterable.py`
    - `trackers/keypoints_tracker/keypoints_tracker.py`
    - `trackers/players_keypoints_tracker/players_keypoints_tracker.py`
    - `trackers/players_tracker/players_tracker.py`
    - `trackers/runner.py`
    - `trackers/velocity_in_time.py`
    - `ui.py`
    - `utils/video.py`

- **pims:** Used for reading and processing sequences of images. It is used in the following files:
    - `app.py`

- **plotly:** Used for creating interactive plots. It is used in the following files:
    - `app.py`
    - `visualizations/padel_court.py`

- **seaborn:** This package is not used in the codebase.

- **supervision:** Used for visualizing and processing object detection results. It is imported as `sv` and used in the following files:
    - `analytics/projected_court.py`
    - `app.py`
    - `main.py`
    - `trackers/ball_tracker/ball_tracker.py`
    - `trackers/keypoints_tracker/keypoints_tracker.py`
    - `trackers/players_keypoints_tracker/players_keypoints_tracker.py`
    - `trackers/players_tracker/players_tracker.py`
    - `trackers/runner.py`
    - `trackers/tracker.py`

- **ultralytics:** Used for running YOLO models. It is used in the following files:
    - `trackers/keypoints_tracker/keypoints_tracker.py`
    - `trackers/players_keypoints_tracker/players_keypoints_tracker.py`
    - `trackers/players_tracker/players_tracker.py`

- **streamlit:** Used for creating the web app. It is used in the following files:
    - `app.py`

- **parse:** Used for parsing strings. It is used in the following files:
    - `trackers/ball_tracker/dataset.py`
