# Padel Analytics Project Explanation

This document provides a detailed explanation of the Padel Analytics project, including its components, architecture, how to run it, how to test it, and potential improvements.

### 1. Project Components Explanation

- **main.py:** The main script to run the analysis. It uses **app.py** to start the analysis and **config.py** to set it up. It includes a diagram showing how to number the court's key points for the analysis.

- **app.py:** This is the main application file that uses the configurations from **config.py** to run the analysis.

- **config.py:** A key file for customizing the analysis, allowing you to set video paths, model weights, and other parameters.

- **ui.py:** Manages the user interface for selecting the 12 key points on the court, which are essential for the geometric calculations in the analysis.

- **trackers/:** This folder contains all the tracking modules. The **runner.py** script coordinates the different trackers, such as those for the ball, players, and key points, to process the video data.

- **analytics/:** Here, the data from the trackers is processed. **data_analytics.py** calculates metrics like player speed and ball velocity, while **projected_court.py** creates a 2D representation of the game.

- **visualizations/:** This folder includes scripts for creating visualizations. **padel_court.py** draws the court, and **player_centric_graphs.py** generates charts related to player performance.

- **utils/:** Provides utility functions for video processing and data conversion, supporting the main analysis pipeline.

- **constants/:** Defines fixed values used across the project, such as court dimensions and average player heights, ensuring consistency in calculations.

### 2. Architecture Overview

The project follows a modular architecture that begins with video input and ends with insightful visualizations. Here’s a simplified breakdown of how the components work together:

1. **Configuration and Input:** The process starts with `main.py`, which loads settings from `config.py`. The user then provides a video and selects key points on the court through the UI managed by `ui.py`.

2. **Tracking:** The `trackers/runner.py` script takes the video and key points as input and uses various models to track the ball and playersframe by frame.

3. **Data Analysis:** The tracking data is passed to the `analytics/` modules, where it is transformed into meaningful metrics like player heatmaps and ball speed.

4. **Visualization:** Finally, the `visualizations/` scripts use the processed data to generate visual outputs, such as a 2D projection of the court and performance graphs, giving you a clear view of the game's dynamics.

### 3. Execution Guide

To run the project, follow these steps:

1. **Set Up Your Environment:** Make sure you have Python 3.12, create a virtual environment, and install the required libraries from **requirements.txt**.

2. **Download Model Weights:** Get the pre-trained model weights from the link provided in the `README.md` and update the paths in **config.py**.

3. **Run the Analysis:** Execute the `main.py` script from your terminal. A window will appear, prompting you to select the 12 key points on the court, and then the analysis will begin.

### 4. Testing Guide

Since there are no dedicated test files in the repository, the best way to test the project is by running it with a sample video and observing the output. You can use the `rally.mp4` video in the `examples/videos/` folder to check if the analysis works as expected. If the program runs without errors and generates visualizations, you can be confident that the core functionality is working correctly.

### 5. Areas for Improvement

- **Automated Testing:** Adding a dedicated testing framework with unit and integration tests would significantly improve the project's reliability.

- **Configuration Flexibility:** The reliance on `config.py` could be reduced by allowing users to pass arguments through the command line for greater flexibility.

- **Code Duplication:** In the `trackers` folder, the `iterable.py` file is duplicated. This could be resolved by creating a shared module to reduce redundancy.

- **Dependency Management:** The `requirements.txt` file is quite large. It could be streamlined by removing unused libraries and organizing the dependencies more efficiently.

### 6. Package Usage

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
