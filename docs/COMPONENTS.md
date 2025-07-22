# Project Components

- **main.py:** The main script to run the analysis. It uses **app.py** to start the analysis and **config.py** to set it up. It includes a diagram showing how to number the court's key points for the analysis.

- **app.py:** This is the main application file that uses the configurations from **config.py** to run the analysis.

- **config.py:** A key file for customizing the analysis, allowing you to set video paths, model weights, and other parameters.

- **ui.py:** Manages the user interface for selecting the 12 key points on the court, which are essential for the geometric calculations in the analysis.

- **trackers/:** This folder contains all the tracking modules. The **runner.py** script coordinates the different trackers, such as those for the ball, players, and key points, to process the video data.

- **analytics/:** Here, the data from the trackers is processed. **data_analytics.py** calculates metrics like player speed and ball velocity, while **projected_court.py** creates a 2D representation of the game.

- **visualizations/:** This folder includes scripts for creating visualizations. **padel_court.py** draws the court, and **player_centric_graphs.py** generates charts related to player performance.

- **utils/:** Provides utility functions for video processing and data conversion, supporting the main analysis pipeline.

- **constants/:** Defines fixed values used across the project, such as court dimensions and average player heights, ensuring consistency in calculations.
