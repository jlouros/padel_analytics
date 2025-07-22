# Architecture Overview

The project follows a modular architecture that begins with video input and ends with insightful visualizations. Here’s a simplified breakdown of how the components work together:

1. **Configuration and Input:** The process starts with `main.py`, which loads settings from `config.py`. The user then provides a video and selects key points on the court through the UI managed by `ui.py`.

2. **Tracking:** The `trackers/runner.py` script takes the video and key points as input and uses various models to track the ball and players frame by frame.

3. **Data Analysis:** The tracking data is passed to the `analytics/` modules, where it is transformed into meaningful metrics like player heatmaps and ball speed.

4. **Visualization:** Finally, the `visualizations/` scripts use the processed data to generate visual outputs, such as a 2D projection of the court and performance graphs, giving you a clear view of the game's dynamics.

## Visual Diagram

```mermaid
graph TD
    subgraph "Entry Points"
        A[main.py]:::python --> B(app.py):::python
    end

    subgraph "Configuration"
        C[config.py]:::python --> B
    end

    subgraph "UI"
        D[ui.py]:::python --> B
    end

    subgraph "Core Logic"
        B --> E{trackers}:::folder
        E --> F[runner.py]:::python
        F --> G[ball_tracker]:::component
        F --> H[players_tracker]:::component
        F --> I[keypoints_tracker]:::component
        F --> J[players_keypoints_tracker]:::component
    end

    subgraph "Analytics"
        E --> K{analytics}:::folder
        K --> L[data_analytics.py]:::python
        K --> M[projected_court.py]:::python
    end

    subgraph "Visualizations"
        B --> N{visualizations}:::folder
        N --> O[padel_court.py]:::python
        N --> P[player_centric_graphs.py]:::python
    end

    subgraph "Utilities"
        Q[utils]:::folder --> B
    end

    subgraph "Packages"
        R[opencv-python]:::package --> E
        S[pims]:::package --> B
        T[plotly]:::package --> N
        U[supervision]:::package --> E
        V[ultralytics]:::package --> E
        W[streamlit]:::package --> B
        X[parse]:::package --> G
    end

    classDef python fill:#3498DB,color:#fff
    classDef folder fill:#F1C40F,color:#fff
    classDef component fill:#E74C3C,color:#fff
    classDef package fill:#2ECC71,color:#fff
```
