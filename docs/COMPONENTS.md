# 🧩 Project Components Guide

This guide explains each component of the Padel Analytics system in detail. Think of it as a **reference manual** that you can return to when you need to understand what each file or folder does.

## 🎯 How to Use This Guide

- **For beginners**: Read the "What it does" sections to understand the big picture
- **For developers**: Focus on "How it works" and "Key functions" sections
- **For troubleshooting**: Check "Common issues" sections when something goes wrong

---

## 🚪 Entry Points

> These are the files you run to start the application

### 📄 `main.py`

**What it does**: The command-line interface for running video analysis

```python
# When you run: python main.py
# This happens:
1. Loads configuration from config.py
2. Shows UI for selecting court keypoints
3. Starts the tracking process
4. Saves results to files
```

**Key features**:

- **Interactive court setup**: Click to select 12 court keypoints
- **Progress tracking**: Shows which frame is being processed
- **Automatic saving**: Results saved in JSON format for later use

**When to use**:

- First-time setup with a new video
- When you want to process videos offline
- For batch processing multiple videos

**Common issues**:

- ❌ **Keypoint window doesn't appear**: Install GUI backend (`pip install PyQt5`)
- ❌ **Can't see court lines clearly**: Adjust video brightness or try a different frame

### 📄 `app.py`

**What it does**: Web-based dashboard using Streamlit

```python
# When you run: streamlit run app.py
# You get:
- Interactive web interface
- Real-time video analysis
- Live charts and graphs
- Easy parameter adjustment
```

**Key features**:

- **Web interface**: No command line needed
- **Real-time visualization**: See results as they're generated
- **Interactive controls**: Adjust settings without editing config files
- **Shareable results**: Send links to others to view analysis

**When to use**:

- Presenting results to others
- Interactive analysis sessions
- When you prefer web interfaces over command line

**How it connects to other components**:

```none
app.py → config.py (loads settings)
       → trackers/runner.py (runs analysis)
       → visualizations/ (creates charts)
```

---

## ⚙️ Configuration

### 📄 `config.py`

**What it does**: Central control panel for all system settings

Think of this as the **mission control** for your analysis - everything important is configured here.

**Key sections**:

#### 🎬 Video Settings

```python
INPUT_VIDEO_PATH = "path/to/your/video.mp4"
OUTPUT_DIRECTORY = "output/"
CACHE_DIRECTORY = "cache/"
```

**What to change**: Video paths, output locations

#### 🤖 AI Model Settings

```python
BALL_TRACKER_MODEL = "weights/ball_detection/model.pt"
PLAYERS_TRACKER_MODEL = "weights/players_detection/model.pt"
```

**What to change**: Model file paths, confidence thresholds

#### 🔧 Performance Settings

```python
BALL_TRACKER_BATCH_SIZE = 16
PLAYERS_TRACKER_BATCH_SIZE = 16
USE_GPU = True
```

**What to change**: Batch sizes (lower if you get memory errors), GPU usage

**Common modifications**:

```python
# For slower computers:
BATCH_SIZE = 8  # Instead of 16

# For better accuracy:
CONFIDENCE_THRESHOLD = 0.3  # Instead of 0.5

# For different video formats:
INPUT_VIDEO_PATH = "my_video.mov"  # Supports mp4, mov, avi
```

**Common issues**:

- ❌ **File not found errors**: Check that all paths exist
- ❌ **Out of memory errors**: Reduce batch sizes
- ❌ **Slow processing**: Enable GPU if available

---

## 🖼️ User Interface

### 📄 `ui.py`

**What it does**: Handles the court keypoint selection interface

This is where the **human meets the machine** - you tell the AI where the court is by clicking on key points.

**How keypoint selection works**:

```none
1. Shows first frame of video
2. You click on 12 specific court points
3. System calculates court geometry
4. Saves keypoints for future use
```

**The 12 keypoints explained**:

```none
Court Layout (top view):
         Back Wall
    k11 ──────────── k12
     │               │
    k8 ── k9 ──── k10    ← Service line
     │    │        │
    k6 ────────────k7     ← Net
     │    │        │
    k3 ── k4 ──── k5     ← Service line  
     │               │
    k1 ──────────── k2
         Front Wall
```

**Tips for accurate selection**:

- **Start with corners**: k1, k2, k11, k12 are usually easiest to see
- **Look for line intersections**: Click exactly where lines meet
- **Use zoom**: If your video player allows it, zoom in for precision
- **Take your time**: Accurate keypoints = better analysis

**What happens with your keypoints**:

1. **Geometric calculation**: System calculates court dimensions and perspective
2. **3D to 2D mapping**: Converts real-world coordinates to video coordinates
3. **Tracking calibration**: Improves accuracy of player and ball tracking

**Common issues**:

- ❌ **GUI doesn't open**: Install display backend for your system
- ❌ **Can't see court lines**: Try different video frame or enhance brightness
- ❌ **Inaccurate results**: Re-select keypoints more carefully

---

## 🎯 Core Tracking System

### 📁 `trackers/`

**What it does**: The brain of the operation - detects and tracks everything in the video

Think of this folder as a **team of AI specialists**, each focused on tracking different things.

### 📄 `trackers/runner.py`

**What it does**: The conductor that orchestrates all tracking operations

```python
# TrackingRunner coordinates everything:
class TrackingRunner:
    def run(self):
        for frame in video:
            # Each tracker processes the same frame
            ball_data = ball_tracker.predict_frame(frame)
            players_data = players_tracker.predict_frame(frame)
            keypoints_data = keypoints_tracker.predict_frame(frame)
            
            # Combine and save results
            combined_results = self.merge_results(ball_data, players_data, keypoints_data)
            self.save_frame_results(combined_results)
```

**Key responsibilities**:

- **Frame distribution**: Sends each video frame to all trackers
- **Result aggregation**: Combines data from all trackers
- **Memory management**: Ensures system doesn't run out of memory
- **Progress tracking**: Reports processing status

**Why this design is smart**:

- **Parallel processing**: All trackers work on the same frame simultaneously
- **Memory efficient**: Processes one frame at a time instead of loading entire video
- **Modular**: Easy to add new trackers or remove existing ones

### 📁 `trackers/ball_tracker/`

**What it does**: Specialist in finding and following the ball

**How ball tracking works**:

```none
1. AI model scans each frame for ball-like objects
2. Filters results by size, shape, and movement patterns
3. Links detections across frames to create ball trajectory
4. Handles occlusions (when ball disappears behind players)
```

**Key files**:

- `ball_tracker.py`: Main tracking logic
- `predict.py`: AI model inference
- `dataset.py`: Data handling utilities

**What makes ball tracking challenging**:

- **Small size**: Ball is tiny compared to players
- **Fast movement**: Ball moves quickly, creating motion blur
- **Occlusions**: Ball hidden behind players or net
- **Similar objects**: Other round objects might confuse the AI

**Output data**:

```python
{
    "frame_number": 150,
    "ball_position": {"x": 320, "y": 240},
    "confidence": 0.85,
    "velocity": {"x": 15.2, "y": -8.7}
}
```

### 📁 `trackers/players_tracker/`

**What it does**: Specialist in detecting and tracking players

**How player tracking works**:

```none
1. AI model detects human-like shapes in each frame
2. Assigns unique IDs to each player
3. Tracks each player across frames
4. Handles when players cross paths or leave/enter frame
```

**Challenges solved**:

- **Player similarity**: Both players look similar to AI
- **Crossing paths**: When players cross, maintaining correct IDs
- **Partial visibility**: Player partially hidden by net or walls
- **Clothing changes**: Different colored clothing affects detection

**Output data**:

```python
{
    "frame_number": 150,
    "players": [
        {
            "player_id": 1,
            "bounding_box": {"x1": 100, "y1": 200, "x2": 150, "y2": 350},
            "confidence": 0.92
        },
        {
            "player_id": 2,
            "bounding_box": {"x1": 400, "y1": 180, "x2": 450, "y2": 340},
            "confidence": 0.88
        }
    ]
}
```

### 📁 `trackers/keypoints_tracker/`

**What it does**: Specialist in detecting court features and lines

**Why track court keypoints**:

- **Calibration**: Ensures court geometry stays consistent
- **Drift correction**: Camera shake or movement compensation
- **Perspective verification**: Confirms initial keypoint selection accuracy

**What it tracks**:

- Court line intersections
- Net posts
- Service boxes
- Court boundaries

### 📁 `trackers/players_keypoints_tracker/`

**What it does**: Specialist in player pose estimation

**How pose estimation works**:

```none
1. Detects players using players_tracker results
2. Identifies body keypoints (head, shoulders, elbows, knees, etc.)
3. Tracks pose changes over time
4. Calculates movement patterns and biomechanics
```

**Body keypoints tracked**:

```none
Head → Shoulders → Elbows → Wrists
  ↓        ↓         ↓       ↓
 Hips → Knees → Ankles → Feet
```

**Applications**:

- **Movement analysis**: How players move around the court
- **Technique analysis**: Serving, hitting, running form
- **Injury prevention**: Identifying poor movement patterns
- **Performance metrics**: Agility, speed, positioning

---

## 📊 Analytics Engine

### 📁 `analytics/`

**What it does**: Transforms raw tracking data into meaningful insights

Think of this as the **statistics department** that turns observations into insights.

### 📄 `analytics/data_analytics.py`

**What it does**: Calculates performance metrics and game statistics

**Types of analysis performed**:

#### 🏃 Player Movement Analysis

```python
# Examples of what gets calculated:
- Distance covered by each player
- Average speed during rallies
- Time spent in different court areas
- Movement efficiency metrics
```

#### ⚡ Ball Physics Analysis

```python
# Ball tracking insights:
- Ball speed throughout rally
- Trajectory analysis
- Bounce patterns
- Shot power distribution
```

#### 🎯 Game Strategy Analysis

```python
# Strategic insights:
- Court coverage patterns
- Preferred shot locations
- Player positioning relative to ball
- Rally length and intensity
```

**Key functions you might use**:

```python
def calculate_player_speed(player_positions, timestamps):
    """Calculate player movement speed over time"""
    
def analyze_ball_trajectory(ball_positions):
    """Analyze ball flight path and physics"""
    
def generate_heatmap_data(player_positions):
    """Create data for court coverage visualization"""
```

### 📄 `analytics/projected_court.py`

**What it does**: Creates the famous 2D top-down court view

This is where the **magic happens** - converting the angled camera view into a clean, top-down court perspective.

**How 3D to 2D projection works**:

1. Uses the 12 keypoints you selected
2. Calculates camera perspective transformation
3. Maps every pixel from video coordinates to court coordinates
4. Creates a "bird's eye view" of the action

**Mathematical concepts involved**:

- **Homography**: Mathematical transformation between perspectives
- **Perspective correction**: Removing camera angle distortion
- **Coordinate mapping**: Converting (x,y) video pixels to (x,y) court meters

**Why this is useful**:

- **Clear visualization**: Easy to see player positions and movements
- **Accurate measurements**: Distances and speeds are now in real-world units
- **Strategy analysis**: Tactics become obvious from top-down view
- **Comparison tool**: Compare different players or games on same court layout

---

## 📈 Visualization System

### 📁 `visualizations/`

**What it does**: Creates the charts, graphs, and visual overlays you see in results

### 📄 `visualizations/padel_court.py`

**What it does**: Draws the court and overlays tracking data

**Visual elements created**:

#### 🏟️ Court Visualization

```python
# Court elements drawn:
- Court boundaries and lines
- Net and service boxes
- Player positions (colored dots/boxes)
- Ball trajectory (trail of positions)
- Movement vectors (arrows showing direction)
```

#### 🔥 Heatmaps

```python
# Types of heatmaps:
- Player positioning (where each player spends time)
- Ball impact zones (where ball hits court)
- Movement intensity (speed and activity levels)
```

#### 📊 Real-time Overlays

```python
# Dynamic information shown:
- Current ball speed
- Player distances from net
- Rally duration
- Score tracking (if available)
```

### 📄 `visualizations/player_centric_graphs.py`

**What it does**: Creates detailed performance charts for individual players

**Types of charts generated**:

#### 📈 Performance Over Time

```python
# Time-series charts:
- Speed variations during match
- Distance covered per minute
- Energy expenditure estimates
- Court position changes
```

#### 📊 Comparative Analysis

```python
# Player comparison charts:
- Side-by-side movement patterns
- Skill metric comparisons
- Strategy differences
- Efficiency measurements
```

#### 🎯 Tactical Analysis

```python
# Strategic visualizations:
- Shot placement preferences
- Defensive vs offensive positioning
- Response times to opponent shots
- Court coverage efficiency
```

---

## 🛠️ Utility Functions

### 📁 `utils/`

**What it does**: Provides helper functions used throughout the system

### 📄 `utils/video.py`

**What it does**: Video processing utilities

**Key functions**:

```python
def read_video_frames(video_path):
    """Efficiently read video frame by frame"""
    
def save_annotated_video(frames, annotations, output_path):
    """Save video with tracking overlays"""
    
def extract_frame_at_timestamp(video_path, timestamp):
    """Get a specific frame from video"""
```

**Why these are needed**:

- **Memory efficiency**: Read large videos without loading everything into RAM
- **Format compatibility**: Handle different video formats (mp4, mov, avi)
- **Quality preservation**: Maintain video quality during processing

### 📄 `utils/conversions.py`

**What it does**: Data format conversion utilities

**Key functions**:

```python
def pixel_to_court_coordinates(pixel_x, pixel_y, homography_matrix):
    """Convert video pixel position to real court position"""
    
def court_to_pixel_coordinates(court_x, court_y, homography_matrix):
    """Convert court position back to video pixels"""
    
def normalize_coordinates(coordinates, court_dimensions):
    """Scale coordinates to standard court size"""
```

**Why conversions matter**:

- **Accuracy**: Measurements in real-world units (meters, not pixels)
- **Consistency**: Same coordinate system across different videos
- **Compatibility**: Data can be shared between different analysis tools

---

## 📏 Constants and Configuration

### 📁 `constants/`

**What it does**: Stores fixed values used throughout the project

### 📄 `constants/court_dimensions.py`

**What it does**: Official padel court measurements

```python
# Standard padel court dimensions (in meters):
COURT_LENGTH = 20.0    # Full court length
COURT_WIDTH = 10.0     # Full court width
NET_HEIGHT = 0.88      # Net height at center
SERVICE_BOX_LENGTH = 6.95  # Service area length
# ... and many more precise measurements
```

**Why these constants are important**:

- **Accuracy**: Ensures calculations use official court dimensions
- **Consistency**: Same measurements across all analysis
- **Calibration**: Helps convert pixel measurements to real distances

### 📄 `constants/player_heights.py`

**What it does**: Average player physical characteristics

```python
# Used for pose estimation and scale calculations:
AVERAGE_PLAYER_HEIGHT = 1.75  # meters
AVERAGE_ARM_LENGTH = 0.65     # meters
AVERAGE_LEG_LENGTH = 0.90     # meters
```

**How these are used**:

- **Scale estimation**: If we know player height, we can estimate distances
- **Pose validation**: Check if detected poses are realistic
- **Biomechanical analysis**: Calculate reach, stride length, etc.

---

## 🔗 How Components Work Together

### Data Flow Example

Let's trace what happens to a single video frame:

```none
1. 📹 Frame from video → trackers/runner.py
2. 🤖 AI models detect objects → ball_tracker/, players_tracker/
3. 📊 Raw detections → analytics/data_analytics.py
4. 🧮 Calculations and metrics → analytics/projected_court.py
5. 🗺️ 2D court projection → visualizations/padel_court.py
6. 📈 Charts and overlays → Output files
```

### Configuration Cascade

How settings flow through the system:

```noen
config.py → All components
    ├── Video paths → utils/video.py
    ├── Model settings → trackers/
    ├── Processing parameters → analytics/
    ├── Output settings → visualizations/
    └── Performance tuning → Everywhere
```

## 🎓 For Junior Developers: Key Takeaways

### Understanding the Architecture

1. **Separation of concerns**: Each component has one main job
2. **Data pipeline**: Clear flow from input to output
3. **Configuration-driven**: Easy to modify behavior without code changes
4. **Modular design**: Components can be used independently

### Best Practices Demonstrated

1. **Error handling**: Each component handles its own error cases
2. **Memory management**: Efficient processing of large videos
3. **Code reuse**: Utilities and constants prevent duplication
4. **Documentation**: Each component is self-explaining

### Extension Points

Want to add new features? Here's where to start:

- **New tracker**: Follow the pattern in `trackers/`
- **New visualization**: Add to `visualizations/`
- **New metric**: Extend `analytics/data_analytics.py`
- **New output format**: Modify `utils/conversions.py`
