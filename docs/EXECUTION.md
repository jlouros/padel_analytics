# 🚀 Execution Guide

This guide provides detailed instructions for running the Padel Analytics system. Whether you're analyzing your first video or your hundredth, this guide will help you get reliable results every time.

## 🎯 Quick Start Checklist

Before running any analysis, verify these prerequisites:

```markdown
✅ Prerequisites Checklist:
- [ ] Python 3.12 installed and working
- [ ] Virtual environment created and activated
- [ ] Requirements installed (`pip install -r requirements.txt`)
- [ ] Model weights downloaded and placed correctly
- [ ] Video file ready and accessible
- [ ] Config file updated with correct paths
```

## 📖 Execution Methods

### Method 1: Command Line Interface (Recommended for first-time users)

#### Step 1: Prepare Your Environment

```bash
# Activate your virtual environment
source padel_analytics_env/bin/activate  # Mac/Linux
# or
padel_analytics_env\Scripts\activate     # Windows

# Verify everything is working
python --version  # Should show Python 3.12.x
pip list | grep ultralytics  # Should show ultralytics package
```

#### Step 2: Configure Your Analysis

Open `config.py` and verify these critical settings:

```python
# 🎬 VIDEO CONFIGURATION
INPUT_VIDEO_PATH = "examples/videos/rally.mp4"  # ← Change this to your video
OUTPUT_DIRECTORY = "output/"                     # ← Where results will be saved
CACHE_DIRECTORY = "cache/"                       # ← Temporary files location

# 🤖 MODEL CONFIGURATION  
BALL_TRACKER_MODEL = "weights/ball_detection/model.pt"
PLAYERS_TRACKER_MODEL = "weights/players_detection/model.pt"
KEYPOINTS_TRACKER_MODEL = "weights/court_keypoints_detection/model.pt"
PLAYERS_KEYPOINTS_TRACKER_MODEL = "weights/players_keypoints_detection/model.pt"

# ⚡ PERFORMANCE SETTINGS
USE_GPU = True                    # Set to False if you don't have a GPU
PLAYERS_TRACKER_BATCH_SIZE = 16   # Lower this if you get memory errors
BALL_TRACKER_BATCH_SIZE = 16      # Lower this if you get memory errors
```

#### Step 3: Run the Analysis

```bash
python main.py
```

#### Step 4: Select Court Keypoints

When the keypoint selection window appears:

1. **Take your time** - accuracy here affects all subsequent analysis
2. **Follow the numbering system** shown in the interface
3. **Click precisely** on line intersections, not just nearby
4. **Start with corners** (easier to identify accurately)

**Keypoint Selection Tips:**

```none
🎯 Pro Tips:
- Zoom in if your video player allows it
- Use a well-lit frame where lines are clearly visible  
- If you make a mistake, close the window and restart
- The system saves keypoints, so you only do this once per court angle
```

#### Step 5: Monitor Progress

During processing, you'll see output like:

```none
🔄 Processing frame 150/1000 (15%)
🎾 Ball detected in frame 150
👥 2 players tracked in frame 150
📊 Analytics updated
```

**Expected processing times:**

- **30-second video**: 2-5 minutes
- **2-minute video**: 8-15 minutes  
- **10-minute video**: 40-75 minutes

*Times vary based on computer speed, GPU availability, and video resolution*

### Method 2: Web Interface (Streamlit Dashboard)

#### Step 1: Start the Web Application

```bash
streamlit run app.py
```

This opens a web browser with an interactive dashboard.

#### Step 2: Upload and Configure

1. **Upload video**: Use the file uploader in the sidebar
2. **Adjust settings**: Modify parameters using sliders and dropdowns
3. **Select keypoints**: Interactive court diagram for point selection
4. **Start analysis**: Click the "Run Analysis" button

#### Advantages of Web Interface

- ✅ **Visual feedback**: See results in real-time
- ✅ **Easy parameter tuning**: Adjust settings with sliders
- ✅ **Shareable results**: Send links to colleagues
- ✅ **No command line needed**: Entirely GUI-based

#### When to Use Each Method

- **Command line**: First-time setup, batch processing, scripting
- **Web interface**: Interactive analysis, presentations, parameter experimentation

## 🔧 Configuration Deep Dive

### Essential Configuration Parameters

#### Video Processing Settings

```python
# Frame sampling - process every nth frame for speed
FRAME_SKIP = 1  # Process every frame (1), every other frame (2), etc.

# Video quality settings
VIDEO_RESOLUTION_SCALE = 1.0  # Scale factor for processing (0.5 = half size)
MIN_FRAME_QUALITY = 0.7       # Skip frames below this quality threshold
```

#### AI Model Performance Tuning

```python
# Confidence thresholds - higher = more conservative detection
BALL_DETECTION_CONFIDENCE = 0.5     # 0.0 to 1.0
PLAYER_DETECTION_CONFIDENCE = 0.7   # 0.0 to 1.0
KEYPOINTS_DETECTION_CONFIDENCE = 0.6 # 0.0 to 1.0

# Batch processing - larger batches = faster processing but more memory
BATCH_SIZE_BALL = 16      # Reduce if memory issues
BATCH_SIZE_PLAYERS = 8    # Reduce if memory issues  
BATCH_SIZE_KEYPOINTS = 4  # Reduce if memory issues
```

#### Output and Caching Settings

```python
# Output formats
SAVE_ANNOTATED_VIDEO = True    # Video with overlays
SAVE_RAW_DATA = True          # JSON files with tracking data
SAVE_ANALYTICS_CHARTS = True  # Performance graphs
SAVE_COURT_PROJECTION = True  # 2D court view

# Caching for faster re-runs
ENABLE_CACHING = True         # Cache intermediate results
CACHE_EXPIRY_HOURS = 24      # How long to keep cached data
```

## 🚨 Troubleshooting Common Issues

### Issue 1: Memory Errors

**Symptoms**: "CUDA out of memory" or "RuntimeError: out of memory"

**Solutions** (try in order):

```python
# 1. Reduce batch sizes in config.py
PLAYERS_TRACKER_BATCH_SIZE = 4  # Instead of 16
BALL_TRACKER_BATCH_SIZE = 4     # Instead of 16

# 2. Scale down video resolution
VIDEO_RESOLUTION_SCALE = 0.5    # Process at half resolution

# 3. Disable GPU processing
USE_GPU = False                 # Use CPU instead

# 4. Enable frame skipping
FRAME_SKIP = 2                  # Process every other frame
```

### Issue 2: Poor Detection Quality  

**Symptoms**: Ball or players not detected accurately

**Diagnosis steps**:

```python
# Check video quality
- Resolution: Minimum 720p recommended
- Lighting: Avoid heavy shadows or glare
- Camera angle: Side view works best
- Court visibility: Full court should be visible

# Adjust detection thresholds
BALL_DETECTION_CONFIDENCE = 0.3    # Lower = more detections
PLAYER_DETECTION_CONFIDENCE = 0.5  # Lower = more detections
```

### Issue 3: Keypoint Selection Problems

**Symptoms**: Can't select keypoints or poor court projection

**Solutions**:

```bash
# Install GUI backend
pip install PyQt5

# Try alternative GUI backend
pip install tkinter

# For remote servers (no display)
export DISPLAY=:0  # Linux
# or use X11 forwarding with SSH
```

## 📊 Understanding Output Results

### Generated Files and Folders

After successful execution, you'll find:

```none
output/
├── 📹 annotated_video.mp4          # Video with tracking overlays
├── 📊 analytics/
│   ├── player_heatmaps.png         # Where players spent time
│   ├── ball_trajectory.png         # Ball path visualization  
│   ├── speed_analysis.png          # Speed over time charts
│   └── court_coverage.png          # Court usage statistics
├── 📈 visualizations/
│   ├── 2d_court_projection.png     # Top-down court view
│   ├── player_movements.gif        # Animated movement patterns
│   └── rally_analysis.png          # Rally-by-rally breakdown
└── 📄 raw_data/
    ├── ball_tracking.json          # Raw ball position data
    ├── player_tracking.json        # Raw player position data
    ├── keypoints_data.json         # Court geometry data
    └── analytics_summary.json      # Calculated metrics
```

### Interpreting Results

#### Ball Tracking Data

```json
{
    "frame_150": {
        "position": {"x": 320, "y": 240},
        "velocity": {"speed": 25.3, "direction": 45},
        "confidence": 0.89
    }
}
```

#### Player Analytics

```json
{
    "player_1": {
        "total_distance": 156.7,      # meters covered
        "average_speed": 2.1,         # m/s
        "max_speed": 8.4,            # m/s
        "court_coverage": 68.3,       # percentage of court visited
        "time_near_net": 23.1        # seconds spent near net
    }
}
```

## 🎓 Best Practices for Reliable Results

### Video Preparation

1. **Stable camera**: Minimize camera shake or movement
2. **Good lighting**: Avoid shadows across court lines  
3. **Full court view**: Ensure entire court is visible
4. **High resolution**: 1080p minimum, 4K preferred
5. **Consistent angle**: Side view works better than end view

### Keypoint Selection

1. **Precise clicking**: Click exactly on line intersections
2. **Good frame choice**: Use frame where all lines are visible
3. **Consistent order**: Always follow the numbered sequence
4. **Double-check**: Verify keypoints look correct before proceeding

### Performance Monitoring

1. **Watch memory usage**: Monitor RAM and GPU memory
2. **Check logs**: Read console output for warnings or errors
3. **Validate results**: Review output videos for tracking accuracy
4. **Iterate parameters**: Adjust settings based on initial results

---

## 🚀 Ready to Start?

With this execution guide, you have everything needed to run reliable, high-quality padel video analysis. Remember:

- **Start simple**: Use default settings for your first analysis
- **Iterate gradually**: Adjust parameters based on results
- **Monitor performance**: Watch for memory issues or slow processing
- **Validate output**: Always check that results make sense

**Next steps**: Run your first analysis, then explore the `TESTING.md` guide to learn about different scenarios and validation techniques.
