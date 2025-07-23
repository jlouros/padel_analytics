# 📦 Package Usage Guide

This guide explains how each Python package is used in the Padel Analytics system. Understanding these dependencies will help you troubleshoot issues, optimize performance, and potentially extend the system.

## 🎯 What You'll Learn

- Purpose of each package in the system
- How packages interact with each other
- Which packages are critical vs optional
- Common issues and solutions for each package
- How to optimize package usage

## 📋 Package Overview

The project uses several key categories of packages:

- **🤖 AI/ML Libraries**: Core computer vision and machine learning
- **🎬 Video Processing**: Video reading, writing, and manipulation
- **📊 Visualization**: Charts, graphs, and interactive displays
- **🛠️ Utilities**: General purpose tools and helpers
- **🖥️ Interface**: Web interface and user interaction

---

## 🤖 AI/ML Libraries

### 📦 `ultralytics` - YOLO Model Framework

**What it does**: Provides the YOLO (You Only Look Once) object detection models

**Where it's used**:

```python
# Core detection in all tracker modules
trackers/keypoints_tracker/keypoints_tracker.py
trackers/players_keypoints_tracker/players_keypoints_tracker.py  
trackers/players_tracker/players_tracker.py
```

**Key functions**:

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Run inference on image
results = model('image.jpg')

# Train custom model
model.train(data='dataset.yaml', epochs=100)
```

**Why it's important**:

- 🎯 **Core detection engine**: Powers all object detection in the system
- ⚡ **State-of-the-art performance**: Latest YOLO models with excellent speed/accuracy balance
- 🔧 **Easy to use**: Simple API for loading and using pre-trained models
- 📈 **Actively maintained**: Regular updates with improvements

**Common issues and solutions**:

```python
# Issue: CUDA out of memory
# Solution: Use smaller model or reduce batch size
model = YOLO('yolov8n.pt')  # Instead of yolov8l.pt
results = model(images, batch=8)  # Instead of batch=16

# Issue: Slow inference on CPU
# Solution: Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('model.pt').to(device)
```

### 📦 `supervision` - Computer Vision Utilities

**What it does**: Provides utilities for processing and visualizing computer vision results

**Where it's used**:

```python
# Visualization and result processing
analytics/projected_court.py
app.py
main.py
trackers/ball_tracker/ball_tracker.py
trackers/keypoints_tracker/keypoints_tracker.py
trackers/players_keypoints_tracker/players_keypoints_tracker.py
trackers/players_tracker/players_tracker.py
trackers/runner.py
trackers/tracker.py
```

**Key functions**:

```python
import supervision as sv

# Create bounding box visualizer
box_annotator = sv.BoxAnnotator()

# Draw detections on image
annotated_frame = box_annotator.annotate(
    scene=frame,
    detections=detections
)

# Track objects across frames
tracker = sv.ByteTracker()
tracks = tracker.update_with_detections(detections)
```

**Why it's important**:

- 🎨 **Rich visualization**: Easy-to-use tools for drawing bounding boxes, labels, trajectories
- 📊 **Data processing**: Utilities for converting between different detection formats
- 🔍 **Tracking**: Built-in object tracking algorithms
- 🎯 **Optimized**: High-performance implementations of common CV tasks

**Common use cases**:

```python
# Visualizing ball trajectory
trail_annotator = sv.TrailAnnotator()
annotated_frame = trail_annotator.annotate(frame, detections)

# Converting YOLO results to supervision format
detections = sv.Detections.from_ultralytics(yolo_results)

# Filtering detections by confidence
high_conf_detections = detections[detections.confidence > 0.8]
```

---

## 🎬 Video Processing

### 📦 `opencv-python` (cv2) - Computer Vision

**What it does**: Core computer vision library for image and video processing

**Where it's used**:

```python
# Video processing throughout the system
analytics/projected_court.py
main.py
trackers/ball_tracker/ball_tracker.py
trackers/ball_tracker/iterable.py
trackers/ball_tracker/predict.py
trackers/keypoints_tracker/iterable.py
trackers/keypoints_tracker/keypoints_tracker.py
trackers/players_keypoints_tracker/players_keypoints_tracker.py
trackers/players_tracker/players_tracker.py
trackers/runner.py
trackers/velocity_in_time.py
ui.py
utils/video.py
```

**Key functions**:

```python
import cv2

# Read video file
cap = cv2.VideoCapture('video.mp4')
ret, frame = cap.read()

# Image processing
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(frame, (640, 480))

# Write video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))
out.write(frame)
```

**Why it's critical**:

- 📹 **Video I/O**: Reading and writing video files
- 🖼️ **Image processing**: Resizing, color conversion, filtering
- 🔧 **Geometric transformations**: Perspective correction, homography
- 🎯 **Core dependency**: Almost every component uses OpenCV

**Performance tips**:

```python
# Efficient video reading
cap = cv2.VideoCapture('video.mp4')
cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)  # Reduce buffer for real-time processing

# Memory-efficient image processing
frame = cv2.resize(frame, None, fx=0.5, fy=0.5)  # Reduce size to save memory

# GPU acceleration (if available)
gpu_frame = cv2.cuda_GpuMat()
gpu_frame.upload(frame)
# Process on GPU
gpu_result = cv2.cuda.resize(gpu_frame, (640, 480))
```

### 📦 `pims` - Video Frame Iterator

**What it does**: Provides efficient video frame iteration and processing

**Where it's used**:

```python
# Video frame handling
app.py
```

**Key functions**:

```python
import pims

# Efficient video iteration
video = pims.Video('video.mp4')
for frame in video:
    # Process each frame
    processed_frame = process_frame(frame)

# Random access to frames
frame_100 = video[100]  # Get frame number 100
```

**Why it's useful**:

- ⚡ **Efficient iteration**: Memory-efficient video frame processing
- 🎯 **Random access**: Jump to specific frames without loading entire video
- 🔧 **Format support**: Handles various video formats consistently
- 📊 **Metadata access**: Easy access to video properties (fps, duration, etc.)

---

## 📊 Visualization Libraries

### 📦 `plotly` - Interactive Visualizations

**What it does**: Creates interactive charts and graphs for data visualization

**Where it's used**:

```python
# Interactive charts and dashboards
app.py
visualizations/padel_court.py
```

**Key functions**:

```python
import plotly.graph_objects as go
import plotly.express as px

# Create interactive line chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=ball_speed, name='Ball Speed'))
fig.show()

# Create heatmap
fig = px.density_heatmap(
    x=player_x_positions, 
    y=player_y_positions,
    title='Player Position Heatmap'
)
```

**Why it's powerful**:

- 🖱️ **Interactive**: Users can zoom, pan, and explore data
- 📱 **Web-ready**: Works seamlessly in web browsers
- 🎨 **Professional**: High-quality, publication-ready visualizations
- 📊 **Variety**: Supports many chart types (line, bar, heatmap, 3D, etc.)

**Common chart types in the project**:

```python
# Ball trajectory visualization
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=ball_x, y=ball_y, z=ball_z,
    mode='lines+markers',
    name='Ball Trajectory'
))

# Player heatmap
fig = go.Figure(data=go.Heatmap(
    z=court_coverage_matrix,
    colorscale='Viridis'
))

# Speed analysis
fig = px.line(
    x=timestamps, y=speeds,
    title='Player Speed Over Time'
)
```

### 📦 `streamlit` - Web Interface

**What it does**: Creates the web-based dashboard for interactive analysis

**Where it's used**:

```python
# Web application interface
app.py
```

**Key functions**:

```python
import streamlit as st

# Create sidebar controls
st.sidebar.title("Analysis Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Display video
st.video("output_video.mp4")

# Show interactive charts
st.plotly_chart(speed_chart)

# File upload
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi'])
```

**Why it's perfect for this project**:

- 🚀 **Rapid development**: Create web apps with minimal code
- 🎯 **Data science focused**: Built specifically for data visualization and analysis
- 📱 **Responsive**: Works on desktop and mobile
- 🔧 **Easy deployment**: Simple to share and deploy

**Interface components used**:

```python
# Video analysis dashboard
col1, col2 = st.columns(2)

with col1:
    st.video(original_video)
    st.caption("Original Video")

with col2:
    st.video(annotated_video)
    st.caption("Analyzed Video")

# Performance metrics
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
metrics_col1.metric("Ball Speed", "25.3 km/h", "2.1 km/h")
metrics_col2.metric("Rally Length", "12 seconds", "-3 seconds")
metrics_col3.metric("Court Coverage", "68%", "5%")
```

---

## 🛠️ Utility Libraries

### 📦 `parse` - String Parsing

**What it does**: Provides pattern-based string parsing capabilities

**Where it's used**:

```python
# Data extraction and parsing
trackers/ball_tracker/dataset.py
```

**Key functions**:

```python
import parse

# Parse filename patterns
pattern = "frame_{frame_number:d}_{timestamp:f}.jpg"
result = parse.parse(pattern, "frame_0150_12.345.jpg")
print(result['frame_number'])  # 150
print(result['timestamp'])     # 12.345

# Parse detection results
template = "Ball detected at ({x:f}, {y:f}) with confidence {conf:f}"
parsed = parse.parse(template, detection_string)
```

**Why it's useful**:

- 🎯 **Pattern matching**: Extract structured data from strings
- 🔧 **Type conversion**: Automatically convert to appropriate data types
- 📝 **Readable**: More intuitive than regular expressions for simple patterns
- ⚡ **Efficient**: Fast parsing for structured data extraction

---

## 🚫 Unused/Optional Packages

### Packages Listed but Not Used

These packages appear in `requirements.txt` but are not actively used in the codebase:

#### 📦 `matplotlib` - Static Plotting

**Status**: Not currently used
**Potential use**: Static chart generation for reports

```python
# Could be used for:
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(time, ball_speed)
plt.title('Ball Speed Over Time')
plt.savefig('ball_speed_chart.png')
```

#### 📦 `seaborn` - Statistical Visualization

**Status**: Not currently used  
**Potential use**: Statistical analysis and visualization

```python
# Could be used for:
import seaborn as sns

# Statistical relationships
sns.scatterplot(x='player_speed', y='court_coverage', data=stats_df)

# Distribution analysis
sns.histplot(data=ball_speeds, bins=30)
```

#### 📦 `ffmpeg` - Video Processing

**Status**: Not directly used
**Potential use**: Advanced video processing and format conversion

```bash
# Could be used for:
# Video format conversion
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4

# Extract frames
ffmpeg -i video.mp4 -vf fps=1 frames/frame_%04d.png
```

#### 📦 `kaleav` - Unknown Package

**Status**: Not used
**Action**: Can be removed from requirements.txt

---

## 🔧 Package Management Best Practices

### Installation Optimization

#### Core Installation (Minimal)

```bash
# Install only essential packages
pip install opencv-python ultralytics supervision pyyaml numpy
```

#### Full Installation (All Features)

```bash
# Install all packages for complete functionality
pip install -r requirements.txt
```

#### Development Installation

```bash
# Additional packages for development
pip install -r requirements.txt
pip install pytest black isort mypy
```

### Dependency Resolution

#### Version Compatibility

```python
# requirements.txt with version constraints
opencv-python>=4.8.0,<5.0.0
ultralytics>=8.0.0,<9.0.0
supervision>=0.16.0,<1.0.0
streamlit>=1.28.0,<2.0.0
plotly>=5.15.0,<6.0.0
```

#### Conditional Dependencies

```python
# Optional GPU support
torch>=2.0.0; sys_platform != "darwin"  # Linux/Windows
torch>=2.0.0,<2.1.0; sys_platform == "darwin"  # macOS

# Optional video processing
ffmpeg-python>=0.2.0; platform_system != "Windows"
```

### Performance Optimization

#### GPU Optimization

```python
# Check for CUDA availability
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name()}")
    device = 'cuda'
else:
    print("Using CPU - consider GPU for better performance")
    device = 'cpu'
```

#### Memory Management

```python
# Monitor package memory usage
import psutil
import gc

def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

# Clear memory when switching between large operations
del large_model
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

---

## 🚨 Troubleshooting Package Issues

### Common Installation Issues

#### Issue 1: OpenCV Installation Problems

```bash
# Symptoms: ImportError: No module named 'cv2'
# Solutions:
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python

# For additional features:
pip install opencv-contrib-python
```

#### Issue 2: CUDA/PyTorch Compatibility

```bash
# Symptoms: CUDA not available or version mismatch
# Check CUDA version:
nvidia-smi

# Install compatible PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Issue 3: Streamlit Port Conflicts

```bash
# Symptoms: "Port 8501 is already in use"
# Solutions:
streamlit run app.py --server.port 8502

# Or kill existing process:
lsof -ti:8501 | xargs kill -9  # Linux/Mac
```

### Performance Issues

#### Issue 1: Slow Video Processing

```python
# Check bottlenecks:
import time

start_time = time.time()
frame = cv2.imread('test.jpg')
print(f"Image load time: {time.time() - start_time:.3f}s")

# Optimize:
# 1. Use smaller image sizes
frame = cv2.resize(frame, (640, 480))

# 2. Reduce color channels if possible
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

#### Issue 2: High Memory Usage

```python
# Monitor memory usage:
def check_memory():
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    import psutil
    print(f"RAM usage: {psutil.virtual_memory().percent}%")

# Reduce memory usage:
# 1. Process in smaller batches
# 2. Use CPU for some operations
# 3. Clear cache regularly
```

---

## 📈 Future Package Considerations

### Potential Additions

#### Performance Enhancements

```python
# TensorRT for NVIDIA GPU optimization
# pip install tensorrt

# ONNX for cross-platform optimization  
# pip install onnxruntime-gpu

# Intel OpenVINO for Intel hardware
# pip install openvino
```

#### Advanced Analytics

```python
# Pandas for advanced data analysis
# pip install pandas

# Scikit-learn for machine learning
# pip install scikit-learn

# NetworkX for graph analysis (player interactions)
# pip install networkx
```

#### Enhanced Visualization

```python
# Bokeh for interactive web visualizations
# pip install bokeh

# Altair for statistical visualizations
# pip install altair

# Dash for more complex web applications
# pip install dash
```

### Package Cleanup Recommendations

#### Remove Unused Packages

```bash
# These can be safely removed if not used:
pip uninstall matplotlib seaborn ffmpeg kaleav
```

#### Reorganize Dependencies

```python
# Split requirements into categories:
# requirements-core.txt (essential packages)
# requirements-web.txt (web interface packages)  
# requirements-dev.txt (development packages)
# requirements-gpu.txt (GPU-specific packages)
```

---

This package guide should help you understand the role of each dependency in the Padel Analytics system and how to optimize their usage for better performance and reliability.
