# 🚀 Beginner's Guide to Padel Analytics

Welcome! This guide will take you from zero to running your first padel game analysis. Don't worry if you're new to computer vision or AI - we'll explain everything step by step.

## 🎯 What You'll Accomplish

By the end of this guide, you'll be able to:

- ✅ Set up the complete development environment
- ✅ Download and configure AI models
- ✅ Run analysis on your own padel videos
- ✅ Understand what each step does and why
- ✅ Troubleshoot common issues

## 📋 Before We Start

### What You Need to Know

- **Basic Python**: You should know what functions and variables are
- **Command Line Basics**: How to navigate folders and run commands
- **File Management**: Creating folders, downloading files, editing text files

### What You Need on Your Computer

- **Python 3.12** (we'll help you install this)
- **At least 8GB RAM** (for AI model processing)
- **5GB free disk space** (for models and example videos)
- **Internet connection** (for downloading models and dependencies)

### Time Commitment

- **Initial setup**: 30-45 minutes
- **First analysis**: 15-20 minutes
- **Understanding the results**: 10-15 minutes

## 🛠️ Step-by-Step Setup Guide

### Step 1: Install Python and Set Up Your Environment

#### 1.1 Install Python 3.12

**Windows/Mac Users:**

1. Go to [python.org](https://python.org/downloads/)
2. Download Python 3.12 (latest version)
3. Run the installer
4. ⚠️ **Important**: Check "Add Python to PATH" during installation

**Linux Users:**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip

# Other distributions - use your package manager
```

#### 1.2 Verify Installation

Open a terminal/command prompt and type:

```bash
python --version
```

You should see: `Python 3.12.x`

If you see an older version or get an error, you might need to use `python3` instead of `python`.

#### 1.3 Create a Virtual Environment

Think of a virtual environment as a **separate workspace** for this project - it keeps all the dependencies organized and prevents conflicts.

```bash
# Navigate to where you want to store the project
cd /path/to/your/projects

# Create the virtual environment
python -m venv padel_analytics_env

# Activate it
# On Windows:
padel_analytics_env\Scripts\activate

# On Mac/Linux:
source padel_analytics_env/bin/activate
```

🎉 **Success indicator**: Your terminal prompt should now start with `(padel_analytics_env)`

### Step 2: Get the Project Code

#### 2.1 Download the Repository

```bash
# If you have git installed:
git clone https://github.com/your-repo/padel_analytics.git
cd padel_analytics

# If you don't have git:
# Download the ZIP file from GitHub and extract it
```

#### 2.2 Install Required Libraries

This step downloads all the AI libraries and tools the project needs:

```bash
pip install -r requirements.txt
```

⏰ **This might take 5-10 minutes** - it's downloading large AI libraries like PyTorch.

**Common Issue**: If you get permission errors, try:

```bash
pip install --user -r requirements.txt
```

### Step 3: Set Up Your Development Environment (VSCode Users)

If you're using Visual Studio Code, these extensions will make your life easier:

#### Essential Extensions

1. **Python** (ms-python.python)
   - Provides Python language support
   - Enables debugging and code completion

2. **Pylance** (ms-python.vscode-pylance)
   - Advanced Python language features
   - Better error detection and code suggestions

#### 3.1 Select Python Interpreter

1. Open VSCode in the project folder: `code .`
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Type "Python: Select Interpreter"
4. Choose the interpreter from your `padel_analytics_env` environment

### Step 4: Download AI Model Weights

The AI models need pre-trained "weights" - think of these as the model's learned knowledge.

#### 4.1 Download the Models

1. Go to the link provided in the main `README.md` file
2. Download the model weights zip file (usually 1-2 GB)
3. Extract the contents to the `weights/` folder in your project

Your folder structure should look like:

```none
padel_analytics/
├── weights/
│   ├── ball_detection/
│   │   └── ball_model.pt
│   ├── players_detection/
│   │   └── players_model.pt
│   ├── court_keypoints_detection/
│   │   └── keypoints_model.pt
│   └── players_keypoints_detection/
│       └── player_keypoints_model.pt
```

#### 4.2 Configure Model Paths

Open `config.py` and update the paths to match your downloaded models:

```python
# Find these lines in config.py and update them:
BALL_TRACKER_MODEL = "weights/ball_detection/ball_model.pt"
PLAYERS_TRACKER_MODEL = "weights/players_detection/players_model.pt"
KEYPOINTS_TRACKER_MODEL = "weights/court_keypoints_detection/keypoints_model.pt"
PLAYERS_KEYPOINTS_TRACKER_MODEL = "weights/players_keypoints_detection/player_keypoints_model.pt"
```

### Step 5: Prepare Your First Video

#### 5.1 Choose a Video

For your first attempt, use the provided example video:

- File: `examples/videos/rally.mp4`
- This video is already optimized for the system

#### 5.2 Update Configuration

Open `config.py` and find the `INPUT_VIDEO_PATH` setting:

```python
# Update this line:
INPUT_VIDEO_PATH = "examples/videos/rally.mp4"
```

## 🎬 Running Your First Analysis

### Step 1: Start the Analysis

```bash
python main.py
```

### Step 2: Select Court Keypoints

A window will appear showing the first frame of your video. You need to click on **12 specific points** on the court in the correct order.

#### Understanding Court Keypoints

The court keypoints define the court's geometry. Here's the numbering system:

```none
    Back Wall
k11────────────────k12
│                   │
k8──────k9─────────k10  ← Service Line
│       │           │
k6─────────────────k7   ← Net Line  
│       │           │
k3──────k4─────────k5   ← Service Line
│                   │
k1─────────────────k2
    Front Wall
```

#### Selection Tips

1. **Start with corners**: k1, k2, k11, k12 are usually easier to spot
2. **Look for line intersections**: Where court lines meet are the keypoints
3. **Be precise**: Click exactly on the intersection points
4. **Take your time**: Accurate keypoints = better analysis results

**What happens next**: The system saves these keypoints so you won't need to select them again for this court angle.

### Step 3: Watch the Magic Happen

Once you've selected all 12 points:

1. The analysis starts automatically
2. You'll see progress messages in the terminal
3. The system processes each frame, detecting:
   - Ball position
   - Player positions
   - Court features
   - Player movements

⏰ **Expected time**: 2-5 minutes for a 30-second video (depends on your computer's power)

### Step 4: View Your Results

After processing completes, you'll find results in the `output/` folder:

- **Annotated video**: Shows tracking overlays on the original video
- **Analytics charts**: Graphs of player movement, ball speed, etc.
- **Court projection**: 2D top-down view of player positions

## 🎉 Understanding Your Results

### What the System Detected

#### Ball Tracking

- **Green dot**: Current ball position
- **Trail**: Ball's path over recent frames
- **Speed indicator**: Current ball velocity

#### Player Tracking

- **Colored boxes**: Each player gets a unique color
- **Player trails**: Shows movement patterns
- **Pose estimation**: Key body points for movement analysis

#### Court Analysis

- **Heatmaps**: Where players spend most time
- **Movement patterns**: Common paths players take
- **Game statistics**: Rally length, court coverage, etc.

### Reading the Analytics

#### Player Heatmaps

- **Red areas**: Where the player spent most time
- **Blue areas**: Rarely visited spots
- **Use this to**: Understand playing style and court positioning

#### Speed Analysis

- **Ball speed graph**: Shows ball velocity throughout the rally
- **Player speed**: How fast players move around the court
- **Use this to**: Analyze game intensity and player fitness

## 🔧 Troubleshooting Common Issues

### Issue 1: "ModuleNotFoundError"

**Symptoms**: Error when running `python main.py`
**Solution**:

```bash
# Make sure your virtual environment is activated
source padel_analytics_env/bin/activate  # Mac/Linux
# or
padel_analytics_env\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Issue 2: Keypoint Selection Window Doesn't Appear

**Symptoms**: Script runs but no window shows up
**Solution**:

```bash
# Install GUI backend
pip install PyQt5
# or
pip install tkinter
```

### Issue 3: "CUDA Out of Memory" Error

**Symptoms**: Error during processing, mentions GPU memory
**Solution**: Open `config.py` and reduce batch sizes:

```python
# Change these values:
PLAYERS_TRACKER_BATCH_SIZE = 8    # Instead of 16
BALL_TRACKER_BATCH_SIZE = 8       # Instead of 16
```

### Issue 4: Model Files Not Found

**Symptoms**: "FileNotFoundError" mentioning .pt files
**Solution**:

1. Verify model files are in the `weights/` folder
2. Check the paths in `config.py` match your file locations
3. Ensure you extracted the full weights archive

### Issue 5: Poor Tracking Quality

**Symptoms**: Ball or players not detected properly
**Possible causes and solutions**:

- **Video quality too low**: Try with a higher resolution video
- **Camera angle too extreme**: Works best with side-view angles
- **Lighting conditions**: Avoid videos with heavy shadows or glare
- **Court visibility**: Ensure the full court is visible in the frame

## 🎓 Next Steps

### Try Different Videos

1. Use your own padel videos
2. Update `INPUT_VIDEO_PATH` in `config.py`
3. Select new keypoints for each different court/angle

### Explore the Streamlit Dashboard

```bash
streamlit run app.py
```

This opens a web interface for interactive analysis.

### Learn More About the System

- Read `docs/ARCHITECTURE.md` to understand how components work together
- Check `docs/COMPONENTS.md` for detailed module descriptions
- Explore `docs/TESTING.md` to learn about testing different scenarios

## 🤝 Getting Help

### Common Resources

- **Project documentation**: All files in the `docs/` folder
- **Example videos**: `examples/videos/` folder has test cases
- **Configuration reference**: `config.py` has comments explaining each setting

### When Something Goes Wrong

1. **Check the terminal output**: Error messages usually explain what's wrong
2. **Verify your setup**: Make sure all steps in this guide were completed
3. **Try the example video first**: Before using your own videos
4. **Check file paths**: Many issues are due to incorrect file locations

### Community and Support

- **GitHub Issues**: Report bugs or ask questions
- **Documentation**: This guide and other docs in the `docs/` folder
- **Example outputs**: Compare your results with expected outputs

---

🎊 **Congratulations!** You've successfully set up and run your first padel game analysis. You're now ready to explore the fascinating world of sports analytics and computer vision!
