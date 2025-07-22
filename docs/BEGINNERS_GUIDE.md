# Beginner's Guide

This guide will walk you through the process of using the Padel Analytics project with your own video file.

#### Step 1: Set Up Your Environment

1. **Install Python:** Make sure you have Python 3.12 installed on your system.
2. **Create a Virtual Environment:** Open a terminal and run the following commands:
   ```bash
   conda create -n python=3.12 padel_analytics pip
   conda activate padel_analytics
   ```
3. **Install Dependencies:** Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

#### Step 2: Download Model Weights

1. **Download:** Download the model weights from the link provided in the `README.md` file.
2. **Configure:** Open the `config.py` file and update the paths to the model weights you just downloaded.

#### Step 3: Prepare Your Video

1. **Place Your Video:** Place your video file in the `examples/videos/` directory.
2. **Update Config:** Open the `config.py` file and update the `VIDEO_PATH` variable to point to your video file.

#### Step 4: Run the Analysis

1. **Run the Script:** Open a terminal and run the following command:
   ```bash
   python main.py
   ```
2. **Select Key Points:** A window will pop up, showing the first frame of your video. You will need to select 12 key points on the court in the correct order. Refer to the diagram in `main.py` for the correct order.
3. **Wait for the Analysis:** Once you have selected the key points, the analysis will begin. This may take some time, depending on the length of your video and the power of your computer.
4. **View the Results:** Once the analysis is complete, the results will be saved to the `output` directory. You will find a video with the analysis overlayed, as well as various graphs and charts.
