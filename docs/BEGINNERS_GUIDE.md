# Beginner's Guide

This guide will walk you through the process of using the Padel Analytics project with your own video file.

## For VSCode Users

If you are using Visual Studio Code, we recommend installing the following extensions to improve your development experience:

- **Python:** The official Python extension for VSCode, providing features like linting, debugging, and IntelliSense.
- **Pylance:** An extension that provides high-performance language support for Python.
- **Jupyter:** An extension that allows you to work with Jupyter Notebooks in VSCode.
- **GitLens:** This extension supercharges the Git capabilities built into Visual Studio Code. It helps you to visualize code authorship at a glance via Git blame annotations and code lens, seamlessly navigate and explore Git repositories, gain valuable insights via powerful comparison commands, and so much more.
- **Markdown All in One:** This extension provides a wide range of features for editing Markdown documents, including live preview, table of contents, and auto-completion.

## Step 1: Set Up Your Environment

1. **Install Python:** Make sure you have Python 3.12 installed on your system. You can download it from the official Python website.
2. **Create a Virtual Environment:** Open a terminal in VSCode (View > Terminal) and run the following commands:
   ```bash
   conda create -n python=3.12 padel_analytics pip
   conda activate padel_analytics
   ```
3. **Select the Interpreter:** Open the Command Palette (Ctrl+Shift+P) and type "Python: Select Interpreter". Choose the `padel_analytics` environment you just created.
4. **Install Dependencies:** Install the required libraries by running the following command in the terminal:
   ```bash
   pip install -r requirements.txt
   ```

## Step 2: Download Model Weights

1. **Download:** Download the model weights from the link provided in the `README.md` file.
2. **Configure:** Open the `config.py` file and update the paths to the model weights you just downloaded.

## Step 3: Prepare Your Video

1. **Place Your Video:** Place your video file in the `examples/videos/` directory.
2. **Update Config:** Open the `config.py` file and update the `VIDEO_PATH` variable to point to your video file.

## Step 4: Run the Analysis

1. **Run the Script:** Open a terminal in VSCode and run the following command:
   ```bash
   python main.py
   ```
2. **Select Key Points:** A window will pop up, showing the first frame of your video. You will need to select 12 key points on the court in the correct order. Refer to the diagram in `main.py` for the correct order.
3. **Wait for the Analysis:** Once you have selected the key points, the analysis will begin. This may take some time, depending on the length of your video and the power of your computer.
4. **View the Results:** Once the analysis is complete, the results will be saved to the `output` directory. You will find a video with the analysis overlayed, as well as various graphs and charts.
