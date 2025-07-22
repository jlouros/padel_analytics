# Testing Guide

Since there are no dedicated test files in the repository, the best way to test the project is by running it with a sample video and observing the output. You can use the `rally.mp4` video in the `examples/videos/` folder to check if the analysis works as expected. If the program runs without errors and generates visualizations, you can be confident that the core functionality is working correctly.

## Example 1: Testing with a Different Video

1. **Download a new video:** Find a new video of a padel game and download it to the `examples/videos/` directory.
2. **Update the config:** Open the `config.py` file and update the `INPUT_VIDEO_PATH` variable to point to your new video.
3. **Run the analysis:** Run the `main.py` script and observe the output.
   ```bash
   python main.py
   ```
   Does the analysis complete without errors? Are the players and ball tracked correctly?

## Example 2: Testing with Different Model Weights

1. **Download new model weights:** Find a new set of YOLOv8 model weights and download them to the `models` directory.
2. **Update the config:** Open the `config.py` file and update the paths to your new model weights.
3. **Run the analysis:** Run the `main.py` script and observe the output.
   ```bash
   python main.py
   ```
   Does the analysis complete without errors? Are the players and ball tracked correctly?

## Example 3: Testing with Different Court Dimensions

1. **Update the court dimensions:** Open the `constants/court_dimensions.py` file and update the court dimensions to match a different court.
2. **Run the analysis:** Run the `main.py` script and observe the output.
   ```bash
   python main.py
   ```
   Does the analysis complete without errors? Is the 2D projection of the court correct?

## Example 4: Testing with a Different Camera Angle

1. **Find a video with a different camera angle:** Find a video of a padel game with a different camera angle and download it to the `examples/videos/` directory.
2. **Update the config:** Open the `config.py` file and update the `INPUT_VIDEO_PATH` variable to point to your new video.
3. **Run the analysis:** Run the `main.py` script and observe the output.
   ```bash
   python main.py
   ```
   Does the analysis complete without errors? Are the players and ball tracked correctly?

## Example 5: Testing with a Low-Quality Video

1. **Find a low-quality video:** Find a low-quality video of a padel game and download it to the `examples/videos/` directory.
2. **Update the config:** Open the `config.py` file and update the `INPUT_VIDEO_PATH` variable to point to your new video.
3. **Run the analysis:** Run the `main.py` script and observe the output.
   ```bash
   python main.py
   ```
   Does the analysis complete without errors? Are the players and ball tracked correctly?

## Example 6: Testing with a Partial Court View

1. **Find a video with a partial court view:** Find a video of a padel game where the entire court is not visible and download it to the `examples/videos/` directory.
2. **Update the config:** Open the `config.py` file and update the `INPUT_VIDEO_PATH` variable to point to your new video.
3. **Run the analysis:** Run the `main.py` script and observe the output.
   ```bash
   python main.py
   ```
   Does the analysis complete without errors? Are the players and ball tracked correctly?

### Resolving Issues with a Partial Court View

If the analysis fails or the results are inaccurate, it is likely because the model is not able to detect all of the key points on the court. To resolve this issue, you can try the following:

- **Manually annotate the key points:** You can manually annotate the key points in the video and use them to train a new model.
- **Use a different model:** You can try using a different model that is better suited for partial court views.

## Example 7: Testing with a 0.5 Lens

1. **Find a video with a 0.5 lens:** Find a video of a padel game that was filmed with a 0.5 lens and download it to the `examples/videos/` directory.
2. **Update the config:** Open the `config.py` file and update the `INPUT_VIDEO_PATH` variable to point to your new video.
3. **Run the analysis:** Run the `main.py` script and observe the output.
   ```bash
   python main.py
   ```
   Does the analysis complete without errors? Are the players and ball tracked correctly?

### Resolving Issues with a 0.5 Lens

If the analysis fails or the results are inaccurate, it is likely because the model is not able to correct for the distortion caused by the lens. To resolve this issue, you can try the following:

- **Undistort the video:** You can use a tool like FFmpeg to undistort the video before running the analysis.
  ```bash
  ffmpeg -i input.mp4 -vf "lenscorrection=cx=0.5:cy=0.5:k1=-0.2:k2=-0.05" output.mp4
  ```
- **Use a different model:** You can try using a different model that is better suited for distorted images.

## Example 8: Testing with Different Frame Rates

1. **Find videos with different frame rates:** Find videos of padel games with different frame rates (30, 60, 90, and 120 fps) and download them to the `examples/videos/` directory.
2. **Update the config:** Open the `config.py` file and update the `INPUT_VIDEO_PATH` variable to point to your new video.
3. **Run the analysis:** Run the `main.py` script and observe the output.
   ```bash
   python main.py
   ```
   Does the analysis complete without errors? Are the players and ball tracked correctly?
