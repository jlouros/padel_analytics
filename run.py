import os
import shutil
import requests

def main():
    """
    This script streamlines the user experience for running the Padel Analytics project.
    """

    # 1. Delete previous results and cache
    if input("Do you want to delete all previous results and cache? (y/n): ").lower() == "y":
        if os.path.exists("cache"):
            shutil.rmtree("cache")
        if os.path.exists("results"):
            shutil.rmtree("results")
        print("Previous results and cache deleted.")

    # 2. Get video path
    video_path = input("Enter the path to your video file (local path or URL): ")
    if video_path.startswith("http"):
        print("Downloading video...")
        r = requests.get(video_path)
        with open("examples/videos/video.mp4", "wb") as f:
            f.write(r.content)
        video_path = "examples/videos/video.mp4"
        print("Video downloaded.")

    # 3. Update config
    with open("config.py", "r") as f:
        lines = f.readlines()
    with open("config.py", "w") as f:
        for line in lines:
            if line.startswith("INPUT_VIDEO_PATH"):
                f.write(f"INPUT_VIDEO_PATH = \"{video_path}\"\n")
            else:
                f.write(line)
    print("Config file updated.")

    # 4. Run analysis
    os.system("python main.py")

if __name__ == "__main__":
    main()
