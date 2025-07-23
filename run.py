import os
import sys
import shutil
import subprocess
import requests
import argparse
import logging
import tempfile
import re
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('padel_analytics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PadelAnalyticsRunner:
    """
    Enhanced runner class for the Padel Analytics project with improved error handling,
    logging, and user experience.
    """
    
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    def __init__(self, video_path=None, auto_clean=False, max_frames=None):
        self.video_path = video_path
        self.auto_clean = auto_clean
        self.max_frames = max_frames
        self.config_backup_path = None
        
    def validate_video_file(self, file_path):
        """Validate if the video file is accessible and has a supported format."""
        try:
            if not os.path.exists(file_path):
                logger.error(f"Video file does not exist: {file_path}")
                return False
                
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.SUPPORTED_VIDEO_FORMATS:
                logger.warning(f"Video format {file_ext} may not be supported. Supported formats: {self.SUPPORTED_VIDEO_FORMATS}")
                
            # Try to open with OpenCV to validate
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video file: {file_path}")
                return False
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            logger.info(f"Video validated: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s duration")
            return True
            
        except Exception as e:
            logger.error(f"Error validating video file: {e}")
            return False
    
    def download_video_with_progress(self, url, destination):
        """Download video with progress bar and proper error handling."""
        try:
            logger.info(f"Starting download from: {url}")
            
            # Get file size for progress bar
            response = requests.head(url, allow_redirects=True, timeout=10)
            total_size = int(response.headers.get('content-length', 0))
            
            # Start download
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            with open(destination, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Download completed: {destination}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during download: {e}")
            return False
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return False
    
    def backup_config(self):
        """Create a backup of the current config file."""
        try:
            self.config_backup_path = "config.py.backup"
            shutil.copy2("config.py", self.config_backup_path)
            logger.info("Config file backed up")
        except Exception as e:
            logger.error(f"Failed to backup config: {e}")
    
    def restore_config(self):
        """Restore the config file from backup."""
        try:
            if self.config_backup_path and os.path.exists(self.config_backup_path):
                shutil.copy2(self.config_backup_path, "config.py")
                os.remove(self.config_backup_path)
                logger.info("Config file restored from backup")
        except Exception as e:
            logger.error(f"Failed to restore config: {e}")
    
    def update_config_file(self, video_path):
        """Update config file with new video path using regex replacement."""
        try:
            self.backup_config()
            
            with open("config.py", "r") as f:
                content = f.read()
            
            # Use regex to replace the entire INPUT_VIDEO_PATH section
            # This pattern matches the if-else block for INPUT_VIDEO_PATH
            pattern = r'(# Input video path\s*\n)(if VIDEO_TYPE:.*?\n.*?INPUT_VIDEO_PATH.*?\n.*?else:\s*\n.*?INPUT_VIDEO_PATH.*?\n)'
            replacement = f'\\1INPUT_VIDEO_PATH = "{video_path}"\n'
            
            new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            
            # If the regex didn't match (maybe file structure changed), try simple replacement
            if new_content == content:
                # Look for any line with INPUT_VIDEO_PATH and replace it
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'INPUT_VIDEO_PATH' in line and not line.strip().startswith('#'):
                        # Replace this line with our assignment
                        lines[i] = f'INPUT_VIDEO_PATH = "{video_path}"'
                        break
                new_content = '\n'.join(lines)
            
            with open("config.py", "w") as f:
                f.write(new_content)
            
            logger.info("Config file updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            self.restore_config()
            return False
    
    def clean_previous_results(self, interactive=True):
        """Clean previous results and cache with optional user confirmation."""
        keypoints_file = "./cache/fixed_keypoints_detection.json"
        keypoints_backup = None
        
        if interactive and not self.auto_clean:
            response = input("Do you want to delete all previous results and cache? (y/n): ").lower()
            if response != 'y':
                logger.info("Keeping previous results")
                return
            
            # Ask about preserving keypoints
            if os.path.exists(keypoints_file):
                preserve_keypoints = input("Preserve court keypoints selection? (y/n): ").lower()
                if preserve_keypoints == 'y':
                    keypoints_backup = keypoints_file + ".backup"
                    shutil.copy2(keypoints_file, keypoints_backup)
                    logger.info("Court keypoints backed up")
        
        try:
            for directory in ["cache", "results"]:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                    logger.info(f"Deleted {directory} directory")
            
            # Restore keypoints if backed up
            if keypoints_backup and os.path.exists(keypoints_backup):
                os.makedirs("cache", exist_ok=True)
                shutil.copy2(keypoints_backup, keypoints_file)
                os.remove(keypoints_backup)
                logger.info("Court keypoints restored")
            
            # Also clean any .log files
            for log_file in Path(".").glob("*.log"):
                if log_file.name != "padel_analytics.log":  # Keep current log
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file}")
                    
        except Exception as e:
            logger.error(f"Error cleaning previous results: {e}")
    
    def get_video_path_interactive(self):
        """Interactive method to get video path from user."""
        while True:
            video_input = input("\nEnter the path to your video file (local path or URL): ").strip()
            
            if not video_input:
                print("Please enter a valid path or URL")
                continue
                
            if video_input.startswith(('http://', 'https://')):
                return self.handle_video_url(video_input)
            else:
                if self.validate_video_file(video_input):
                    return video_input
                else:
                    retry = input("Video validation failed. Try another file? (y/n): ").lower()
                    if retry != 'y':
                        return None
    
    def handle_video_url(self, url):
        """Handle video URL download."""
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                logger.error("Invalid URL format")
                return None
            
            # Generate filename from URL or use default
            filename = Path(parsed_url.path).name
            if not filename or not Path(filename).suffix:
                filename = "downloaded_video.mp4"
            
            video_dir = Path("examples") / "videos"
            video_file_path = video_dir / filename
            
            # Check if already downloaded
            if video_file_path.exists():
                redownload = input(f"Video {filename} already exists. Re-download? (y/n): ").lower()
                if redownload != 'y':
                    if self.validate_video_file(str(video_file_path)):
                        return str(video_file_path)
                    else:
                        logger.info("Existing file is invalid, downloading fresh copy...")
            
            if self.download_video_with_progress(url, str(video_file_path)):
                if self.validate_video_file(str(video_file_path)):
                    return str(video_file_path)
                else:
                    logger.error("Downloaded video is invalid")
                    video_file_path.unlink(missing_ok=True)
                    return None
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error handling video URL: {e}")
            return None
    
    def check_prerequisites(self):
        """Check if all prerequisites are met before running analysis."""
        try:
            # Check if keypoints file exists
            keypoints_file = "./cache/fixed_keypoints_detection.json"
            source_keypoints = "./source_keypoints.json"
            
            if not os.path.exists(keypoints_file):
                logger.warning("Court keypoints not found in cache")
                
                if os.path.exists(source_keypoints):
                    if self.auto_clean:
                        # In auto mode, automatically copy source keypoints if available
                        os.makedirs("cache", exist_ok=True)
                        shutil.copy2(source_keypoints, keypoints_file)
                        logger.info("Source keypoints automatically copied to cache")
                        return True
                    else:
                        copy_source = input(f"Copy keypoints from {source_keypoints}? (y/n): ").lower()
                        if copy_source == 'y':
                            os.makedirs("cache", exist_ok=True)
                            shutil.copy2(source_keypoints, keypoints_file)
                            logger.info("Source keypoints copied to cache")
                            return True
                
                if self.auto_clean:
                    logger.error("Auto-clean mode enabled but no source keypoints found. Cannot proceed.")
                    logger.error("Please run without --auto-clean to manually select keypoints.")
                    return False
                
                logger.warning("Court keypoints will need to be selected manually")
                logger.warning("The main.py script will open an interactive window for keypoint selection")
                proceed = input("Continue with manual keypoint selection? (y/n): ").lower()
                return proceed == 'y'
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking prerequisites: {e}")
            return False
    
    def run_analysis(self):
        """Run the main analysis with proper error handling."""
        try:
            logger.info("Starting Padel Analytics...")
            result = subprocess.run(
                [sys.executable, "main.py"], 
                capture_output=True, 
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("Analysis completed successfully")
                if result.stdout:
                    print("Analysis output:")
                    print(result.stdout)
            else:
                logger.error(f"Analysis failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
                return False
                
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Analysis timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"Error running analysis: {e}")
            return False
        """Run the main analysis with proper error handling."""
        try:
            logger.info("Starting Padel Analytics...")
            result = subprocess.run(
                [sys.executable, "main.py"], 
                capture_output=True, 
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("Analysis completed successfully")
                if result.stdout:
                    print("Analysis output:")
                    print(result.stdout)
            else:
                logger.error(f"Analysis failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
                return False
                
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Analysis timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"Error running analysis: {e}")
            return False
    
    def run(self):
        """Main execution method."""
        try:
            logger.info("=== Padel Analytics Runner Started ===")
            
            # Step 1: Clean previous results
            self.clean_previous_results(interactive=not self.auto_clean)
            
            # Step 2: Get video path
            if not self.video_path:
                self.video_path = self.get_video_path_interactive()
                if not self.video_path:
                    logger.error("No valid video path provided. Exiting.")
                    return False
            else:
                # Validate provided video path
                if self.video_path.startswith(('http://', 'https://')):
                    self.video_path = self.handle_video_url(self.video_path)
                    if not self.video_path:
                        return False
                elif not self.validate_video_file(self.video_path):
                    logger.error(f"Provided video path is invalid: {self.video_path}")
                    return False
            
            # Step 3: Check prerequisites (keypoints, etc.)
            if not self.check_prerequisites():
                logger.error("Prerequisites not met. Exiting.")
                return False
            
            # Step 4: Update config
            if not self.update_config_file(self.video_path):
                return False
            
            # Step 5: Run analysis
            success = self.run_analysis()
            
            # Step 6: Cleanup
            self.restore_config()
            
            if success:
                logger.info("=== Padel Analytics completed successfully ===")
                print(f"\nResults should be available in: {os.path.abspath('results.mp4')}")
            else:
                logger.error("=== Padel Analytics failed ===")
            
            return success
            
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
            self.restore_config()
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.restore_config()
            return False

def main():
    """
    Enhanced main function with command-line argument support and improved user experience.
    """
    parser = argparse.ArgumentParser(
        description="Padel Analytics - Computer vision system for analyzing padel game recordings"
    )
    parser.add_argument(
        "--video", "-v",
        help="Path to video file or URL to analyze"
    )
    parser.add_argument(
        "--auto-clean", "-c",
        action="store_true",
        help="Automatically clean previous results without prompting"
    )
    parser.add_argument(
        "--max-frames", "-f",
        type=int,
        help="Maximum number of frames to analyze"
    )
    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run the analytics runner
    runner = PadelAnalyticsRunner(
        video_path=args.video,
        auto_clean=args.auto_clean,
        max_frames=args.max_frames
    )
    
    success = runner.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
