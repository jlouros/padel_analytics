""" General configurations for main.py """

# Path to the input video file
INPUT_VIDEO_PATH = "./examples/videos/rally.mp4"

# Path to save the output video with inferences
OUTPUT_VIDEO_PATH = "results.mp4"

# Flag to indicate whether to collect 2D projection data
COLLECT_DATA = False
# Path to save the collected data
COLLECT_DATA_PATH = "data.csv"

# Maximum number of frames to be analyzed (None means all frames)
MAX_FRAMES = None

# Path to load fixed court keypoints (if available)
FIXED_COURT_KEYPOINTS_LOAD_PATH = "./cache/fixed_keypoints_detection.json"
# Path to save fixed court keypoints (set to None to disable saving)
FIXED_COURT_KEYPOINTS_SAVE_PATH = None  # "./cache/fixed_keypoints_detection.json"

# Configuration for players tracker
PLAYERS_TRACKER_MODEL = "./weights/players_detection/yolov8m.pt"  # Path to the model weights for player detection
PLAYERS_TRACKER_BATCH_SIZE = 8  # Batch size for processing frames
PLAYERS_TRACKER_ANNOTATOR = "rectangle_bounding_box"  # Type of annotation to use for displaying detections
PLAYERS_TRACKER_LOAD_PATH = "./cache/players_detections.json"  # Path to load player detections (if available)
PLAYERS_TRACKER_SAVE_PATH = "./cache/players_detections.json"  # Path to save player detections

# Configuration for players keypoints tracker
PLAYERS_KEYPOINTS_TRACKER_MODEL = "./weights/players_keypoints_detection/best.pt"  # Path to the model weights for player keypoints detection
PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE = 1280  # Training image size for the model
PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE = 8  # Batch size for processing frames
PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH = "./cache/players_keypoints_detections.json"  # Path to load player keypoints detections (if available)
PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH = "./cache/players_keypoints_detections.json"  # Path to save player keypoints detections

# Configuration for ball tracker
BALL_TRACKER_MODEL = "./weights/ball_detection/TrackNet_best.pt"  # Path to the model weights for ball detection
BALL_TRACKER_INPAINT_MODEL = "./weights/ball_detection/InpaintNet_best.pt"  # Path to the model weights for inpainting missing ball detections
BALL_TRACKER_BATCH_SIZE = 8  # Batch size for processing frames
BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM = 400  # Maximum number of samples for median filtering
BALL_TRACKER_LOAD_PATH = "./cache/ball_detections.json"  # Path to load ball detections (if available)
BALL_TRACKER_SAVE_PATH = "./cache/ball_detections.json"  # Path to save ball detections

# Configuration for court keypoints tracker
KEYPOINTS_TRACKER_MODEL = "./weights/court_keypoints_detection/best.pt"  # Path to the model weights for court keypoints detection
KEYPOINTS_TRACKER_BATCH_SIZE = 8  # Batch size for processing frames
KEYPOINTS_TRACKER_MODEL_TYPE = "yolo"  # Type of model used for keypoints detection
KEYPOINTS_TRACKER_LOAD_PATH = None  # Path to load court keypoints detections (if available)
KEYPOINTS_TRACKER_SAVE_PATH = None  # Path to save court keypoints detections (set to None to disable saving)

