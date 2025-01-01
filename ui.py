import cv2  # OpenCV library for image and video processing
import json  # Used for saving keypoints data in JSON format
from utils import read_video  # Custom utility function to read video frames

KEYPOINTS = []

# Function to handle mouse click events and display the coordinates of the points clicked on the image
def click_event(event, x, y, flags, params): 
    # Checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        # Appending the coordinates to the KEYPOINTS list
        KEYPOINTS.append((x, y))
        # Displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2) 
        cv2.imshow('image', img) 

if __name__ == "__main__":
    # Path to the input video file
    input_video_path = "./videos/trimmed_padel.mp4"
    # Alternative input video path (commented out)
    # input_video_path = "./videos/FINAL A1PADEL MARBELLA MASTER  Tolito Aguirre  Alfonso vs Allemandi  Pereyra HIGHLIGHTS.mp4"
    frames, fps, w, h = read_video(input_video_path, max_frames=10)

    img = frames[0]
    cv2.imshow('image', img)

    # Setting mouse handler for the image and calling the click_event() function 
    cv2.setMouseCallback('image', click_event) 

    # Wait for a key to be pressed to exit 
    cv2.waitKey(0) 

    # Save the keypoints to a JSON file
    with open("source_keypoints.json", "w") as f:
        json.dump(KEYPOINTS, f)
  
    # Close the window 
    cv2.destroyAllWindows()