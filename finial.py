import cv2
import numpy as np
import os

# Set the path to the directory containing the original images
original_path = "C:/Users/psraa/Desktop/Internship/task3/Video_Testframes"

# Set the path to the directory where the normalized grayscale images will be saved
gray_path = "C:/Users/psraa/Desktop/Internship/task3/greyscale_flower_with_sequences"

# Convert images to grayscale and normalize
for filename in os.listdir(original_path):
    if filename.endswith(".jpg"):
        # Load the image using OpenCV
        img = cv2.imread(os.path.join(original_path, filename))
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize the grayscale image
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Save the normalized grayscale image to disk
        output_filename = os.path.join(gray_path, filename)
        cv2.imwrite(output_filename, normalized)
        

# Set the path to the directory containing the normalized grayscale images
path = "C:/Users/psraa/Desktop/Internship/task3/greyscale_flower_with_sequences"

# Set the path to the painting image
painting_path = "C:/Users/psraa/Desktop/Internship/task3/Video_Testframes/image_30.jpg"

# Define the parameters for the Dense Optical Flow algorithm
params = {
    "pyr_scale": 0.2,
    "levels": 2,
    "winsize": 15,
    "iterations": 1,
    "poly_n": 3,
    "poly_sigma": 1.2,
    "flags": 0
}

# Initialize the previous frame as None
prev_frame = None

# Load the painting image
painting = cv2.imread(painting_path)



# Initialize the warped image as a copy of the painting image
warped = painting.copy()

# Create a VideoWriter object to save the animated image as an MP4 video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 5
output_video = cv2.VideoWriter('animated_image.mp4', fourcc, fps, (warped.shape[1], warped.shape[0]))

# Loop through all images in the directory
for filename in os.listdir(path):
    if filename.endswith(".jpg"):
        # Load the image using OpenCV
        curr_frame = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        # Load the painting image and resize it to match the size of the sequence images
        painting = cv2.imread(painting_path)
        painting = cv2.resize(painting, (curr_frame.shape[1], curr_frame.shape[0]))
        
        # If this is not the first frame, calculate the optical flow
        if prev_frame is not None:
            # Check that the current and previous frames have the same size and number of channels
            if curr_frame.shape != prev_frame.shape or curr_frame.ndim != prev_frame.ndim:
                continue
            
            # Calculate the optical flow using the current and previous frames
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, **params)
            
            # Resize the flow array to match the size of the warped image
            flow_resized = cv2.resize(flow, (warped.shape[1], warped.shape[0]))
            
            # Apply the motion vectors to the painting image
            x, y = np.meshgrid(np.arange(warped.shape[1]), np.arange(warped.shape[0]))
            map_x = (x + flow_resized[..., 0]).astype(np.float32)
            map_y = (y + flow_resized[..., 1]).astype(np.float32)
            warped = cv2.remap(painting, map_x, map_y, cv2.INTER_LINEAR)
            
            # Write the warped image to the video file
            output_video.write(warped)
            painting = os.path.join("C:/Users/psraa/Desktop/Internship/task3/optical_flow", "flow_" + filename)
            cv2.imwrite(painting, warped * 255)
            
            # Save the animated image
            img_filename = os.path.splitext(filename)[0] + ".png"
            img_path = os.path.join("C:/Users/psraa/Desktop/Internship/task3/animated_image/", img_filename)
            cv2.imwrite(img_path, warped)
        
        # Set the current frame as the previous frame for the next iteration
        prev_frame = curr_frame

# Release the VideoWriter object
output_video.release()

print("Animation complete!")