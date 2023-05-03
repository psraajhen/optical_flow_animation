# Optical Flow Animation

This code creates an optical flow animation by applying the dense optical flow algorithm to a sequence of grayscale images and warping a painting image based on the calculated motion vectors. The resulting images are saved as PNG files and combined into an MP4 video.

## Prerequisites

This code requires the following libraries to be installed:

- OpenCV (cv2)
- NumPy

## Usage

1. Set the path to the directory containing the original images in the `original_path` variable.
2. Set the path to the directory where the normalized grayscale images will be saved in the `gray_path` variable.
3. Run the code to convert the original images to grayscale and normalize them.
4. Set the path to the directory containing the normalized grayscale images in the `path` variable.
5. Set the path to the painting image in the `painting_path` variable.
6. Define the parameters for the Dense Optical Flow algorithm in the `params` dictionary.
7. Run the code to generate the optical flow animation.

## Explanation

### Converting the images to grayscale and normalizing

The code starts by converting the original images to grayscale and normalizing them. This is done using the OpenCV `cvtColor()` function to convert the images to grayscale and the `normalize()` function to normalize the grayscale values to a range of 0-255.

### Applying the Dense Optical Flow algorithm

The code then applies the Dense Optical Flow algorithm to the sequence of grayscale images. This is done using the `calcOpticalFlowFarneback()` function in OpenCV, which takes two consecutive frames and calculates the motion vectors between them. The motion vectors are then used to warp the painting image using the `remap()` function in OpenCV.

### Saving the animation

The resulting images are saved as PNG files and combined into an MP4 video using the OpenCV `VideoWriter()` class.

## Acknowledgments

This code was adapted from the "Optical Flow" tutorial on the OpenCV website.
