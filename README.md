# Real Time Object DImension measurement

The project reads an image and based on the dimensions of a reference object find the dimensions of other objects in a scene. The reference object must be the background object in the scene. In this project an A4 sheet is taken as a reference object. For any other reference object provide actual width of the object. 

For any other reference object provide actual width of the object.

# Constraints
1.Shadow effect: use dark braground
2.Object boundary: use contrasting background

# Getting Started

## Prerequisites
Python 3 Pip OpenCV Numpy

## Algorithm
1. Image pre-processing
  - Read an image and convert it it no grayscale
  - Blur the image using Gaussian Kernel to remove un-necessary edges
  - Edge detection using Canny edge detector
  - Perform morphological closing operation to remove noisy contours

2. Object Segmentation
  - Find contours
  - Remove small contours by calculating its area
  - Sort contours from left to right to find the reference objects
  
3. Reference object 
  - Calculate how many pixels are there per metric (centi meter is used here)

4. Compute results
  - Draw bounding boxes around each object and calculate its height and width

## Author
Sanket Raikar
