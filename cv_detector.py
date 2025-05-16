import cv2
import numpy as np
import os
from glob import glob

# Input and output directories
input_dir = 'samples'
output_dir = 'cv_output'
os.makedirs(output_dir, exist_ok=True)

# Get all JPEG images in the input directory
image_paths = glob(os.path.join(input_dir, '*.jpeg'))

def process_image(image_path):
    image = cv2.imread(image_path)

    # Resize (optional, based on your image size)
    scale_percent = 80
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization
    equalized = cv2.equalizeHist(gray)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 30, 100)

    # Morphological Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Contour Detection
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    output = image.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return output

# Process all images
for path in image_paths:
    filename = os.path.basename(path)
    print(f"Processing {filename}...")
    result = process_image(path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, result)

print("✅ All images processed and saved in 'output/' directory.")
