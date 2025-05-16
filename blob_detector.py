import cv2
import numpy as np
import os
from glob import glob

input_dir = 'samples'
output_dir = 'blob_output'
os.makedirs(output_dir, exist_ok=True)

image_paths = glob(os.path.join(input_dir, '*.jpeg'))

def process_image(image_path):
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize lighting
    gray = cv2.equalizeHist(gray)

    # Apply adaptive thresholding to highlight dark spots
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 35, 10
    )

    # Morphology to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 30 < area < 300:  # Likely a defect spot, tweak as needed
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0:  # Avoid elongated structures like rails
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output

# Process all images
for path in image_paths:
    filename = os.path.basename(path)
    print(f"Processing {filename}...")
    result = process_image(path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, result)

print("✅ Spot detection completed and saved in 'output/'")
