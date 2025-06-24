import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# === DEMO CONFIGURATION ===
DEMO_IMAGE = "sample_rail.jpg"  # Put your demo image here
DEMO_OUTPUT_FOLDER = "demo_output"

# === CONFIGURABLE MACROS ===
# ROI Configuration
ROI_START_PERCENT = 0.50          # % of image width to extract for rail ROI
ROI_WIDTH_PERCENT = 0.30          # % of image width to extract for rail ROI

# LED Light Removal Configuration
LED_LIGHT_THRESHOLD = 400         # LED detection brightness threshold
LED_KERNEL_SIZE = 80              # Size of morphological structuring element
LED_MIN_AREA = 200                # Minimum area to trigger inpainting
INPAINT_RADIUS = 20               # Radius for inpainting blur

# Defect Detection Thresholds
CORROSION_THRESHOLD = 50          # Dark spot threshold
CRACK_THRESHOLD = 5               # Crack detection threshold
MIN_DEFECT_AREA = 2               # Ignore small defects below this area
MAX_DEFECT_AREA = 1000            # Ignore overly large blobs
MIN_CRACK_LENGTH = 1              # Minimum length to be considered a crack
ROUGHNESS_INTENSITY_THRESHOLD = 10 # For surface wear / rough track

# Image Processing Parameters
BILATERAL_FILTER_D = 9            # Diameter for bilateral filter
BILATERAL_FILTER_SIGMA_COLOR = 80 # Sigma color for bilateral filter
BILATERAL_FILTER_SIGMA_SPACE = 80 # Sigma space for bilateral filter
GAUSSIAN_BLUR_KERNEL = (3, 3)     # Kernel size for Gaussian blur
ADAPTIVE_THRESH_BLOCK_SIZE = 21   # Block size for adaptive threshold
ADAPTIVE_THRESH_C = 10            # Constant for adaptive threshold
MORPH_KERNEL_SIZE = (5, 5)        # Size for morphological operations
LINE_DETECTION_KERNEL_SIZE = 15   # Size for line detection kernels
CANNY_THRESHOLD1 = 30             # Canny edge detection threshold 1
CANNY_THRESHOLD2 = 100            # Canny edge detection threshold 2
CANNY_APERTURE_SIZE = 3           # Canny aperture size
HOUGH_THRESHOLD = 30              # Threshold for Hough line detection
HOUGH_MIN_LINE_LENGTH = 1         # Minimum line length for Hough
HOUGH_MAX_LINE_GAP = 10           # Maximum line gap for Hough

# Defect Classification Thresholds
CORROSION_SEVERE_AREA = 2000      # Area threshold for severe corrosion
CORROSION_MEDIUM_AREA = 800       # Area threshold for medium corrosion
WEAR_SEVERE_AREA = 1500           # Area threshold for severe wear
CRACK_SEVERE_LENGTH = 100         # Length threshold for severe cracks
DEFECT_LABEL_AREA = 500           # Minimum area to show defect label

# Visualization Parameters
ROI_BOUNDARY_COLOR = (255, 255, 0) # Color for ROI boundary (BGR)
DEFECT_COLORS = {
    'Corrosion/Dark Spot': (0, 0, 255),    # Red
    'Surface Wear/Lines': (255, 0, 0),     # Blue
    'Crack/Linear Defect': (0, 255, 0)     # Green
}

DEFECT_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFECT_LABEL_FONT_SCALE = 0.6      # Increased font size
DEFECT_LABEL_THICKNESS = 2         # Increased thickness
SUMMARY_FONT_SCALE = 1.0           # Increased summary font size
SUMMARY_FONT_THICKNESS = 2

# Live Processing Parameters
LIVE_FRAME_WIDTH = 640             # Reduced for RPi performance
LIVE_FRAME_HEIGHT = 480
LIVE_FPS = 15                      # Reduced FPS for RPi
RECORDING_DURATION = 20            # seconds
VIDEO_FPS = 15                     # frames per second for recording
OUTPUT_VIDEO_CODEC = 'XVID'        # Codec for video recording
BUTTON_COLOR = (0, 255, 0)         # Green color for active buttons
BUTTON_INACTIVE_COLOR = (50, 50, 50) # Gray color for inactive buttons

class ThirdRailDefectDetectorDemo:
    def __init__(self, demo_image_path, output_folder="demo_output"):
        self.demo_image_path = demo_image_path
        self.output_folder = output_folder
        self.setup_output_folder()
        self.step_counter = 0
        
        # Initialize detection parameters
        self.corrosion_threshold = CORROSION_THRESHOLD
        self.crack_threshold = CRACK_THRESHOLD
        self.min_defect_area = MIN_DEFECT_AREA
        self.max_defect_area = MAX_DEFECT_AREA
        self.min_crack_length = MIN_CRACK_LENGTH
        self.roughness_intensity_threshold = ROUGHNESS_INTENSITY_THRESHOLD
        self.led_light_threshold = LED_LIGHT_THRESHOLD
        self.led_kernel_size = LED_KERNEL_SIZE
        self.led_min_area = LED_MIN_AREA
        self.inpaint_radius = INPAINT_RADIUS

    def setup_output_folder(self):
        """Create output directory structure"""
        Path(self.output_folder).mkdir(exist_ok=True)
        print(f"Demo output folder created: {self.output_folder}")

    def save_step_image(self, image, step_name, description=""):
        """Save image for each processing step"""
        self.step_counter += 1
        filename = f"step_{self.step_counter:02d}_{step_name}.jpg"
        filepath = os.path.join(self.output_folder, filename)
        
        # Add step label to image
        labeled_image = self.add_step_label(image, f"Step {self.step_counter}: {step_name}", description)
        
        cv2.imwrite(filepath, labeled_image)
        print(f"Saved: {filename} - {description}")
        return filepath

    def add_step_label(self, image, title, description=""):
        """Add step label and description to image"""
        labeled_image = image.copy()
        
        # Add title at top
        font = DEFECT_LABEL_FONT
        font_scale = 1.0
        thickness = 2
        color = (255, 255, 255)
        
        # Get text size for background rectangle
        title_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        
        # Add black background for title
        cv2.rectangle(labeled_image, (10, 10), (title_size[0] + 20, title_size[1] + 20), (0, 0, 0), -1)
        cv2.putText(labeled_image, title, (15, title_size[1] + 15), font, font_scale, color, thickness)
        
        # Add description if provided
        if description:
            desc_y = title_size[1] + 40
            cv2.putText(labeled_image, description, (15, desc_y), font, 0.5, (200, 200, 200), 1)
        
        return labeled_image

    def extract_rail_roi(self, image):
        """Extract ROI and save visualization"""
        height, width = image.shape[:2]
        roi_start_x = int(width * ROI_START_PERCENT)
        roi_end_x = int(width * (ROI_START_PERCENT + ROI_WIDTH_PERCENT))
        roi_end_x = min(roi_end_x, width)
        
        # Create ROI visualization
        roi_vis = image.copy()
        cv2.rectangle(roi_vis, (roi_start_x, 0), (roi_end_x, height), ROI_BOUNDARY_COLOR, 3)
        
        # Add ROI annotation
        cv2.putText(roi_vis, "Region of Interest (Rail Area)", 
                   (roi_start_x, height - 20), DEFECT_LABEL_FONT, 0.6, ROI_BOUNDARY_COLOR, 2)
        
        self.save_step_image(roi_vis, "ROI_Selection", "Selecting rail region for analysis")
        
        # Extract actual ROI
        roi = image[:, roi_start_x:roi_end_x]
        self.save_step_image(roi, "ROI_Extracted", "Extracted rail region for processing")
        
        return roi, (roi_start_x, roi_end_x)

    def remove_led_light(self, image):
        """Remove LED glare and save intermediate steps"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Save grayscale conversion
        self.save_step_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "Grayscale_Conversion", 
                           "Converting to grayscale for LED detection")
        
        # Bright region detection
        led_mask = gray > self.led_light_threshold
        led_mask_vis = cv2.cvtColor((led_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.save_step_image(led_mask_vis, "LED_Detection_Mask", 
                           f"Detecting bright areas (threshold: {self.led_light_threshold})")
        
        # Morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.led_kernel_size, self.led_kernel_size))
        led_mask_morph = cv2.morphologyEx(led_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        led_mask_morph_vis = cv2.cvtColor(led_mask_morph * 255, cv2.COLOR_GRAY2BGR)
        self.save_step_image(led_mask_morph_vis, "LED_Mask_Refined", 
                           "Morphological processing to merge LED regions")
        
        # Inpaint if needed
        led_area = np.sum(led_mask_morph)
        if led_area > self.led_min_area:
            result = cv2.inpaint(image, led_mask_morph, self.inpaint_radius, cv2.INPAINT_TELEA)
            self.save_step_image(result, "LED_Removal_Applied", 
                               f"LED glare removed using inpainting (area: {led_area})")
        else:
            result = image.copy()
            self.save_step_image(result, "LED_Removal_Skipped", 
                               f"No significant LED detected (area: {led_area} < {self.led_min_area})")
        
        return result, led_mask_morph

    def detect_corrosion_spots(self, image):
        """Detect corrosion with step-by-step visualization"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Bilateral filter
        filtered = cv2.bilateralFilter(gray, BILATERAL_FILTER_D, 
                                     BILATERAL_FILTER_SIGMA_COLOR, BILATERAL_FILTER_SIGMA_SPACE)
        self.save_step_image(cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR), "Bilateral_Filter", 
                           "Noise reduction while preserving edges")
        
        # Threshold methods
        _, dark_spots1 = cv2.threshold(filtered, self.corrosion_threshold, 255, cv2.THRESH_BINARY_INV)
        self.save_step_image(cv2.cvtColor(dark_spots1, cv2.COLOR_GRAY2BGR), "Simple_Threshold", 
                           f"Simple threshold (threshold: {self.corrosion_threshold})")
        
        dark_spots2 = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C)
        self.save_step_image(cv2.cvtColor(dark_spots2, cv2.COLOR_GRAY2BGR), "Adaptive_Threshold", 
                           "Adaptive threshold for local variations")
        
        # Combine methods
        dark_spots = cv2.bitwise_or(dark_spots1, dark_spots2)
        self.save_step_image(cv2.cvtColor(dark_spots, cv2.COLOR_GRAY2BGR), "Combined_Threshold", 
                           "Combined threshold results")
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        dark_spots_clean = cv2.morphologyEx(dark_spots, cv2.MORPH_OPEN, kernel)
        dark_spots_clean = cv2.morphologyEx(dark_spots_clean, cv2.MORPH_CLOSE, kernel)
        self.save_step_image(cv2.cvtColor(dark_spots_clean, cv2.COLOR_GRAY2BGR), "Morphology_Cleanup", 
                           "Morphological noise removal")
        
        # Find and filter contours
        contours, _ = cv2.findContours(dark_spots_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Visualize all contours
        all_contours_vis = image.copy()
        cv2.drawContours(all_contours_vis, contours, -1, (0, 255, 255), 2)
        self.save_step_image(all_contours_vis, "All_Corrosion_Contours", 
                           f"All detected contours ({len(contours)} found)")
        
        # Filter contours
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_defect_area < area < self.max_defect_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if aspect_ratio < 10:
                    valid_contours.append(contour)
        
        # Visualize filtered contours
        filtered_contours_vis = image.copy()
        cv2.drawContours(filtered_contours_vis, valid_contours, -1, DEFECT_COLORS['Corrosion/Dark Spot'], 2)
        self.save_step_image(filtered_contours_vis, "Filtered_Corrosion", 
                           f"Valid corrosion spots ({len(valid_contours)} found)")
        
        return valid_contours, dark_spots_clean

    def detect_surface_wear_lines(self, image):
        """Detect surface wear with visualization"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
        self.save_step_image(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), "Gaussian_Blur", 
                           "Gaussian blur for noise reduction")
        
        # High-pass filter
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        high_pass = cv2.filter2D(blurred, -1, kernel)
        high_pass = np.absolute(high_pass)
        self.save_step_image(cv2.cvtColor(high_pass.astype(np.uint8), cv2.COLOR_GRAY2BGR), "High_Pass_Filter", 
                           "High-pass filter for edge enhancement")
        
        # Line detection kernels
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_DETECTION_KERNEL_SIZE, 1))
        horizontal_lines = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, horizontal_kernel)
        self.save_step_image(cv2.cvtColor(horizontal_lines, cv2.COLOR_GRAY2BGR), "Horizontal_Lines", 
                           "Horizontal line detection")
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_DETECTION_KERNEL_SIZE))
        vertical_lines = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, vertical_kernel)
        self.save_step_image(cv2.cvtColor(vertical_lines, cv2.COLOR_GRAY2BGR), "Vertical_Lines", 
                           "Vertical line detection")
        
        # Combine line detections
        lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        self.save_step_image(cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR), "Combined_Lines", 
                           "Combined line patterns")
        
        # Laplacian edge detection
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        laplacian = np.absolute(laplacian).astype(np.uint8)
        self.save_step_image(cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR), "Laplacian_Edges", 
                           "Laplacian edge detection")
        
        # Combine all methods
        combined = cv2.addWeighted(high_pass.astype(np.uint8), 0.4, lines, 0.3, 0)
        combined = cv2.addWeighted(combined, 0.7, laplacian, 0.3, 0)
        self.save_step_image(cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR), "Combined_Wear_Detection", 
                           "Combined wear detection methods")
        
        # Threshold
        _, wear_mask = cv2.threshold(combined, self.roughness_intensity_threshold, 255, cv2.THRESH_BINARY)
        self.save_step_image(cv2.cvtColor(wear_mask, cv2.COLOR_GRAY2BGR), "Wear_Threshold", 
                           f"Wear pattern threshold ({self.roughness_intensity_threshold})")
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
        wear_mask_clean = cv2.morphologyEx(wear_mask, cv2.MORPH_OPEN, kernel)
        wear_mask_clean = cv2.morphologyEx(wear_mask_clean, cv2.MORPH_CLOSE, kernel)
        self.save_step_image(cv2.cvtColor(wear_mask_clean, cv2.COLOR_GRAY2BGR), "Wear_Mask_Clean", 
                           "Cleaned wear mask")
        
        # Find contours
        contours, _ = cv2.findContours(wear_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_defect_area:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if 1.5 < aspect_ratio < 8:
                        valid_contours.append(contour)
        
        # Visualize wear contours
        wear_vis = image.copy()
        cv2.drawContours(wear_vis, valid_contours, -1, DEFECT_COLORS['Surface Wear/Lines'], 2)
        self.save_step_image(wear_vis, "Surface_Wear_Final", 
                           f"Final surface wear detection ({len(valid_contours)} found)")
        
        return valid_contours, wear_mask_clean

    def detect_cracks_and_linear_defects(self, image):
        """Detect cracks with visualization"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Bilateral filter
        filtered = cv2.bilateralFilter(gray, BILATERAL_FILTER_D, 
                                     BILATERAL_FILTER_SIGMA_COLOR, BILATERAL_FILTER_SIGMA_SPACE)
        self.save_step_image(cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR), "Crack_Bilateral_Filter", 
                           "Bilateral filter for crack detection")
        
        # Edge detection
        edges1 = cv2.Canny(filtered, CANNY_THRESHOLD1, CANNY_THRESHOLD2, apertureSize=CANNY_APERTURE_SIZE)
        self.save_step_image(cv2.cvtColor(edges1, cv2.COLOR_GRAY2BGR), "Canny_Edges_1", 
                           f"Canny edges (thresholds: {CANNY_THRESHOLD1}, {CANNY_THRESHOLD2})")
        
        edges2 = cv2.Canny(filtered, 50, 150, apertureSize=CANNY_APERTURE_SIZE)
        self.save_step_image(cv2.cvtColor(edges2, cv2.COLOR_GRAY2BGR), "Canny_Edges_2", 
                           "Canny edges (alternative thresholds: 50, 150)")
        
        # Combine edges
        edges = cv2.bitwise_or(edges1, edges2)
        self.save_step_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "Combined_Edges", 
                           "Combined edge detection results")
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=HOUGH_THRESHOLD, 
                               minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)
        
        # Visualize detected lines
        line_vis = image.copy()
        line_mask = np.zeros_like(gray)
        crack_contours = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_vis, (x1, y1), (x2, y2), DEFECT_COLORS['Crack/Linear Defect'], 2)
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
                
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > self.min_crack_length:
                    contour = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.int32)
                    crack_contours.append(contour)
        
        self.save_step_image(line_vis, "Hough_Lines", 
                           f"Hough line detection ({len(lines) if lines is not None else 0} lines)")
        
        # Also detect from edge contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        edge_contour_vis = image.copy()
        linear_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_defect_area:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > 3.0:
                        linear_contours.append(contour)
                        crack_contours.append(contour)
        
        cv2.drawContours(edge_contour_vis, linear_contours, -1, DEFECT_COLORS['Crack/Linear Defect'], 2)
        self.save_step_image(edge_contour_vis, "Linear_Contours", 
                           f"Linear contours from edges ({len(linear_contours)} found)")
        
        # Final crack visualization
        crack_vis = image.copy()
        cv2.drawContours(crack_vis, crack_contours, -1, DEFECT_COLORS['Crack/Linear Defect'], 2)
        self.save_step_image(crack_vis, "Crack_Detection_Final", 
                           f"Final crack detection ({len(crack_contours)} found)")
        
        return crack_contours, line_mask

    def classify_and_visualize_defects(self, original_image, corrosion_contours, wear_contours, crack_contours, roi_bounds):
        """Classify defects and create final visualization"""
        defects = []
        roi_start_x, roi_end_x = roi_bounds
        
        # Process corrosion spots
        for contour in corrosion_contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            if area > self.min_defect_area:
                severity = "High" if area > CORROSION_SEVERE_AREA else "Medium" if area > CORROSION_MEDIUM_AREA else "Low"
                defects.append({
                    'type': 'Corrosion/Dark Spot',
                    'contour': contour,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'severity': severity,
                    'color': DEFECT_COLORS['Corrosion/Dark Spot']
                })
        
        # Process surface wear
        for contour in wear_contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            if area > self.min_defect_area:
                severity = "Medium" if area > WEAR_SEVERE_AREA else "Low"
                defects.append({
                    'type': 'Surface Wear/Lines',
                    'contour': contour,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'severity': severity,
                    'color': DEFECT_COLORS['Surface Wear/Lines']
                })
        
        # Process cracks
        for contour in crack_contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            
            if length > self.min_crack_length and area > 100:
                severity = "High" if length > CRACK_SEVERE_LENGTH else "Medium"
                defects.append({
                    'type': 'Crack/Linear Defect',
                    'contour': contour,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'length': length,
                    'severity': severity,
                    'color': DEFECT_COLORS['Crack/Linear Defect']
                })
        
        # Create final visualization
        result_image = original_image.copy()
        
        # Draw ROI boundary
        cv2.rectangle(result_image, (roi_start_x, 0), (roi_end_x, original_image.shape[0]), 
                     ROI_BOUNDARY_COLOR, 2)
        
        defect_count = {'Corrosion/Dark Spot': 0, 'Surface Wear/Lines': 0, 'Crack/Linear Defect': 0}
        
        for defect in defects:
            color = defect['color']
            contour = defect['contour']
            defect_count[defect['type']] += 1
            
            # Adjust contour coordinates to original image
            adjusted_contour = contour.copy()
            adjusted_contour[:, :, 0] += roi_start_x
            
            # Draw filled contour with transparency
            overlay = result_image.copy()
            cv2.fillPoly(overlay, [adjusted_contour], color)
            result_image = cv2.addWeighted(result_image, 0.8, overlay, 0.2, 0)
            
            # Draw contour outline
            cv2.drawContours(result_image, [adjusted_contour], -1, color, 2)
            
            # Draw bounding box for significant defects
            if defect['area'] > DEFECT_LABEL_AREA:
                x, y, w, h = defect['bbox']
                x += roi_start_x
                cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
                
                # Add label
                label = f"{defect['type'][:4]}-{defect['severity'][0]}"
                cv2.putText(result_image, label, (x, y-5), 
                           DEFECT_LABEL_FONT, DEFECT_LABEL_FONT_SCALE, color, DEFECT_LABEL_THICKNESS)
        
        # Add summary text
        y_offset = 30
        cv2.putText(result_image, f"Total Defects: {len(defects)}", (10, y_offset), 
                   DEFECT_LABEL_FONT, 0.7, (255, 255, 255), 2)
        
        for defect_type, count in defect_count.items():
            if count > 0:
                y_offset += 25
                cv2.putText(result_image, f"{defect_type}: {count}", (10, y_offset), 
                           DEFECT_LABEL_FONT, 0.5, (255, 255, 255), 1)
        
        self.save_step_image(result_image, "Final_Result", 
                           f"Complete defect detection results ({len(defects)} defects found)")
        
        return result_image, defects

    def run_demo(self):
        """Run the complete demo process"""
        print("=" * 60)
        print("THIRD RAIL DEFECT DETECTION - STEP-BY-STEP DEMO")
        print("=" * 60)
        
        # Check if demo image exists
        if not os.path.exists(self.demo_image_path):
            print(f"Error: Demo image '{self.demo_image_path}' not found!")
            print("Please place a rail image named 'demo_rail.jpg' in the current directory.")
            return
        
        # Load image
        original_image = cv2.imread(self.demo_image_path)
        if original_image is None:
            print(f"Error: Could not load image '{self.demo_image_path}'")
            return
        
        print(f"Processing demo image: {self.demo_image_path}")
        print(f"Image size: {original_image.shape[1]}x{original_image.shape[0]}")
        print()
        
        # Step 0: Save original image
        self.save_step_image(original_image, "Original_Image", "Original input image")
        
        # Step 1: Extract ROI
        roi_image, roi_bounds = self.extract_rail_roi(original_image)
        
        # Step 2: Remove LED light
        processed_image, led_mask = self.remove_led_light(roi_image)
        
        # Step 3: Detect corrosion spots
        corrosion_contours, corrosion_mask = self.detect_corrosion_spots(processed_image)
        
        # Step 4: Detect surface wear
        wear_contours, wear_mask = self.detect_surface_wear_lines(processed_image)
        
        # Step 5: Detect cracks
        crack_contours, crack_mask = self.detect_cracks_and_linear_defects(processed_image)
        
        # Step 6: Final classification and visualization
        final_result, defects = self.classify_and_visualize_defects(
            original_image, corrosion_contours, wear_contours, crack_contours, roi_bounds
        )
        
        # Create summary report
        self.create_summary_report(defects)
        
        print()
        print("=" * 60)
        print("DEMO COMPLETED")
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)

    def create_summary_report(self, defects):
        """Generate and save a text summary report"""
        report_lines = []
        report_lines.append("=== THIRD RAIL DEFECT DETECTION REPORT ===")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Defects Found: {len(defects)}\n")

        for idx, defect in enumerate(defects, 1):
            info = [
                f"Defect #{idx}:",
                f"  Type     : {defect['type']}",
                f"  Severity : {defect['severity']}",
                f"  Area     : {defect['area']:.2f}",
                f"  BBox     : {defect['bbox']}",
            ]
            if 'length' in defect:
                info.append(f"  Length   : {defect['length']:.2f}")
            report_lines.extend(info)
            report_lines.append("")

        report_text = "\n".join(report_lines)
        report_path = os.path.join(self.output_folder, "defect_summary_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        print(f"Summary report saved to: {report_path}")


# === Script Entry Point ===
if __name__ == "__main__":
    demo = ThirdRailDefectDetectorDemo(DEMO_IMAGE, DEMO_OUTPUT_FOLDER)
    demo.run_demo()
