import cv2
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime


# === CONFIGURABLE MACROS ===
# ROI Configuration
ROI_START_PERCENT = 0.40          # % of image width to extract for rail ROI
ROI_WIDTH_PERCENT = 0.40          # % of image width to extract for rail ROI

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
ROUGHNESS_INTENSITY_THRESHOLD = 20 # For surface wear / rough track

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
CORROSION_SEVERE_AREA = 500      # Area threshold for severe corrosion
CORROSION_MEDIUM_AREA = 200       # Area threshold for medium corrosion
WEAR_SEVERE_AREA = 500           # Area threshold for severe wear
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
DEFECT_LABEL_FONT_SCALE = 0.7
DEFECT_LABEL_THICKNESS = 2
SUMMARY_FONT_SCALE = 0.7
SUMMARY_FONT_THICKNESS = 2

# Live Processing Parameters
LIVE_FRAME_WIDTH = 1280           # Camera frame width
LIVE_FRAME_HEIGHT = 720           # Camera frame height
LIVE_FPS = 30                     # Camera frames per second
LIVE_PROCESS_EVERY_N_FRAME = 3    # Process every Nth frame for performance


class ThirdRailDefectDetector:
    def __init__(self, input_folder="images", output_folder="output"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.setup_output_folder()
        
        # Initialize detection parameters from macros
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
        """Create simplified output directory structure"""
        Path(self.output_folder).mkdir(exist_ok=True)
        Path(f"{self.output_folder}/processed").mkdir(exist_ok=True)
        Path(f"{self.output_folder}/comparisons").mkdir(exist_ok=True)
    
    def extract_rail_roi(self, image):
        """Extract ROI starting at center and covering next 30% of the width"""
        height, width = image.shape[:2]
        roi_start_x = int(width * ROI_START_PERCENT)
        roi_end_x = int(width * (ROI_START_PERCENT + ROI_WIDTH_PERCENT))
        roi_end_x = min(roi_end_x, width)
        roi = image[:, roi_start_x:roi_end_x]
        return roi, (roi_start_x, roi_end_x)
    
    def remove_led_light(self, image):
        """Remove LED glare based on brightness and morphology"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Bright region detection using threshold
        led_mask = gray > self.led_light_threshold
        
        # Morphological operation to merge glare blobs
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.led_kernel_size, self.led_kernel_size)
        )
        led_mask = cv2.morphologyEx(led_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Check if glare is large enough to inpaint
        led_area = np.sum(led_mask)
        if led_area > self.led_min_area:
            result = cv2.inpaint(image, led_mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        else:
            result = image.copy()
        
        return result, led_mask
    
    def detect_corrosion_spots(self, image):
        """Improved corrosion detection with better sensitivity"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, BILATERAL_FILTER_D, 
                                      BILATERAL_FILTER_SIGMA_COLOR, 
                                      BILATERAL_FILTER_SIGMA_SPACE)
        
        # Multiple threshold approaches for better detection
        _, dark_spots1 = cv2.threshold(filtered, self.corrosion_threshold, 255, cv2.THRESH_BINARY_INV)
        dark_spots2 = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C)
        
        # Combine both methods
        dark_spots = cv2.bitwise_or(dark_spots1, dark_spots2)
        
        # Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_OPEN, kernel)
        dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and filter by area
        contours, _ = cv2.findContours(dark_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_defect_area < area < self.max_defect_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                if aspect_ratio < 10:  # Not too elongated (likely noise)
                    valid_contours.append(contour)
        
        return valid_contours, dark_spots
    
    def detect_surface_wear_lines(self, image):
        """Improved detection of white lines and surface wear patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
        
        # Detect bright lines/wear patterns using multiple methods
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        high_pass = cv2.filter2D(blurred, -1, kernel)
        high_pass = np.absolute(high_pass)
        
        # Morphological operations to find linear patterns
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_DETECTION_KERNEL_SIZE, 1))
        horizontal_lines = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, horizontal_kernel)
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_DETECTION_KERNEL_SIZE))
        vertical_lines = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine line detections
        lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # Laplacian for edge/texture detection
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(laplacian)
        
        # Combine all methods
        combined = cv2.addWeighted(high_pass, 0.4, lines, 0.3, 0)
        combined = cv2.addWeighted(combined, 0.7, laplacian, 0.3, 0)
        
        # Threshold to find wear patterns
        _, wear_mask = cv2.threshold(combined, self.roughness_intensity_threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
        wear_mask = cv2.morphologyEx(wear_mask, cv2.MORPH_OPEN, kernel)
        wear_mask = cv2.morphologyEx(wear_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(wear_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours for wear patterns
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_defect_area:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if 1.5 < aspect_ratio < 8:  # Wear patterns are moderately elongated
                        valid_contours.append(contour)
        
        return valid_contours, wear_mask
    
    def detect_cracks_and_linear_defects(self, image):
        """Enhanced crack detection with better linear feature detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply bilateral filter to preserve edges
        filtered = cv2.bilateralFilter(gray, BILATERAL_FILTER_D, 
                                     BILATERAL_FILTER_SIGMA_COLOR, 
                                     BILATERAL_FILTER_SIGMA_SPACE)
        
        # Multi-scale edge detection
        edges1 = cv2.Canny(filtered, CANNY_THRESHOLD1, CANNY_THRESHOLD2, apertureSize=CANNY_APERTURE_SIZE)
        edges2 = cv2.Canny(filtered, 50, 150, apertureSize=CANNY_APERTURE_SIZE)
        
        # Combine edges
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=HOUGH_THRESHOLD, 
                               minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)
        
        # Create mask for detected lines
        line_mask = np.zeros_like(gray)
        crack_contours = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
                
                # Create contour from line
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > self.min_crack_length:
                    contour = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.int32)
                    crack_contours.append(contour)
        
        # Also detect crack-like contours from edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_defect_area:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > 3.0:  # Very elongated shapes (cracks)
                        crack_contours.append(contour)
        
        return crack_contours, line_mask
    
    def classify_and_filter_defects(self, corrosion_contours, wear_contours, crack_contours):
        """Classify defects and filter out insignificant ones"""
        defects = []
        
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
        
        return defects
    
    def visualize_defects(self, original_image, defects, roi_bounds):
        """Create visualization with better annotations"""
        result_image = original_image.copy()
        roi_start_x, roi_end_x = roi_bounds
        
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
                   DEFECT_LABEL_FONT, SUMMARY_FONT_SCALE, (255, 255, 255), SUMMARY_FONT_THICKNESS)
        
        for defect_type, count in defect_count.items():
            if count > 0:
                y_offset += 25
                cv2.putText(result_image, f"{defect_type}: {count}", (10, y_offset), 
                           DEFECT_LABEL_FONT, 0.5, (255, 255, 255), 1)
        
        return result_image
    
    def create_side_by_side_comparison(self, original_image, processed_image, filename):
        """Create side-by-side comparison image with labels"""
        # Ensure both images have the same height
        orig_h, orig_w = original_image.shape[:2]
        proc_h, proc_w = processed_image.shape[:2]
        
        if orig_h != proc_h:
            target_height = min(orig_h, proc_h)
            original_resized = cv2.resize(original_image, 
                                        (int(orig_w * target_height / orig_h), target_height))
            processed_resized = cv2.resize(processed_image, 
                                         (int(proc_w * target_height / proc_h), target_height))
        else:
            original_resized = original_image.copy()
            processed_resized = processed_image.copy()
        
        # Add spacing between images
        spacing = 20
        spacing_strip = np.ones((original_resized.shape[0], spacing, 3), dtype=np.uint8) * 128
        
        # Concatenate images horizontally with spacing
        comparison_image = np.hstack([original_resized, spacing_strip, processed_resized])
        
        # Add labels at the top
        label_height = 40
        label_area = np.ones((label_height, comparison_image.shape[1], 3), dtype=np.uint8) * 50
        
        # Add text labels
        font = DEFECT_LABEL_FONT
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)
        
        # Original image label
        orig_text = "ORIGINAL"
        orig_text_size = cv2.getTextSize(orig_text, font, font_scale, thickness)[0]
        orig_x = (original_resized.shape[1] - orig_text_size[0]) // 2
        cv2.putText(label_area, orig_text, (orig_x, 25), font, font_scale, color, thickness)
        
        # Processed image label
        proc_text = "DEFECTS DETECTED"
        proc_text_size = cv2.getTextSize(proc_text, font, font_scale, thickness)[0]
        proc_x = original_resized.shape[1] + spacing + (processed_resized.shape[1] - proc_text_size[0]) // 2
        cv2.putText(label_area, proc_text, (proc_x, 25), font, font_scale, color, thickness)
        
        # Add filename at the bottom
        filename_text = f"File: {filename}"
        filename_size = cv2.getTextSize(filename_text, font, 0.6, 1)[0]
        filename_x = (comparison_image.shape[1] - filename_size[0]) // 2
        
        # Create bottom label area
        bottom_label = np.ones((30, comparison_image.shape[1], 3), dtype=np.uint8) * 50
        cv2.putText(bottom_label, filename_text, (filename_x, 20), font, 0.6, color, 1)
        
        # Combine all parts
        final_image = np.vstack([label_area, comparison_image, bottom_label])
        
        return final_image
    
    def process_single_image(self, image_path):
        """Process a single image with improved detection"""
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return None
        
        original_image = image.copy()
        
        # Extract rail ROI with better coverage
        roi_image, roi_bounds = self.extract_rail_roi(image)
        
        # Remove LED light more carefully
        processed_image, led_mask = self.remove_led_light(roi_image)
        
        # Detect different types of defects with improved algorithms
        corrosion_contours, corrosion_mask = self.detect_corrosion_spots(processed_image)
        wear_contours, wear_mask = self.detect_surface_wear_lines(processed_image)
        crack_contours, crack_mask = self.detect_cracks_and_linear_defects(processed_image)
        
        print(f"  - Raw detections: {len(corrosion_contours)} corrosion, {len(wear_contours)} wear, {len(crack_contours)} cracks")
        
        # Classify and filter defects
        defects = self.classify_and_filter_defects(corrosion_contours, wear_contours, crack_contours)
        
        # Create visualization
        result_image = self.visualize_defects(original_image, defects, roi_bounds)
        
        print(f"  - Final significant defects: {len(defects)}")
        
        return {
            'original_image': original_image,
            'processed_image': result_image,
            'roi_image': roi_image,
            'defects': defects,
            'masks': {
                'corrosion': corrosion_mask,
                'wear': wear_mask,
                'cracks': crack_mask
            }
        }
    
    def process_all_images(self):
        """Process all images in the input folder"""
        image_paths = glob.glob(f"{self.input_folder}/*.jpg")
        
        if not image_paths:
            print(f"No images found in {self.input_folder}")
            return
        
        print(f"Found {len(image_paths)} images to process")
        print("=" * 50)
        
        for image_path in image_paths:
            result = self.process_single_image(image_path)
            
            if result is not None:
                filename = os.path.basename(image_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                # Save processed image
                output_path = f"{self.output_folder}/processed/{name_without_ext}_defects.jpg"
                cv2.imwrite(output_path, result['processed_image'])
                
                # Create and save side-by-side comparison
                comparison_image = self.create_side_by_side_comparison(
                    result['original_image'], 
                    result['processed_image'], 
                    filename
                )
                comparison_path = f"{self.output_folder}/comparisons/{name_without_ext}_comparison.jpg"
                cv2.imwrite(comparison_path, comparison_image)
                
                print(f"  - Saved processed: {output_path}")
                print(f"  - Saved comparison: {comparison_path}")
            
            print("-" * 30)
        
        print(f"\nProcessing complete!")
        print(f"Results saved to:")
        print(f"  - Processed images: {self.output_folder}/processed/")
        print(f"  - Side-by-side comparisons: {self.output_folder}/comparisons/")

class LiveRailDefectDetector(ThirdRailDefectDetector):
    """Live video processing with optimized performance"""
    
    def __init__(self, camera_index=0, **kwargs):
        super().__init__(**kwargs)
        self.camera_index = camera_index
        self.cap = None
    
    def initialize_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, LIVE_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LIVE_FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, LIVE_FPS)
        
        return True
    
    def process_live_video(self):
        """Process live video stream"""
        if not self.initialize_camera():
            return
        
        print("Starting live defect detection. Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading from camera")
                break
            
            frame_count += 1
            
            # Process every 3rd frame for better performance
            if frame_count % LIVE_PROCESS_EVERY_N_FRAME == 0:
                # Extract rail ROI
                roi_image, roi_bounds = self.extract_rail_roi(frame)
                
                # Quick defect detection (simplified for real-time)
                processed_image, _ = self.remove_led_light(roi_image)
                corrosion_contours, _ = self.detect_corrosion_spots(processed_image)
                wear_contours, _ = self.detect_surface_wear_lines(processed_image)
                
                # Classify defects
                defects = self.classify_and_filter_defects(corrosion_contours, wear_contours, [])
                
                # Create visualization
                result_frame = self.visualize_defects(frame, defects, roi_bounds)
                
                cv2.imshow('Rail Defect Detection - Live', result_frame)
            else:
                cv2.imshow('Rail Defect Detection - Live', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save original frame
                original_path = f"{self.output_folder}/live_capture_original_{timestamp}.jpg"
                cv2.imwrite(original_path, frame)
                
                # Process and save comparison
                roi_image, roi_bounds = self.extract_rail_roi(frame)
                processed_image, _ = self.remove_led_light(roi_image)
                corrosion_contours, _ = self.detect_corrosion_spots(processed_image)
                wear_contours, _ = self.detect_surface_wear_lines(processed_image)
                crack_contours, _ = self.detect_cracks_and_linear_defects(processed_image)
                defects = self.classify_and_filter_defects(corrosion_contours, wear_contours, crack_contours)
                result_frame = self.visualize_defects(frame, defects, roi_bounds)
                
                # Create comparison
                comparison_image = self.create_side_by_side_comparison(
                    frame, result_frame, f"live_capture_{timestamp}.jpg"
                )
                comparison_path = f"{self.output_folder}/comparisons/live_capture_comparison_{timestamp}.jpg"
                cv2.imwrite(comparison_path, comparison_image)
                
                print(f"Frame saved: {original_path}")
                print(f"Comparison saved: {comparison_path}")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    print("Third Rail Defect Detection System - Enhanced with Side-by-Side Comparisons")
    print("=" * 70)
    
    # Initialize detector
    detector = ThirdRailDefectDetector(input_folder="images", output_folder="output")
    
    # Process all images
    detector.process_all_images()
    
    # Ask for live detection
    # response = input("\nStart live video detection? (y/n): ")
    # if response.lower() == 'y':
    #     live_detector = LiveRailDefectDetector(camera_index=0, 
    #                                          input_folder="images", 
    #                                          output_folder="output")
    #     live_detector.process_live_video()

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)  
    os.makedirs("output", exist_ok=True)
    main()