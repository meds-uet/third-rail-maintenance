import cv2
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime
import time

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
        """Create output directory structure"""
        Path(self.output_folder).mkdir(exist_ok=True)
        Path(f"{self.output_folder}/processed").mkdir(exist_ok=True)
        Path(f"{self.output_folder}/comparisons").mkdir(exist_ok=True)
        Path(f"{self.output_folder}/live_captures").mkdir(exist_ok=True)
        Path(f"{self.output_folder}/live_recordings").mkdir(exist_ok=True)
    
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
                               minLineLength=self.min_crack_length, maxLineGap=HOUGH_MAX_LINE_GAP)
        
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
                cv2.putText(result_image, label, (x, y-10), 
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
        self.recording = False
        self.video_writer = None
        self.recording_start_time = 0
        self.frame_count = 0
        self.last_frame = None
    
    def create_video_writer(self, filename):
        """Create video writer object"""
        frame_width = LIVE_FRAME_WIDTH * 2 + 20  # Account for side-by-side plus spacing
        frame_height = LIVE_FRAME_HEIGHT + 70     # Account for label areas
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
        return cv2.VideoWriter(filename, fourcc, VIDEO_FPS, (frame_width, frame_height))

    def draw_control_panel(self, frame):
        """Draw buttons and status indicators on the frame"""
        # Record button (red when active)
        record_color = (0, 0, 255) if self.recording else BUTTON_INACTIVE_COLOR
        cv2.rectangle(frame, (10, 10), (150, 60), record_color, -1)
        cv2.putText(frame, "RECORD" if not self.recording else "RECORDING", 
                   (20, 40), DEFECT_LABEL_FONT, 0.6, (255, 255, 255), 2)

        # Capture button
        cv2.rectangle(frame, (170, 10), (310, 60), BUTTON_COLOR, -1)
        cv2.putText(frame, "CAPTURE", (190, 40), 
                   DEFECT_LABEL_FONT, 0.6, (0, 0, 0), 2)

        # Exit button
        cv2.rectangle(frame, (330, 10), (470, 60), (0, 0, 255), -1)
        cv2.putText(frame, "EXIT", (370, 40), 
                   DEFECT_LABEL_FONT, 0.6, (255, 255, 255), 2)

        return frame

    def handle_mouse_clicks(self, event, x, y, flags, param):
        """Handle mouse click events for buttons"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Record button (10-150)
            if 10 <= x <= 150 and 10 <= y <= 60:
                if not self.recording:
                    self.start_recording()
                else:
                    self.stop_recording()

            # Capture button (170-310)
            elif 170 <= x <= 310 and 10 <= y <= 60:
                self.capture_image()

            # Exit button (330-470)
            elif 330 <= x <= 470 and 10 <= y <= 60:
                self.cleanup()
                cv2.destroyAllWindows()
                exit()

    def start_recording(self):
        """Start a new video recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_folder}/live_recordings/recording_{timestamp}.avi"
        self.video_writer = self.create_video_writer(filename)
        self.recording = True
        self.recording_start_time = time.time()
        print(f"Started recording: {filename}")

    def stop_recording(self):
        """Stop the current recording"""
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("Recording stopped and saved")

    def capture_image(self):
        """Capture and save the current frame"""
        if self.last_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_folder}/live_captures/capture_{timestamp}.jpg"
            cv2.imwrite(filename, self.last_frame)
            print(f"Image captured: {filename}")

    def process_frame(self, frame):
        """Process a single frame and return side-by-side view"""
        # Process the frame
        roi_image, roi_bounds = self.extract_rail_roi(frame)
        processed_image, _ = self.remove_led_light(roi_image)
        corrosion_contours, _ = self.detect_corrosion_spots(processed_image)
        wear_contours, _ = self.detect_surface_wear_lines(processed_image)
        crack_contours, _ = self.detect_cracks_and_linear_defects(processed_image)
        defects = self.classify_and_filter_defects(corrosion_contours, wear_contours, crack_contours)
        result_frame = self.visualize_defects(frame, defects, roi_bounds)

        # Create side-by-side comparison
        comparison_frame = self.create_side_by_side_comparison(frame, result_frame, "Live View")
        self.last_frame = comparison_frame  # Store for capture
        
        return comparison_frame

    def process_live_video(self):
        """Main processing loop for live video"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return

        # Set camera properties for RPi compatibility
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, LIVE_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LIVE_FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, LIVE_FPS)

        cv2.namedWindow('Rail Defect Detection - Live')
        cv2.setMouseCallback('Rail Defect Detection - Live', self.handle_mouse_clicks)

        print("Live detection started. Controls:")
        print("- Click RECORD to start/stop 20s recording")
        print("- Click CAPTURE to save still image")
        print("- Click EXIT to quit")
        print("Keyboard shortcuts: R=Record, C=Capture, Q=Quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading from camera")
                break

            # Process frame and get side-by-side view
            processed_frame = self.process_frame(frame)
            control_frame = self.draw_control_panel(processed_frame.copy())

            # Display the frame
            cv2.imshow('Rail Defect Detection - Live', control_frame)

            # Handle recording if active
            if self.recording:
                self.video_writer.write(processed_frame)
                # Auto-stop after RECORDING_DURATION seconds
                if time.time() - self.recording_start_time >= RECORDING_DURATION:
                    self.stop_recording()

            # Handle key presses (alternative to buttons)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not self.recording:
                    self.start_recording()
                else:
                    self.stop_recording()
            elif key == ord('c'):
                self.capture_image()

        self.cleanup()
        cv2.destroyAllWindows()

    def cleanup(self):
        """Release resources"""
        if self.recording:
            self.stop_recording()
        if self.cap is not None:
            self.cap.release()

def main():
    """Main function with mode selection"""
    print("Rail Defect Detection System")
    print("=" * 50)
    print("1. Process images in folder")
    print("2. Live webcam detection")
    print("3. Exit")
    
    choice = input("Select mode (1-3): ")
    
    if choice == '1':
        # Process images in folder
        detector = ThirdRailDefectDetector(input_folder="images", output_folder="output")
        detector.process_all_images()
    elif choice == '2':
        # Live webcam detection
        live_detector = LiveRailDefectDetector(
            camera_index=0,  # Change to 1 if webcam not detected
            input_folder="images",
            output_folder="output"
        )
        live_detector.process_live_video()
    elif choice == '3':
        return
    else:
        print("Invalid choice")

if __name__ == "__main__":
    # Create required directories
    os.makedirs("images", exist_ok=True)
    os.makedirs("output/processed", exist_ok=True)
    os.makedirs("output/comparisons", exist_ok=True)
    os.makedirs("output/live_captures", exist_ok=True)
    os.makedirs("output/live_recordings", exist_ok=True)
    
    main()