import cv2
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime
import time

# =============================================================================
# CONFIGURABLE MACROS
# =============================================================================

# ROI Configuration — horizontal strip of the image containing the rail contact surface
ROI_START_PERCENT = 0.40          # Start of ROI as fraction of frame width
ROI_WIDTH_PERCENT  = 0.40         # Width of ROI as fraction of frame width

# LED / Glare Removal
# FIX: original value was 400 which is larger than max uint8 (255), so it NEVER fired.
# Correct range is 0-255. 210 targets only genuine LED-bright pixels.
LED_BRIGHTNESS_THRESHOLD = 210    # Pixels brighter than this are treated as glare (uint8, 0-255)
LED_KERNEL_SIZE           = 25    # Morphological dilation kernel for glare mask (px)
LED_MIN_AREA              = 150   # Minimum glare blob area to trigger inpainting
INPAINT_RADIUS            = 15    # Inpainting fill radius (px)

# Corrosion / Dark-Spot Detection
# FIX: original combined simple-threshold (THRESH_BINARY_INV at 50) with adaptive,
# then OR'd them — the adaptive alone already catches every dark region relative to
# its local neighbourhood, so OR-ing with a global threshold just adds noise.
# We now use adaptive-only with stricter block size and C constant.
CORROSION_ADAPTIVE_BLOCK  = 51    # Adaptive threshold block size (must be odd)
CORROSION_ADAPTIVE_C      = 18    # Constant subtracted from local mean
CORROSION_MIN_AREA        = 80    # Minimum contour area to count as a defect (px²)
CORROSION_MAX_AREA        = 8000  # Maximum — prevents giant background blobs
CORROSION_MORPH_KERNEL    = (9, 9)# Morphological kernel — larger = fewer tiny blobs
CORROSION_SEVERE_AREA     = 2000  # px² threshold for "High" severity
CORROSION_MEDIUM_AREA     = 500   # px² threshold for "Medium" severity

# Surface Wear / Line Detection
# FIX: original threshold was 10, causing 82%+ of the ROI to be flagged as wear
# (producing the large blue flood). We now use a per-frame statistical threshold
# (mean + N*std) so it adapts to each frame's actual lighting, not a hard value.
WEAR_SIGMA_MULTIPLIER     = 2.8   # Threshold = mean + N*std of combined signal
WEAR_MIN_AREA             = 300   # Minimum contour area (px²)
WEAR_MORPH_KERNEL         = (5, 5)
WEAR_SEVERE_AREA          = 1500  # px² threshold for "High" severity

# Crack / Linear Defect Detection
# FIX: original minLineLength=1 let HoughLinesP detect virtually every edge pixel
# as a "crack". Raised to a meaningful minimum and tightened Hough parameters.
CANNY_THRESHOLD1          = 40    # Lower Canny hysteresis threshold
CANNY_THRESHOLD2          = 120   # Upper Canny hysteresis threshold
HOUGH_VOTE_THRESHOLD      = 50    # Minimum Hough accumulator votes
HOUGH_MIN_LINE_LENGTH     = 30    # Minimum line length in px — key false-positive filter
HOUGH_MAX_LINE_GAP        = 8     # Max allowed gap within a line (px)
CRACK_MIN_LENGTH          = 30    # Minimum crack length to report
CRACK_SEVERE_LENGTH       = 80    # Length threshold for "High" severity

# Image Processing
BILATERAL_D               = 9
BILATERAL_SIGMA_COLOR     = 75
BILATERAL_SIGMA_SPACE     = 75
GAUSSIAN_KERNEL           = (3, 3)
LINE_DETECTION_KERNEL_SZ  = 15    # Horizontal/vertical morphological kernel

# Visualization
ROI_BOUNDARY_COLOR = (255, 200, 0)  # Cyan-yellow box around ROI
DEFECT_COLORS = {
    'Corrosion/Dark Spot':  {'bgr': (0, 60, 255),  'name': 'Red'},
    'Surface Wear/Lines':   {'bgr': (255, 80, 0),  'name': 'Blue'},
    'Crack/Linear Defect':  {'bgr': (0, 220, 50),  'name': 'Green'},
}
DEFECT_LABEL_FONT       = cv2.FONT_HERSHEY_SIMPLEX
DEFECT_LABEL_FONT_SCALE = 0.55
DEFECT_LABEL_THICKNESS  = 2
SUMMARY_FONT_SCALE      = 0.85
SUMMARY_FONT_THICKNESS  = 2
DEFECT_LABEL_MIN_AREA   = 300    # Only draw label text for defects above this area

# Live Video
LIVE_FRAME_WIDTH    = 640
LIVE_FRAME_HEIGHT   = 480
LIVE_FPS            = 15
RECORDING_DURATION  = 20         # seconds
VIDEO_FPS           = 15
OUTPUT_VIDEO_CODEC  = 'XVID'
BUTTON_COLOR        = (0, 200, 0)
BUTTON_INACTIVE_COLOR = (60, 60, 60)


# =============================================================================
# DETECTION ENGINE
# =============================================================================

class ThirdRailDefectDetector:
    def __init__(self, input_folder="images", output_folder="output"):
        self.input_folder  = input_folder
        self.output_folder = output_folder
        self.setup_output_folder()

    def setup_output_folder(self):
        for sub in ["processed", "comparisons", "live_captures", "live_recordings"]:
            Path(f"{self.output_folder}/{sub}").mkdir(parents=True, exist_ok=True)

    # ── ROI ──────────────────────────────────────────────────────────────────

    def extract_rail_roi(self, image):
        """Crop horizontally to the rail contact band."""
        _, width = image.shape[:2]
        x0 = int(width * ROI_START_PERCENT)
        x1 = min(int(width * (ROI_START_PERCENT + ROI_WIDTH_PERCENT)), width)
        return image[:, x0:x1], (x0, x1)

    # ── LED / GLARE REMOVAL ──────────────────────────────────────────────────

    def remove_led_glare(self, image):
        """
        Detect and inpaint LED glare.

        FIX: The original LED_LIGHT_THRESHOLD was 400, which exceeds the maximum
        uint8 pixel value of 255. The condition `gray > 400` is ALWAYS False, so
        led_mask was always all-zeros and inpainting never ran.
        Corrected threshold: 210 (targets genuine overexposed highlights only).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Binary mask of bright pixels
        _, bright_mask = cv2.threshold(gray, LED_BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Dilate to cover halo around glare spots
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (LED_KERNEL_SIZE, LED_KERNEL_SIZE))
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)

        glare_area = np.count_nonzero(bright_mask)
        if glare_area > LED_MIN_AREA:
            result = cv2.inpaint(image, bright_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        else:
            result = image.copy()

        return result, bright_mask

    # ── CORROSION DETECTION ──────────────────────────────────────────────────

    def detect_corrosion_spots(self, image):
        """
        Detect dark oxidation / burn marks on the rail surface.

        FIX: Original approach OR'd a global threshold (THRESH_BINARY_INV at 50)
        with an adaptive threshold. The global threshold on its own flagged the
        entire dark background of many images, producing dozens of spurious
        contours even on a clean rail. We now use adaptive-only with a larger
        block size and stricter C constant, paired with a bigger morphological
        kernel to merge noise into ignorable blobs and raise the minimum area.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Edge-preserving smoothing
        filtered = cv2.bilateralFilter(gray, BILATERAL_D,
                                       BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

        # Adaptive threshold only — responds to local illumination changes
        dark_mask = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            CORROSION_ADAPTIVE_BLOCK,
            CORROSION_ADAPTIVE_C
        )

        # Morphological cleanup with a larger kernel — removes salt-pepper noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CORROSION_MORPH_KERNEL)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN,  kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if CORROSION_MIN_AREA < area < CORROSION_MAX_AREA:
                x, y, w, h = cv2.boundingRect(c)
                # Reject extremely elongated blobs (likely noise lines, not spots)
                aspect = max(w, h) / (min(w, h) + 1e-6)
                if aspect < 8:
                    valid.append(c)

        return valid, dark_mask

    # ── SURFACE WEAR DETECTION ────────────────────────────────────────────────

    def detect_surface_wear_lines(self, image):
        """
        Detect worn / scratched surface texture.

        FIX: The original threshold was the constant 10, which is far below the
        typical signal mean (~18-22). At threshold=10 the wear mask covered 82%+
        of every frame, causing the massive blue flood on screen.

        New approach: compute the mean and standard deviation of the combined
        signal map for each frame and set threshold = mean + WEAR_SIGMA_MULTIPLIER*std.
        This adapts to each frame's actual lighting and only flags pixels that are
        statistically anomalous — preventing false positives on clean/new rail.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        blurred = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)

        # High-frequency texture (Laplacian of Gaussian style)
        lap_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        high_pass  = np.clip(np.abs(cv2.filter2D(blurred.astype(np.float32), -1,
                                                  lap_kernel.astype(np.float32))), 0, 255)

        # Morphological line detection
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_DETECTION_KERNEL_SZ, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_DETECTION_KERNEL_SZ))
        h_lines = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, hk).astype(np.float32)
        v_lines = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, vk).astype(np.float32)
        lines   = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0)

        # Laplacian edge energy
        laplacian = np.clip(np.abs(cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)), 0, 255)

        # Combine signals
        combined = (0.4 * high_pass + 0.3 * lines + 0.3 * laplacian).astype(np.uint8)

        # ── Per-frame adaptive threshold (the core fix) ─────────────────────
        mean_val = float(combined.mean())
        std_val  = float(combined.std())
        adaptive_threshold = mean_val + WEAR_SIGMA_MULTIPLIER * std_val
        adaptive_threshold = float(np.clip(adaptive_threshold, 30, 200))

        _, wear_mask = cv2.threshold(combined, adaptive_threshold, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, WEAR_MORPH_KERNEL)
        wear_mask = cv2.morphologyEx(wear_mask, cv2.MORPH_OPEN,  kernel)
        wear_mask = cv2.morphologyEx(wear_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(wear_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > WEAR_MIN_AREA:
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                if rw > 0 and rh > 0:
                    aspect = max(rw, rh) / (min(rw, rh) + 1e-6)
                    # Wear marks are moderately elongated (scratches, grooves)
                    if 1.2 < aspect < 10:
                        valid.append(c)

        return valid, wear_mask

    # ── CRACK DETECTION ───────────────────────────────────────────────────────

    def detect_cracks(self, image):
        """
        Detect cracks and linear structural defects.

        FIX: Original minLineLength=1 let HoughLinesP return hundreds of
        1-pixel edge fragments as "cracks". Raised to 30px minimum with a
        tighter vote threshold. Also removed the secondary contour-from-edges
        pass which duplicated and amplified the noise.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        filtered = cv2.bilateralFilter(gray, BILATERAL_D,
                                       BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

        edges = cv2.Canny(filtered, CANNY_THRESHOLD1, CANNY_THRESHOLD2, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=HOUGH_VOTE_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP
        )

        crack_contours = []
        line_mask = np.zeros_like(gray)

        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < CRACK_MIN_LENGTH:
                    continue

                # Angle filter — genuine cracks run diagonally.
                # Near-horizontal (0-15°) and near-vertical (75-90°) lines are
                # overwhelmingly the rail's own structural profile edges, not defects.
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                is_horizontal = angle < 15 or angle > 165
                is_vertical   = 75 < angle < 105
                if is_horizontal or is_vertical:
                    continue

                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                crack_contours.append(
                    np.array([[[x1, y1]], [[x2, y2]]], dtype=np.int32))

        return crack_contours, line_mask

    # ── CLASSIFICATION ────────────────────────────────────────────────────────

    def classify_defects(self, corrosion_contours, wear_contours, crack_contours):
        defects = []

        for c in corrosion_contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            severity = ("High"   if area > CORROSION_SEVERE_AREA else
                        "Medium" if area > CORROSION_MEDIUM_AREA else "Low")
            defects.append({'type': 'Corrosion/Dark Spot',
                            'contour': c, 'area': area,
                            'bbox': (x, y, w, h), 'severity': severity,
                            'color': DEFECT_COLORS['Corrosion/Dark Spot']['bgr']})

        for c in wear_contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            severity = "High" if area > WEAR_SEVERE_AREA else "Medium"
            defects.append({'type': 'Surface Wear/Lines',
                            'contour': c, 'area': area,
                            'bbox': (x, y, w, h), 'severity': severity,
                            'color': DEFECT_COLORS['Surface Wear/Lines']['bgr']})

        for c in crack_contours:
            x, y, w, h = cv2.boundingRect(c)
            length = max(w, h)
            area   = cv2.contourArea(c)
            severity = "High" if length > CRACK_SEVERE_LENGTH else "Medium"
            defects.append({'type': 'Crack/Linear Defect',
                            'contour': c, 'area': area,
                            'bbox': (x, y, w, h), 'length': length,
                            'severity': severity,
                            'color': DEFECT_COLORS['Crack/Linear Defect']['bgr']})

        return defects

    # ── VISUALISATION ─────────────────────────────────────────────────────────

    def visualize_defects(self, original_image, defects, roi_bounds):
        result = original_image.copy()
        x0, x1 = roi_bounds

        # Draw ROI boundary
        cv2.rectangle(result, (x0, 0), (x1, original_image.shape[0]),
                      ROI_BOUNDARY_COLOR, 2)

        counts = {k: 0 for k in DEFECT_COLORS}

        for d in defects:
            color   = d['color']
            contour = d['contour'].copy()
            contour[:, :, 0] += x0          # shift contour into full-image coords
            counts[d['type']] += 1

            # Semi-transparent fill
            overlay = result.copy()
            cv2.fillPoly(overlay, [contour], color)
            result = cv2.addWeighted(result, 0.82, overlay, 0.18, 0)
            cv2.drawContours(result, [contour], -1, color, 2)

            # Bounding box + label for significant defects only
            if d['area'] > DEFECT_LABEL_MIN_AREA:
                x, y, w, h = d['bbox']
                x += x0
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                label = f"{d['type'][:4]}-{d['severity'][0]}"
                cv2.putText(result, label, (x, max(y - 6, 12)),
                            DEFECT_LABEL_FONT, DEFECT_LABEL_FONT_SCALE,
                            color, DEFECT_LABEL_THICKNESS)

        # Summary overlay (top-left)
        y_off = 28
        cv2.putText(result, f"Defects: {len(defects)}", (10, y_off),
                    DEFECT_LABEL_FONT, SUMMARY_FONT_SCALE,
                    (255, 255, 255), SUMMARY_FONT_THICKNESS)
        for dtype, cnt in counts.items():
            if cnt > 0:
                y_off += 22
                cv2.putText(result, f"  {dtype}: {cnt}", (10, y_off),
                            DEFECT_LABEL_FONT, 0.45, (220, 220, 220), 1)

        return result

    # ── SIDE-BY-SIDE COMPARISON ───────────────────────────────────────────────

    def create_comparison(self, original, processed, label=""):
        oh, ow = original.shape[:2]
        ph, pw = processed.shape[:2]
        if oh != ph:
            th = min(oh, ph)
            original  = cv2.resize(original,  (int(ow * th / oh), th))
            processed = cv2.resize(processed, (int(pw * th / ph), th))

        gap  = np.full((original.shape[0], 10, 3), 100, dtype=np.uint8)
        comp = np.hstack([original, gap, processed])

        header = np.full((36, comp.shape[1], 3), 40, dtype=np.uint8)
        ow2 = original.shape[1]
        cv2.putText(header, "ORIGINAL",       (ow2 // 2 - 50, 24),
                    DEFECT_LABEL_FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(header, "DEFECT ANALYSIS", (ow2 + 10 + 50, 24),
                    DEFECT_LABEL_FONT, 0.7, (255, 255, 255), 2)

        footer = np.full((28, comp.shape[1], 3), 40, dtype=np.uint8)
        if label:
            cv2.putText(footer, label, (10, 20),
                        DEFECT_LABEL_FONT, 0.5, (180, 180, 180), 1)

        return np.vstack([header, comp, footer])

    # ── FULL PIPELINE (single image) ──────────────────────────────────────────

    def process_image(self, image):
        """Run the full detection pipeline on a BGR image array."""
        original = image.copy()

        roi, roi_bounds          = self.extract_rail_roi(image)
        deglared, _              = self.remove_led_glare(roi)
        corr_c, _                = self.detect_corrosion_spots(deglared)
        wear_c, _                = self.detect_surface_wear_lines(deglared)
        crack_c, _               = self.detect_cracks(deglared)
        defects                  = self.classify_defects(corr_c, wear_c, crack_c)
        result                   = self.visualize_defects(original, defects, roi_bounds)

        return result, defects, roi_bounds

    def process_image_file(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading: {image_path}")
            return None
        result, defects, _ = self.process_image(image)
        print(f"{os.path.basename(image_path)}: {len(defects)} defects "
              f"({sum(1 for d in defects if d['type']=='Corrosion/Dark Spot')} corrosion, "
              f"{sum(1 for d in defects if d['type']=='Surface Wear/Lines')} wear, "
              f"{sum(1 for d in defects if d['type']=='Crack/Linear Defect')} cracks)")
        return {'original': image, 'processed': result, 'defects': defects}

    def process_all_images(self):
        paths = sorted(glob.glob(f"{self.input_folder}/*.jpg") +
                       glob.glob(f"{self.input_folder}/*.png"))
        if not paths:
            print(f"No images found in {self.input_folder}/")
            return

        print(f"Processing {len(paths)} images …")
        for p in paths:
            r = self.process_image_file(p)
            if r is None:
                continue
            name = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(f"{self.output_folder}/processed/{name}_defects.jpg", r['processed'])
            comp = self.create_comparison(r['original'], r['processed'], os.path.basename(p))
            cv2.imwrite(f"{self.output_folder}/comparisons/{name}_comparison.jpg", comp)

        print("Done.")


# =============================================================================
# LIVE VIDEO DETECTOR
# =============================================================================

class LiveRailDefectDetector(ThirdRailDefectDetector):

    def __init__(self, camera_index=0, **kwargs):
        super().__init__(**kwargs)
        self.camera_index        = camera_index
        self.cap                 = None
        self.recording           = False
        self.video_writer        = None
        self.recording_start     = 0
        self.last_comparison     = None

    # ── VIDEO WRITER ─────────────────────────────────────────────────────────

    def _make_writer(self, filename, w, h):
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
        return cv2.VideoWriter(filename, fourcc, VIDEO_FPS, (w, h))

    # ── CONTROL PANEL ────────────────────────────────────────────────────────

    def _draw_controls(self, frame):
        rec_color = (0, 0, 220) if self.recording else BUTTON_INACTIVE_COLOR
        cv2.rectangle(frame, (10, 10), (160, 58), rec_color, -1)
        cv2.putText(frame, "RECORDING" if self.recording else "RECORD",
                    (18, 40), DEFECT_LABEL_FONT, 0.58, (255, 255, 255), 2)

        cv2.rectangle(frame, (175, 10), (315, 58), BUTTON_COLOR, -1)
        cv2.putText(frame, "CAPTURE", (190, 40),
                    DEFECT_LABEL_FONT, 0.58, (0, 0, 0), 2)

        cv2.rectangle(frame, (330, 10), (450, 58), (0, 0, 200), -1)
        cv2.putText(frame, "EXIT", (358, 40),
                    DEFECT_LABEL_FONT, 0.58, (255, 255, 255), 2)

        # Recording timer
        if self.recording:
            elapsed = int(time.time() - self.recording_start)
            remaining = max(0, RECORDING_DURATION - elapsed)
            cv2.putText(frame, f"{remaining}s", (465, 40),
                        DEFECT_LABEL_FONT, 0.65, (0, 0, 220), 2)
        return frame

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if 10 <= x <= 160 and 10 <= y <= 58:
            self.stop_recording() if self.recording else self.start_recording()
        elif 175 <= x <= 315 and 10 <= y <= 58:
            self._capture()
        elif 330 <= x <= 450 and 10 <= y <= 58:
            self._quit()

    # ── RECORDING / CAPTURE ──────────────────────────────────────────────────

    def start_recording(self):
        if self.last_comparison is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"{self.output_folder}/live_recordings/rec_{ts}.avi"
        h, w = self.last_comparison.shape[:2]
        self.video_writer    = self._make_writer(fn, w, h)
        self.recording       = True
        self.recording_start = time.time()
        print(f"Recording → {fn}")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        print("Recording saved.")

    def _capture(self):
        if self.last_comparison is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"{self.output_folder}/live_captures/capture_{ts}.jpg"
        cv2.imwrite(fn, self.last_comparison)
        print(f"Captured → {fn}")

    def _quit(self):
        self._cleanup()
        cv2.destroyAllWindows()
        raise SystemExit

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────

    def process_live_video(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Cannot open camera {self.camera_index}")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  LIVE_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LIVE_FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          LIVE_FPS)

        win = 'Third Rail Defect Detection'
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._on_mouse)

        print("Live detection active.")
        print("Controls: R = Record | C = Capture | Q = Quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera read error.")
                break

            processed, _, _ = self.process_image(frame)
            comp             = self.create_comparison(frame, processed, "Live")
            self.last_comparison = comp

            display = self._draw_controls(comp.copy())
            cv2.imshow(win, display)

            if self.recording:
                self.video_writer.write(comp)
                if time.time() - self.recording_start >= RECORDING_DURATION:
                    self.stop_recording()

            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'):  break
            elif key == ord('r'):  self.stop_recording() if self.recording else self.start_recording()
            elif key == ord('c'):  self._capture()

        self._cleanup()
        cv2.destroyAllWindows()

    def _cleanup(self):
        if self.recording:
            self.stop_recording()
        if self.cap:
            self.cap.release()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("=" * 52)
    print("  Third Rail Defect Detection System  (v2.0)")
    print("=" * 52)
    print("  1. Batch process image folder")
    print("  2. Live webcam detection")
    print("  3. Exit")
    choice = input("Select mode (1-3): ").strip()

    if choice == '1':
        ThirdRailDefectDetector(
            input_folder="images", output_folder="output"
        ).process_all_images()
    elif choice == '2':
        LiveRailDefectDetector(
            camera_index=0, input_folder="images", output_folder="output"
        ).process_live_video()
    elif choice == '3':
        return
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    for d in ["images", "output/processed", "output/comparisons",
              "output/live_captures", "output/live_recordings"]:
        os.makedirs(d, exist_ok=True)
    main()
