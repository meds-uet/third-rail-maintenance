"""
Third Rail Defect Detection System  —  v4.1
===========================================
Detects sparking marks, corrosion, surface wear, and cracks on METRO Orange Line
third-rail imagery.

Fix history vs original script
-------------------------------
v1  LED threshold 400 → 210  (original > 255, never fired on uint8 images)
v1  Wear threshold 10  → per-frame mean+2.8σ  (original flooded 82% of frame blue)
v1  Crack minLineLength 1 → 30 + diagonal-angle filter  (original: every edge = crack)
v2  Fixed ROI 40-80% → auto-detected rail band via column brightness profiling
v2  Added column gate: exclude dim rail-edge fade zones from detection
v2  Replaced adaptive threshold with Gaussian local-contrast detector
v3  Added secondary absolute-dark detector for uniformly-corroded surfaces
v3  Added row gate (exclude top/bottom 8%) to suppress tunnel ceiling/floor noise
v3  Relaxed aspect ratio to 5.0 for secondary detector to catch narrow spark trails
v3  Added very-dark-pixel fraction filter (≥3% of region pixels < 20) to reject
    bright structural features that happen to have a dark minimum pixel
v4  Reverted auto-ROI to fixed percentages (ROI_START_PCT / ROI_WIDTH_PCT).
    In trolley deployment the rail always falls in the same band — no need
    for dynamic detection. Eliminates false positives outside the rail.

"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime
import time

# =============================================================================
# CONFIGURABLE MACROS  — all tuning lives here, not inside functions
# =============================================================================

# ── Fixed ROI ─────────────────────────────────────────────────────────────────
# Camera is mounted on a trolley at fixed distance → rail always in the same window.
# Tune all four values once to match your mounting geometry.
ROI_START_PCT       = 0.55   # left edge of ROI as fraction of frame width
ROI_WIDTH_PCT       = 0.40   # width of ROI as fraction of frame width
ROI_TOP_PCT         = 0.20   # top edge of ROI as fraction of frame height
ROI_HEIGHT_PCT      = 0.60   # height of ROI as fraction of frame height

# ── Glare removal ─────────────────────────────────────────────────────────────
GLARE_THRESH        = 205    # pixels above this = LED glare  (uint8, 0-255)
GLARE_DILATE_PX     = 31     # halo dilation — keep SMALL so adjacent defects survive
GLARE_MIN_AREA      = 100
INPAINT_RADIUS      = 12

# ── Primary dark-spot detector  (local Gaussian contrast) ────────────────────
LOCAL_BG_KERNEL     = (101, 101)  # large Gaussian for background estimate
LOCAL_DARK_T        = 12          # pixel must be this much darker than neighbourhood
SPOT_MIN_AREA       = 400         # px²  — filters isolated bolt holes / texture noise
SPOT_MAX_AREA       = 80000
SPOT_MAX_ASPECT     = 5.0         # h/w  — rejects vertical rail-joint lines
SPOT_DARK_FRAC      = 0.45        # region min pixel < this × rail_mean
SPOT_VD_FRAC        = 0.03        # ≥ this fraction of region pixels must be < 20

# ── Secondary dark-spot detector  (absolute darkness within lit zone) ─────────
ABS_DARK_FRAC       = 0.30        # pixel < this × rail_mean = "absolutely dark"
ABS_DARK_MIN_ABS    = 20          # absolute floor for the threshold
ABS_SPOT_MAX_ASPECT = 5.0

# ── Severity thresholds for sparking / corrosion ──────────────────────────────
SPOT_SEVERE_AREA    = 5000
SPOT_MEDIUM_AREA    = 1500

# ── Surface wear detection ────────────────────────────────────────────────────
WEAR_SIGMA_MULT     = 2.8         # threshold = mean + N×std of texture signal
WEAR_MIN_AREA       = 300
WEAR_MORPH_K        = (5, 5)
WEAR_SEVERE_AREA    = 1500

# ── Crack / linear defect detection ──────────────────────────────────────────
CANNY_T1            = 40
CANNY_T2            = 120
HOUGH_VOTES         = 50
HOUGH_MIN_LEN       = 30
HOUGH_MAX_GAP       = 8
CRACK_MIN_LEN       = 30
CRACK_SEVERE_LEN    = 80
CRACK_ANGLE_MARGIN  = 15   # degrees — lines within this of H or V are rejected

# ── Bilateral / Gaussian ──────────────────────────────────────────────────────
BIL_D, BIL_SC, BIL_SS = 9, 75, 75
GAUSS_K             = (3, 3)
LINE_DETECT_K       = 15

# ── Visualization ─────────────────────────────────────────────────────────────
ROI_BOX_COLOR       = (0, 220, 255)
DEFECT_COLORS = {
    'Sparking/Corrosion':  {'bgr': (0,  50, 255), 'name': 'Red'},
    'Surface Wear/Lines':  {'bgr': (255, 80,  0), 'name': 'Blue'},
    'Crack/Linear Defect': {'bgr': (0, 220,  50), 'name': 'Green'},
}
FONT            = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE     = 0.55
LABEL_THICK     = 2
SUMMARY_SCALE   = 0.80
LABEL_MIN_AREA  = 300

# ── Live video ────────────────────────────────────────────────────────────────
LIVE_W, LIVE_H  = 640, 480
LIVE_FPS        = 15
REC_DURATION    = 20
VIDEO_FPS       = 15
VIDEO_CODEC     = 'XVID'
BTN_ON          = (0, 200, 0)
BTN_OFF         = (60, 60, 60)


# =============================================================================
# HELPER
# =============================================================================

def _morph_open_close(mask, k):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def _glare_suppress(roi_gray):
    """Return dilated glare mask for a grayscale ROI."""
    _, gm = cv2.threshold(roi_gray, GLARE_THRESH, 255, cv2.THRESH_BINARY)
    return cv2.dilate(gm, np.ones((GLARE_DILATE_PX, GLARE_DILATE_PX), np.uint8))


# =============================================================================
# DETECTOR
# =============================================================================

class ThirdRailDefectDetector:

    def __init__(self, input_folder="images", output_folder="output"):
        self.input_folder  = input_folder
        self.output_folder = output_folder
        for sub in ["processed","comparisons","live_captures","live_recordings"]:
            Path(f"{self.output_folder}/{sub}").mkdir(parents=True, exist_ok=True)

    # ── 1. FIXED ROI ─────────────────────────────────────────────────────────

    def extract_rail_roi(self, image):
        """
        Crop to a fixed 2-D window: horizontal (ROI_START_PCT / ROI_WIDTH_PCT)
        and vertical (ROI_TOP_PCT / ROI_HEIGHT_PCT).
        Trimming both axes eliminates false positives on the top/bottom edges
        of the frame where the rail is not present in trolley deployment.
        Adjust all four macros once to match your camera mounting.
        """
        h, w = image.shape[:2]
        x0 = int(w * ROI_START_PCT)
        x1 = min(int(w * (ROI_START_PCT + ROI_WIDTH_PCT)), w)
        y0 = int(h * ROI_TOP_PCT)
        y1 = min(int(h * (ROI_TOP_PCT + ROI_HEIGHT_PCT)), h)
        return image[y0:y1, x0:x1], (x0, x1, y0, y1)

    def _col_gate(self, roi_gray):
        """All columns within the fixed ROI are valid — return all-ones mask."""
        return np.ones(roi_gray.shape, dtype=np.uint8)

    def _row_gate(self, roi_gray):
        """All rows within the fixed ROI are valid — return all-ones mask."""
        return np.ones(roi_gray.shape, dtype=np.uint8)

    # ── 2. GLARE REMOVAL ─────────────────────────────────────────────────────

    def remove_glare(self, image):
        gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
        mask_d = _glare_suppress(gray)
        if int(mask_d.sum()//255) > GLARE_MIN_AREA:
            result = cv2.inpaint(image, mask_d, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        else:
            result = image.copy()
        return result, mask_d

    # ── 3a. PRIMARY DARK-SPOT DETECTOR  (local Gaussian contrast) ────────────

    def detect_dark_spots_local(self, roi_gray, col_gate, row_gate, glare_mask, rail_mean):
        """
        Flag pixels significantly darker than their 101px Gaussian neighbourhood.
        Catches sparking clusters and corrosion against a lit background.
        """
        filtered  = cv2.bilateralFilter(roi_gray, BIL_D, BIL_SC, BIL_SS)
        bg        = cv2.GaussianBlur(filtered.astype(np.float32), LOCAL_BG_KERNEL, 0)
        ld        = np.clip(bg - filtered.astype(np.float32), 0, 255).astype(np.uint8)

        _, mask   = cv2.threshold(ld, LOCAL_DARK_T, 255, cv2.THRESH_BINARY)
        mask      = mask & (col_gate * 255) & (row_gate * 255)
        mask      = cv2.bitwise_and(mask, cv2.bitwise_not(glare_mask))

        k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask      = _morph_open_close(mask, k)

        cntrs, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._filter_spot_contours(cntrs, roi_gray, rail_mean, SPOT_MAX_ASPECT)

    # ── 3b. SECONDARY DARK-SPOT DETECTOR  (absolute darkness) ────────────────

    def detect_dark_spots_absolute(self, roi_gray, col_gate, row_gate, glare_mask, rail_mean):
        """
        Flag pixels below an absolute darkness level relative to the rail mean.
        Catches uniformly-corroded rail surfaces where local contrast is weak
        because the ENTIRE area is darkened by heavy oxidation or spark trails.
        """
        thresh = max(rail_mean * ABS_DARK_FRAC, ABS_DARK_MIN_ABS)
        _, mask = cv2.threshold(roi_gray, int(thresh), 255, cv2.THRESH_BINARY_INV)
        mask    = mask & (col_gate * 255) & (row_gate * 255)
        mask    = cv2.bitwise_and(mask, cv2.bitwise_not(glare_mask))

        k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask    = _morph_open_close(mask, k)

        cntrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._filter_spot_contours(cntrs, roi_gray, rail_mean, ABS_SPOT_MAX_ASPECT)

    def _filter_spot_contours(self, cntrs, roi_gray, rail_mean, max_aspect):
        """
        Common filter applied to both detectors:
          • area in [SPOT_MIN_AREA, SPOT_MAX_AREA]
          • aspect ratio h/w ≤ max_aspect   (rejects vertical rail-joint lines)
          • region minimum pixel < rail_mean × SPOT_DARK_FRAC  (genuinely dark)
          • ≥ SPOT_VD_FRAC of region pixels < 20  (rejects bright structural features)
        """
        valid = []
        for c in cntrs:
            area = cv2.contourArea(c)
            if not (SPOT_MIN_AREA < area < SPOT_MAX_AREA):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h / (w + 1e-6) > max_aspect:
                continue
            reg = roi_gray[y:y+h, x:x+w]
            if float(reg.min()) > rail_mean * SPOT_DARK_FRAC:
                continue
            if (reg < 20).mean() < SPOT_VD_FRAC:
                continue
            valid.append(c)
        return valid

    # ── 4. SURFACE WEAR ───────────────────────────────────────────────────────

    def detect_surface_wear(self, roi_gray, col_gate, glare_mask):
        blurred = cv2.GaussianBlur(roi_gray, GAUSS_K, 0)

        hp_k    = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)
        hp      = np.clip(np.abs(cv2.filter2D(blurred.astype(np.float32), -1, hp_k)), 0, 255)

        hk      = cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_DETECT_K, 1))
        vk      = cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_DETECT_K))
        h_lines = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, hk).astype(np.float32)
        v_lines = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, vk).astype(np.float32)
        lines   = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0)

        lap     = np.clip(np.abs(cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)), 0, 255)
        comb    = (0.4*hp + 0.3*lines + 0.3*lap).astype(np.uint8)

        m, s    = float(comb.mean()), float(comb.std())
        t       = float(np.clip(m + WEAR_SIGMA_MULT * s, 30, 200))

        _, mask = cv2.threshold(comb, t, 255, cv2.THRESH_BINARY)
        mask    = mask & (col_gate * 255)
        mask    = cv2.bitwise_and(mask, cv2.bitwise_not(glare_mask))

        k       = cv2.getStructuringElement(cv2.MORPH_RECT, WEAR_MORPH_K)
        mask    = _morph_open_close(mask, k)

        cntrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in cntrs:
            area = cv2.contourArea(c)
            if area < WEAR_MIN_AREA:
                continue
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            if rw > 0 and rh > 0 and 1.2 < max(rw,rh)/(min(rw,rh)+1e-6) < 10:
                valid.append(c)
        return valid

    # ── 5. CRACKS ─────────────────────────────────────────────────────────────

    def detect_cracks(self, roi_gray, col_gate):
        filtered  = cv2.bilateralFilter(roi_gray, BIL_D, BIL_SC, BIL_SS)
        edges     = cv2.Canny(filtered, CANNY_T1, CANNY_T2, apertureSize=3)
        edges     = edges & (col_gate * 255)

        lines     = cv2.HoughLinesP(edges, 1, np.pi/180,
                                    threshold=HOUGH_VOTES,
                                    minLineLength=HOUGH_MIN_LEN,
                                    maxLineGap=HOUGH_MAX_GAP)
        contours  = []
        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                if float(np.hypot(x2-x1, y2-y1)) < CRACK_MIN_LEN:
                    continue
                ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                if (ang < CRACK_ANGLE_MARGIN or ang > 180-CRACK_ANGLE_MARGIN or
                        90-CRACK_ANGLE_MARGIN < ang < 90+CRACK_ANGLE_MARGIN):
                    continue
                contours.append(np.array([[[x1,y1]],[[x2,y2]]], dtype=np.int32))
        return contours

    # ── 6. CLASSIFY ───────────────────────────────────────────────────────────

    def classify(self, spot_c, wear_c, crack_c):
        defects = []

        for c in spot_c:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            sev  = ("High"   if area > SPOT_SEVERE_AREA  else
                    "Medium" if area > SPOT_MEDIUM_AREA  else "Low")
            defects.append({'type':'Sparking/Corrosion','contour':c,'area':area,
                            'bbox':(x,y,w,h),'severity':sev,
                            'color':DEFECT_COLORS['Sparking/Corrosion']['bgr']})

        for c in wear_c:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            defects.append({'type':'Surface Wear/Lines','contour':c,'area':area,
                            'bbox':(x,y,w,h),
                            'severity':"High" if area > WEAR_SEVERE_AREA else "Medium",
                            'color':DEFECT_COLORS['Surface Wear/Lines']['bgr']})

        for c in crack_c:
            x,y,w,h = cv2.boundingRect(c)
            length   = max(w,h)
            defects.append({'type':'Crack/Linear Defect','contour':c,
                            'area':cv2.contourArea(c),'bbox':(x,y,w,h),'length':length,
                            'severity':"High" if length > CRACK_SEVERE_LEN else "Medium",
                            'color':DEFECT_COLORS['Crack/Linear Defect']['bgr']})

        return defects

    # ── 7. VISUALIZE ──────────────────────────────────────────────────────────

    def visualize(self, original, defects, roi_bounds):
        result         = original.copy()
        x0, x1, y0, y1 = roi_bounds
        cv2.rectangle(result,(x0,y0),(x1,y1),ROI_BOX_COLOR,2)

        counts = {k:0 for k in DEFECT_COLORS}
        for d in defects:
            color   = d['color']
            contour = d['contour'].copy()
            contour[:,:,0] += x0
            contour[:,:,1] += y0
            counts[d['type']] += 1

            overlay = result.copy()
            cv2.fillPoly(overlay, [contour], color)
            result  = cv2.addWeighted(result, 0.82, overlay, 0.18, 0)
            cv2.drawContours(result,[contour],-1,color,2)

            if d['area'] > LABEL_MIN_AREA:
                x,y,w,h = d['bbox']
                x += x0
                y += y0
                cv2.rectangle(result,(x,y),(x+w,y+h),color,2)
                cv2.putText(result,f"{d['type'][:5]}-{d['severity'][0]}",
                            (x,max(y-6,y0+12)),FONT,LABEL_SCALE,color,LABEL_THICK)

        # ── Legend panel ────────────────────────────────────────────────
        # Draw a semi-transparent dark background strip so text is readable
        # over any rail surface brightness.
        legend_w   = 310
        legend_h   = 28 + len(DEFECT_COLORS) * 28 + 10
        overlay_lg = result.copy()
        cv2.rectangle(overlay_lg, (4, 4), (4 + legend_w, 4 + legend_h), (20, 20, 20), -1)
        result     = cv2.addWeighted(result, 0.35, overlay_lg, 0.65, 0)

        y_off = 24
        cv2.putText(result, f"Defects detected: {len(defects)}",
                    (10, y_off), FONT, SUMMARY_SCALE, (255, 255, 255), 2)

        for dt, info in DEFECT_COLORS.items():
            y_off  += 28
            bgr     = info['bgr']
            cname   = info['name']          # human-readable colour name
            cnt     = counts[dt]

            # Filled colour swatch
            sx = 10
            cv2.rectangle(result, (sx, y_off - 13), (sx + 18, y_off + 5), bgr, -1)
            cv2.rectangle(result, (sx, y_off - 13), (sx + 18, y_off + 5), (255,255,255), 1)

            # Label: "Defect type (Colour): count"
            label = f"{dt} ({cname}): {cnt}"
            cv2.putText(result, label, (sx + 24, y_off),
                        FONT, 0.46, (255, 255, 255), 1)

        return result

    # ── 8. COMPARISON ─────────────────────────────────────────────────────────

    def create_comparison(self, original, processed, label=""):
        oh,ow = original.shape[:2]
        ph,pw = processed.shape[:2]
        if oh!=ph:
            th = min(oh,ph)
            original  = cv2.resize(original,  (int(ow*th/oh),th))
            processed = cv2.resize(processed, (int(pw*th/ph),th))
        gap  = np.full((original.shape[0],10,3),100,dtype=np.uint8)
        comp = np.hstack([original,gap,processed])
        hdr  = np.full((36,comp.shape[1],3),40,dtype=np.uint8)
        cv2.putText(hdr,"ORIGINAL",        (original.shape[1]//2-50,24),FONT,0.7,(255,255,255),2)
        cv2.putText(hdr,"DEFECT ANALYSIS", (original.shape[1]+60,24), FONT,0.7,(255,255,255),2)
        ftr  = np.full((28,comp.shape[1],3),40,dtype=np.uint8)
        if label:
            cv2.putText(ftr,label,(10,20),FONT,0.5,(180,180,180),1)
        return np.vstack([hdr,comp,ftr])

    # ── 9. FULL PIPELINE ──────────────────────────────────────────────────────

    def process_image(self, image):
        """Run complete detection pipeline. Returns (annotated_image, defects, roi_bounds)."""
        original            = image.copy()
        roi, roi_bounds     = self.extract_rail_roi(image)
        x0, x1, y0, y1     = roi_bounds

        roi_gray            = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        col_gate            = self._col_gate(roi_gray)
        row_gate            = self._row_gate(roi_gray)

        # Glare removal
        deglared, _         = self.remove_glare(roi)
        dg_gray             = cv2.cvtColor(deglared, cv2.COLOR_BGR2GRAY)
        glare_mask          = _glare_suppress(dg_gray)

        # Rail mean (used by both spot detectors)
        lit_pixels          = dg_gray[col_gate.astype(bool)]
        rail_mean           = float(lit_pixels.mean()) if lit_pixels.size > 0 else float(dg_gray.mean())

        # Run all detectors
        spot_local          = self.detect_dark_spots_local(dg_gray, col_gate, row_gate, glare_mask, rail_mean)
        spot_abs            = self.detect_dark_spots_absolute(dg_gray, col_gate, row_gate, glare_mask, rail_mean)
        wear_c              = self.detect_surface_wear(dg_gray, col_gate, glare_mask)
        crack_c             = self.detect_cracks(dg_gray, col_gate)

        # Merge spot detectors (simple concatenation — both pass same filters)
        all_spots           = spot_local + spot_abs

        defects             = self.classify(all_spots, wear_c, crack_c)
        result              = self.visualize(original, defects, roi_bounds)
        return result, defects, roi_bounds

    def process_image_file(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Cannot load: {image_path}"); return None
        result, defects, _ = self.process_image(image)
        sp = sum(1 for d in defects if d['type']=='Sparking/Corrosion')
        wr = sum(1 for d in defects if d['type']=='Surface Wear/Lines')
        cr = sum(1 for d in defects if d['type']=='Crack/Linear Defect')
        print(f"  {os.path.basename(image_path)}: {len(defects)} defects  "
              f"({sp} spark/corr, {wr} wear, {cr} crack)")
        return {'original':image,'processed':result,'defects':defects}

    def process_all_images(self):
        paths = sorted(glob.glob(f"{self.input_folder}/*.jpg") +
                       glob.glob(f"{self.input_folder}/*.png"))
        if not paths:
            print(f"No images found in {self.input_folder}/"); return
        print(f"Processing {len(paths)} images …\n")
        for p in paths:
            r = self.process_image_file(p)
            if r is None: continue
            stem = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(f"{self.output_folder}/processed/{stem}_defects.jpg",    r['processed'])
            comp = self.create_comparison(r['original'],r['processed'],os.path.basename(p))
            cv2.imwrite(f"{self.output_folder}/comparisons/{stem}_comparison.jpg", comp)
        print("\nDone.")


# =============================================================================
# LIVE DETECTOR
# =============================================================================

class LiveRailDefectDetector(ThirdRailDefectDetector):

    def __init__(self, camera_index=0, **kwargs):
        super().__init__(**kwargs)
        self.camera_index = camera_index
        self.cap = self.video_writer = None
        self.recording = False
        self.rec_start = 0
        self.last_comp = None

    def _make_writer(self, path, w, h):
        return cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*VIDEO_CODEC),VIDEO_FPS,(w,h))

    def _draw_controls(self, frame):
        rc = (0,0,200) if self.recording else BTN_OFF
        cv2.rectangle(frame,(10,10),(165,56),rc,-1)
        cv2.putText(frame,"RECORDING" if self.recording else "RECORD",(18,38),FONT,0.56,(255,255,255),2)
        cv2.rectangle(frame,(178,10),(318,56),BTN_ON,-1)
        cv2.putText(frame,"CAPTURE",(193,38),FONT,0.56,(0,0,0),2)
        cv2.rectangle(frame,(330,10),(450,56),(0,0,180),-1)
        cv2.putText(frame,"EXIT",(356,38),FONT,0.56,(255,255,255),2)
        if self.recording:
            cv2.putText(frame,f"{max(0,REC_DURATION-int(time.time()-self.rec_start))}s",
                        (460,38),FONT,0.65,(0,0,220),2)
        return frame

    def _on_mouse(self, event, x, y, flags, param):
        if event!=cv2.EVENT_LBUTTONDOWN: return
        if   10<=x<=165 and 10<=y<=56:  self.stop_recording() if self.recording else self.start_recording()
        elif 178<=x<=318 and 10<=y<=56: self._capture()
        elif 330<=x<=450 and 10<=y<=56: self._quit()

    def start_recording(self):
        if self.last_comp is None: return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"{self.output_folder}/live_recordings/rec_{ts}.avi"
        h,w = self.last_comp.shape[:2]
        self.video_writer = self._make_writer(fn,w,h)
        self.recording    = True
        self.rec_start    = time.time()
        print(f"Recording → {fn}")

    def stop_recording(self):
        if self.video_writer: self.video_writer.release(); self.video_writer=None
        self.recording = False; print("Recording saved.")

    def _capture(self):
        if self.last_comp is None: return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"{self.output_folder}/live_captures/capture_{ts}.jpg"
        cv2.imwrite(fn,self.last_comp); print(f"Captured → {fn}")

    def _quit(self):
        self._cleanup(); cv2.destroyAllWindows(); raise SystemExit

    def process_live_video(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened(): print(f"Cannot open camera {self.camera_index}"); return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  LIVE_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LIVE_H)
        self.cap.set(cv2.CAP_PROP_FPS,          LIVE_FPS)
        win = "Third Rail Defect Detection v3"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win,self._on_mouse)
        print("Live — R=Record  C=Capture  Q=Quit")
        while True:
            ret,frame = self.cap.read()
            if not ret: print("Camera read error."); break
            processed,_,_ = self.process_image(frame)
            comp           = self.create_comparison(frame,processed,"Live")
            self.last_comp = comp
            cv2.imshow(win,self._draw_controls(comp.copy()))
            if self.recording:
                self.video_writer.write(comp)
                if time.time()-self.rec_start >= REC_DURATION: self.stop_recording()
            key = cv2.waitKey(1)&0xFF
            if   key==ord('q'): break
            elif key==ord('r'): self.stop_recording() if self.recording else self.start_recording()
            elif key==ord('c'): self._capture()
        self._cleanup(); cv2.destroyAllWindows()

    def _cleanup(self):
        if self.recording: self.stop_recording()
        if self.cap: self.cap.release()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    print("="*56)
    print("  Third Rail Defect Detection System  (v4.1)")
    print("="*56)
    print("  1. Batch process image folder")
    print("  2. Live webcam / camera detection")
    print("  3. Exit")
    c = input("Select mode (1-3): ").strip()
    if   c=='1': ThirdRailDefectDetector(input_folder="images",output_folder="output").process_all_images()
    elif c=='2': LiveRailDefectDetector(camera_index=0,input_folder="images",output_folder="output").process_live_video()
    elif c=='3': return
    else: print("Invalid.")

if __name__=="__main__":
    for d in ["images","output/processed","output/comparisons",
              "output/live_captures","output/live_recordings"]:
        os.makedirs(d,exist_ok=True)
    main()
