# 🚇 Third Rail Defect Detection System (TRDDS)

Automated image-processing system for detecting surface defects on the **METRO Orange Line third rail**, developed by the [MEDS Lab](https://uet.edu.pk) at UET Lahore.

The system identifies three categories of defects:

| Defect | Colour | Description |
|--------|--------|-------------|
| **Sparking / Corrosion** | 🔴 Red | Dark oxidation patches and burn marks from electrical sparking |
| **Surface Wear / Lines** | 🔵 Blue | Scratches, grooves, and mechanical wear patterns |
| **Crack / Linear Defect** | 🟢 Green | Diagonal structural cracks in the rail surface |

---

## 🌟 Key Features

- **Dual operation modes** — batch processing of image folders and live camera feed
- **Gaussian local-contrast detection** — adapts to each frame's actual brightness instead of fixed global thresholds
- **Dual-pass dark-spot detection** — catches both locally-dark sparking clusters and uniformly-corroded surfaces
- **LED glare removal** — inpaints bright reflections before detection runs
- **Diagonal-only crack filter** — rejects rail-profile edges, only flags genuine structural cracks
- **Colour-coded legend overlay** — on-screen panel shows swatch + colour name + count for each defect type
- **Fixed ROI** — simple percentage crop designed for trolley-mounted camera deployment
- **Optimised for Raspberry Pi 5** — live mode runs at 10–15 FPS at 640×480

---

## 🛠️ Installation

### Requirements
- Python 3.8+
- OpenCV 4.5+
- NumPy

```bash
git clone https://github.com/meds-uet/third-rail-maintenance.git
cd third-rail-maintenance
pip3 install opencv-python numpy
```

---

## 🚀 Usage

### Mode 1 — Batch Image Processing (`TRD_training.py`)

Processes every image in a folder and writes annotated outputs.

```bash
# Place rail images in the images/ folder, then:
python3 TRD_training.py
```

Outputs written to:

```
output/
├── processed/      # Annotated images with defect overlays
└── comparisons/    # Side-by-side original vs. detected
```

---

### Mode 2 — Live Camera / Video (`TRD_video.py`)

Real-time detection on a connected camera with an interactive GUI.

```bash
python3 TRD_video.py
# Select option 2 at the menu
```

**On-screen controls:**

| Button | Keyboard | Action |
|--------|----------|--------|
| RECORD | `R` | Start / stop a 20-second video clip |
| CAPTURE | `C` | Save the current frame as a JPEG |
| EXIT | `Q` | Quit the application |

Live outputs saved to:

```
output/
├── live_captures/    # Still images
└── live_recordings/  # Video clips (.avi)
```

---

## 📂 Repository Structure

```
third-rail-maintenance/
├── TRD_training.py          # Batch image processing engine
├── TRD_video.py             # Live camera detection + GUI
walkthrough
├── data/                    # 30+ real METRO Orange Line rail images
├── output/
│   ├── processed/           # Annotated defect images
│   ├── comparisons/         # Before/after side-by-side
│   ├── live_captures/       # Webcam still captures
│   └── live_recordings/     # Recorded video clips
```

---

## ⚙️ Configuration

All tuning parameters are grouped at the top of `TRD_video.py`. Edit them once — no changes needed inside functions.

### ROI (Region of Interest)

The camera is mounted on a trolley at a fixed distance, so the rail always falls in the same horizontal band. Set these two values to match your mounting geometry and leave them.

```python
ROI_START_PCT  = 0.50   # Left edge of ROI as fraction of frame width
ROI_WIDTH_PCT  = 0.30   # Width of ROI as fraction of frame width
```

### Glare Removal

```python
GLARE_THRESH   = 205    # Pixels brighter than this are treated as LED glare (0-255)
GLARE_DILATE_PX = 31   # Dilation radius around glare spots (px)
INPAINT_RADIUS  = 12   # Inpainting fill radius (px)
```

> **Note:** The original script used `LED_LIGHT_THRESHOLD = 400`, which is above the maximum uint8 value of 255 and therefore never fired. This has been corrected.

### Sparking / Corrosion Detection (Primary — local contrast)

Detects pixels significantly darker than their 101px Gaussian neighbourhood. Catches sparking clusters and corrosion against a brighter background.

```python
LOCAL_DARK_T    = 12    # How much darker than neighbourhood to flag (px intensity units)
SPOT_MIN_AREA   = 400   # Minimum contour area in px² (filters bolt holes / texture)
SPOT_MAX_AREA   = 80000
SPOT_MAX_ASPECT = 5.0   # Max height/width ratio (rejects vertical rail-joint lines)
SPOT_DARK_FRAC  = 0.45  # Region minimum pixel must be < this × rail mean
SPOT_VD_FRAC    = 0.03  # ≥ this fraction of region pixels must be below intensity 20
```

### Sparking / Corrosion Detection (Secondary — absolute darkness)

Catches uniformly-corroded surfaces where the whole rail is darkened so local contrast is weak.

```python
ABS_DARK_FRAC     = 0.30   # Pixel < this × rail mean = "absolutely dark"
ABS_DARK_MIN_ABS  = 20     # Absolute floor for the threshold
ABS_SPOT_MAX_ASPECT = 5.0
```

### Severity thresholds (Sparking / Corrosion)

```python
SPOT_SEVERE_AREA  = 5000   # px² → High severity
SPOT_MEDIUM_AREA  = 1500   # px² → Medium severity (below = Low)
```

### Surface Wear Detection

Uses a per-frame statistical threshold (`mean + N×std`) on a combined texture signal so it adapts to each image's lighting — prevents the wear detector flooding the frame on clean rail.

```python
WEAR_SIGMA_MULT  = 2.8    # Threshold = mean + N × std of texture signal
WEAR_MIN_AREA    = 300    # Minimum contour area in px²
WEAR_SEVERE_AREA = 1500   # px² → High severity
```

> **Note:** The original script used `ROUGHNESS_INTENSITY_THRESHOLD = 10`, which was below the signal mean and caused 82%+ of every frame to be classified as wear (the "blue flood"). Replaced with per-frame statistical thresholding.

### Crack Detection

```python
CANNY_T1           = 40    # Lower Canny hysteresis threshold
CANNY_T2           = 120   # Upper Canny hysteresis threshold
HOUGH_VOTES        = 50    # Minimum Hough accumulator votes
HOUGH_MIN_LEN      = 30    # Minimum line length in px
HOUGH_MAX_GAP      = 8     # Maximum gap within a line (px)
CRACK_MIN_LEN      = 30    # Minimum length to report as a crack
CRACK_SEVERE_LEN   = 80    # px → High severity
CRACK_ANGLE_MARGIN = 15    # Lines within this many degrees of horizontal or
                           # vertical are rejected (rail profile edges, not cracks)
```

> **Note:** The original script used `HOUGH_MIN_LINE_LENGTH = 1`, which returned every single edge pixel as a "crack". Raised to 30 and added the diagonal-angle filter.

### Bilateral / Gaussian Filter

```python
BIL_D, BIL_SC, BIL_SS = 9, 75, 75   # Bilateral filter: diameter, sigma colour, sigma space
GAUSS_K                = (3, 3)       # Gaussian blur kernel
LINE_DETECT_K          = 15           # Morphological line-detection kernel size
```

### Live Video

```python
LIVE_W, LIVE_H  = 640, 480   # Camera capture resolution
LIVE_FPS        = 15          # Target FPS (reduced for Raspberry Pi 5)
REC_DURATION    = 20          # Auto-stop recording after this many seconds
VIDEO_CODEC     = 'XVID'      # Output video codec
```

---

## 🧩 Detection Pipeline

`TRD_video.py` runs the following stages on every frame:

1. **Fixed ROI crop** — isolates the rail band using `ROI_START_PCT` / `ROI_WIDTH_PCT`
2. **LED glare removal** — bright pixels inpainted before any detection
3. **Primary dark-spot detection** — Gaussian local-contrast map, morphological cleanup, contour filter
4. **Secondary dark-spot detection** — absolute brightness relative to rail mean, catches uniformly-dark surfaces
5. **Surface wear detection** — high-pass texture + morphological line detection, per-frame statistical threshold
6. **Crack detection** — Canny edges + probabilistic Hough, diagonal-angle filter
7. **Classification** — each contour assigned type and Low / Medium / High severity
8. **Visualisation** — colour-coded overlays + legend panel with colour swatches

`TRD_demo.py` exposes all 32 intermediate steps of the pipeline on a single image for inspection and presentation.

---


## 📈 Performance

| Mode | Resolution | FPS — Raspberry Pi 5 | FPS — PC |
|------|------------|----------------------|----------|
| Batch (`TRD_training.py`) | Original (1920×1080) | N/A | ~3–5 img/s |
| Live (`TRD_video.py`) | 640×480 | 10–15 | 25–30 |

---


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/improvement`
3. Commit: `git commit -m 'Add improvement'`
4. Push: `git push origin feature/improvement`
5. Open a Pull Request

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ✉️ Contact

**Umer Shahid**
Lecturer, Department of Electrical Engineering
University of Engineering & Technology (UET), Lahore
📧 umer.shahid@uet.edu.pk
