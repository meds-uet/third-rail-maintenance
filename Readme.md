# 🚇 Third Rail Defect Detection System (TRDDS)


A comprehensive system for detecting defects in metro third rails using both **image processing** and **real-time webcam analysis**. The system identifies:
- Corrosion/Dark spots
- Surface wear patterns
- Cracks/Linear defects
- LED glare interference

## 🌟 Key Features

### 📷 Dual Operation Modes
1. **Batch Image Processing** (`TRD_training.py`)
   - Process folders of rail images
   - Generate detailed defect reports
   - Create before/after comparisons

2. **Live Webcam Analysis** (`TRD_video.py`)
   - Real-time defect detection
   - Interactive controls:
     - 📹 Record 20-second videos
     - 📸 Capture still images
     - ↔️ Side-by-side comparison view
   - Optimized for Raspberry Pi 5

### 🔍 Advanced Detection Capabilities
- **Multi-stage defect classification** by type and severity
- **LED glare removal** for clearer analysis
- **Adaptive thresholds** for varying lighting conditions
- **Configurable sensitivity** through easy-to-edit macros

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- OpenCV 4.5+
- Raspberry Pi 5 (for live mode) or any modern PC

```bash
# Clone repository
git clone https://github.com/meds-uet/third-rail-maintenance.git
cd third-rail-defect-detection

# Install dependencies
pip3 install opencv-python numpy
```

## 🚀 Usage

### Batch Processing Mode
1. Place your rail images in `/images` folder
2. Run:
   ```bash
   python3 TRD_training.py
   ```
3. Results saved in:
   - `/output/processed` - Annotated images
   - `/output/comparisons` - Before/after comparisons

### Live Webcam Mode
1. Connect your webcam
2. Run:
   ```bash
   python3 TRD_video.py
   ```
3. Use the interactive controls:
   - **RECORD** button: Capture 20-second video
   - **CAPTURE** button: Save still image
   - **EXIT** button: Quit application
   - *Keyboard shortcuts*: R=Record, C=Capture, Q=Quit

4. Outputs saved in:
   - `/output/live_captures` - Still images
   - `/output/live_recordings` - Video clips

## 📂 Folder Structure
```
project-root/
├── images/                  # Input images for batch processing
├── output/
│   ├── processed/           # Processed images with defects marked
│   ├── comparisons/         # Side-by-side before/after comparisons
│   ├── live_captures/       # Webcam still images
│   └── live_recordings/     # Recorded video clips
├── assets/                  # Documentation assets
├── TRD_training.py          # Batch image processing 
└── TRD_video.py             # Live webcam processing script
```

## ⚙️ Configuration Guide
Key parameters in the code (all adjustable at top of files):

```python
# Detection Sensitivity
CORROSION_THRESHOLD = 50      # Lower = more sensitive to dark spots
CRACK_THRESHOLD = 5           # Lower = detects fainter cracks
MIN_DEFECT_AREA = 2           # Minimum defect size (pixels)

# Visualization
DEFECT_LABEL_FONT_SCALE = 0.6 # Text size for defect labels
DEFECT_COLORS = {             # Customize defect highlight colors
    'Corrosion': (0, 0, 255), # Red
    'Wear': (255, 0, 0),      # Blue
    'Crack': (0, 255, 0)      # Green
}

# Live Mode Settings
RECORDING_DURATION = 20       # Seconds per video clip
LIVE_FPS = 15                 # Framerate (lower for RPi optimization)
```

## 📊 Sample Outputs

### Batch Processing Example
| Original Image | Processed Result |
|----------------|------------------|
| ![Original](assets/sample_original.jpg) | ![Processed](assets/sample_processed.jpg) |


## 🧩 Technical Approach
1. **ROI Extraction**: Focuses analysis on the critical rail surface area
2. **LED Glare Removal**: Uses adaptive thresholding and inpainting
3. **Multi-Method Detection**:
   - Corrosion: Adaptive thresholding + contour analysis
   - Cracks: Canny edge detection + Hough line transform
   - Wear: Morphological operations + texture analysis
4. **Classification**: Defects categorized by type and severity

## 📈 Performance Metrics
| Mode          | Resolution | FPS (RPi 5) | FPS (PC) |
|---------------|-----------|------------|----------|
| Batch         | Original  | N/A        | ~3-5/img |
| Live          | 640x480   | 10-15      | 25-30    |

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📜 License
MIT License - See [LICENSE](LICENSE) for details

## ✉️ Contact
**Umer Shahid**  
Department of Electrical Engineering  
University of Engineering & Technology (UET), Lahore  
📧 umershahid@uet.edu.pk