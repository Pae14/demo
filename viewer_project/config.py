from PyQt6.QtGui import QColor
import os


# https://www.gelsight.com/wp-content/uploads/2023/01/GelSight_Datasheet_GSMini_12.20.22.pdf

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION & PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO12_PATH    = os.path.join(BASE_DIR, 'models', 'yolo12', 'best.pt')
YOLO26_PATH    = os.path.join(BASE_DIR, 'models', 'yolo26', 'best.pt')
RFDETR_PATH    = os.path.join(BASE_DIR, 'models', 'checkpoint_best_ema.pt')
DINOV3_PATH    = os.path.join(BASE_DIR, 'models', 'best.pt')
RESNET_PATH    = os.path.join(BASE_DIR, 'models', 'checkpoint_epoch_49.pth.tar')
CNN_CLS_PATH   = os.path.join(BASE_DIR, 'models', 'best_model.weights.h5')

SEGMENTATION_MODELS = {
    0: 'YOLO12n-seg',
    1: 'YOLO26n-seg',
    2: 'RF-DETR',
    3: 'DINOv3',
}

SEGMENTATION_TITLES = {
    0: 'YOLO12n-seg',
    1: 'YOLO26n-seg',
    2: 'RF-DETR',
    3: 'DINOv3',
}

IMG_DIR        = os.path.join(BASE_DIR, 'data', 'test2', 'images')
LBL_DIR        = os.path.join(BASE_DIR, 'data', 'test2', 'labels')

SAVE_DIR       = os.path.join(BASE_DIR, 'exports')
CONF_THRESHOLD = 0.25
# Set after validating the repair process against calibrated depth samples.
MAX_REPAIR_DEPTH_MM = None
# Exploratory 3D thresholds: P90 per class of dataset_defectos (Estudio_Global_Metricas_3D.xlsx).
# NON-REPAIRABLE (3D) if ANY metric exceeds its threshold. Not validated against real repairs.
REPAIR_3D_THRESHOLDS = {
    0: {'area_mm2': 45.93, 'depth_p95_mm': 0.0706, 'depth_volume_mm3': 1.2139},
    1: {'major_axis_length_mm': 5.7288, 'depth_p95_mm': 0.0694, 'depth_volume_mm3': 0.1212},
    2: {'area_mm2': 47.9318, 'depth_p95_mm': 0.0848, 'depth_volume_mm3': 2.0236},
    3: {'area_mm2': 11.2003, 'depth_p95_mm': 0.1777, 'depth_volume_mm3': 1.297},
}
REPAIR_MODEL_PATH = os.path.join(SAVE_DIR, 'repairability.joblib')
CAMERA_DEVICE  = os.environ.get('CAMERA_DEVICE', '/dev/video0')
GELSIGHT_CAM   = (
    int(CAMERA_DEVICE.removeprefix('/dev/video'))
    if CAMERA_DEVICE.startswith('/dev/video')
    else int(CAMERA_DEVICE)
)
STREAM_FPS     = 30
INFER_INTERVAL = 0.2  # Limit ROS/USB inference to 5 Hz by default.

# ─────────────────────────────────────────────────────────────────────────────
#  GELSIGHT MINI SPECS (Calibration)
# ─────────────────────────────────────────────────────────────────────────────
GS_FOV_W_MM = 18.6
GS_FOV_H_MM = 14.3

# ─────────────────────────────────────────────────────────────────────────────
#  CLASS / COLOR DEFINITIONS 
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: 'Scratch-Abrasion',
    2: 'Scratch-Friction',
    1: 'Crack',
    3: 'Material loss - Holes',
    4: 'No Defect',
}

CLASS_NAMES_SHORT = {
    0: 'Abrasion',
    2: 'Friction',
    1: 'Crack',
    3: 'Holes',
    4: 'No Defect',
}

RESNET_CLASS_NAMES = {
    3: 'Material loss - Holes',
    1: 'Crack',
    6: 'Crack',
    0: 'Scratch-Abrasion',
    2: 'Scratch-Friction',
    5: 'Scratch-Friction',
    4: 'No Defect',
}

CLASS_COLORS_RGB = {
    0: (210, 80,  70),
    1: (70,  165, 95),
    2: (80,  120, 205),
    3: (215, 145, 55),
    4: (155, 155, 155),
}

# ─────────────────────────────────────────────────────────────────────────────
#  PALETTE — warm peach / rose pastel
# ─────────────────────────────────────────────────────────────────────────────
P = {
    "bg":        "#fdf0ea",
    "bg2":       "#fff5f0",
    "border":    "#f0ddd5",
    "border2":   "#e5cfc5",
    "accent":    "#df5f4a",
    "accentL":   "#f08070",
    "accentSoft":"rgba(223,95,74,0.10)",
    "rose":      "#c5486e",
    "green":     "#4eaa72",
    "red":       "#d94f4f",
    "amber":     "#d98a42",
    "text":      "#291410",
    "text2":     "#66403a",
    "muted":     "#b09488",
    "divider":   "#f0ddd5",
}

C_BG      = QColor("#fdf0ea")
C_ACCENT  = QColor("#df5f4a")
C_BORDER  = QColor("#f0ddd5")
C_CARD    = QColor(255, 255, 255, 225)
C_MUTED   = QColor("#b09488")

ORB_COLS = [
    (238, 178, 155, 52),
    (208, 155, 195, 44),
    (175, 200, 238, 38),
    (195, 228, 198, 36),
    (248, 198, 155, 48),
]

PANEL_ACCENTS = [P["muted"], P["accent"], P["rose"], P["green"]]