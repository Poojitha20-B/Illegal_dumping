MODEL_NAME  = "rtdetr-l.pt"
IMGSZ       = 640           # video is 768x432 — 960 adds no benefit, hurts NMS
CONF_THRESH = 0.35          # raise slightly to kill ghost detections
IOU_THRESH  = 0.35          # lower = more aggressive merging of overlapping boxes

DEVICE = "mps"              # Mac Apple Silicon

KEEP_CLASSES = {
    "person",
    "bottle", "cup", "backpack", "handbag", "suitcase",
    "bag", "sports ball",
    "chair", "couch", "tv", "laptop",
    "box", "clock", "vase", "book",
    # For Layer 5 bin-context detection:
    "trash can", "waste container",  # won't match COCO but note for future
}

# Trash detection via background subtraction
TRASH_ENABLE          = True
TRASH_MIN_AREA        = 600       # minimum pixel area to consider as dropped object
TRASH_HISTORY         = 200       # background model history frames
TRASH_DIST2THRESHOLD  = 50.0      # sensitivity (lower = more sensitive)
TRASH_LABEL           = "trash"

PERSON_COLOR  = (0, 200, 0)
OBJECT_COLOR  = (0, 100, 255)
TRASH_COLOR   = (0, 0, 255)       # Red for detected trash
TEXT_COLOR    = (255, 255, 255)
BOX_THICKNESS = 2
FONT_SCALE    = 0.55