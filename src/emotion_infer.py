from pathlib import Path
from typing import Optional, List, Tuple
import json
import numpy as np
import torch
import cv2
from datetime import datetime

# CONFIG
DEFAULT_LABELS = ['angry', 'disgust', 'fear',
                  'happy', 'neutral', 'sad', 'surprise']

IMAGE_SIZE = 48
USE_GRAYSCALE = True
# IMPORTANT: Must match training normalization!
# Training used: transforms.Normalize((0.5,), (0.5,)) which is mean_std
NORM_MODE = "mean_std"  # "0_1" or "minus1_1" or "mean_std"
MEAN = (0.5,)  # Used when NORM_MODE="mean_std"
STD = (0.5,)


def load_labels(label_path: Optional[str]) -> List[str]:
    if not label_path:
        return DEFAULT_LABELS

    p = Path(label_path)
    if not p.exists():
        return DEFAULT_LABELS

    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj[i] for i in range(len(obj))]

    return DEFAULT_LABELS


def _save_debug_face(normalized_array: np.ndarray, prediction: str, confidence: float):
    """
    Save preprocessed face image for debugging.
    
    Args:
        normalized_array: Normalized numpy array [C, H, W]
        prediction: Predicted emotion label
        confidence: Prediction confidence
    """
    try:
        # Create debug directory
        debug_dir = Path("debug_faces")
        debug_dir.mkdir(exist_ok=True)
        
        # Denormalize for visualization
        if NORM_MODE == "mean_std":
            # Reverse: (x - mean) / std  =>  x = (value * std) + mean
            img = normalized_array.copy()
            mean = np.array(MEAN, dtype=np.float32)[:, None, None]
            std = np.array(STD, dtype=np.float32)[:, None, None]
            img = (img * (std + 1e-6)) + mean
            img = (img * 255.0).clip(0, 255)
        elif NORM_MODE == "minus1_1":
            # Reverse: (x / 127.5) - 1  =>  x = (value + 1) * 127.5
            img = ((normalized_array + 1.0) * 127.5).clip(0, 255)
        else:  # 0_1 or default
            # Reverse: x / 255  =>  x = value * 255
            img = (normalized_array * 255.0).clip(0, 255)
        
        # Convert to uint8
        if img.shape[0] == 1:  # Grayscale [1, H, W]
            img = img[0].astype(np.uint8)
        else:  # RGB [3, H, W]
            img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]  # HHMMSSmmm
        filename = f"{timestamp}_{prediction}_{confidence:.3f}.png"
        filepath = debug_dir / filename
        
        # Save image
        cv2.imwrite(str(filepath), img)
        print(f"[DEBUG] Saved face: {filename}")
        
    except Exception as e:
        print(f"[DEBUG] Failed to save face: {e}")


def preprocess_face_roi(bgr: np.ndarray, debug_save: bool = False, 
                       prediction: str = "", confidence: float = 0.0) -> torch.Tensor:
    """
    Return tensor shape [1, C, H, W] float32
    
    Args:
        bgr: Input BGR image
        debug_save: If True, save the preprocessed image for inspection
        prediction: Predicted emotion label (for debug filename)
        confidence: Prediction confidence (for debug filename)
    """

    if USE_GRAYSCALE:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        x = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE),
                       interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32)
        x = x[None, :, :]  # [1, H, W]
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE),
                       interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32)
        x = np.transpose(x, (2, 0, 1))  # [3, H, W]

    if NORM_MODE == "0_1":
        x = x / 255.0
    elif NORM_MODE == "minus1_1":
        x = (x / 127.5) - 1.0
    elif NORM_MODE == "mean_std":
        x = x / 255.0
        mean = np.array(MEAN, dtype=np.float32)[:, None, None]
        std = np.array(STD, dtype=np.float32)[:, None, None]
        x = (x - mean) / (std + 1e-6)
    else:
        x = x / 255.0

    t = torch.from_numpy(x).unsqueeze(0)  # [1, C ,H, W]
    
    # Debug: Save preprocessed face image
    if debug_save and prediction:
        _save_debug_face(x, prediction, confidence)
    
    return t


def crop_with_margin(bgr: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.25) -> Optional[np.ndarray]:
    """
    bbox: (x1,y1,x2,y2)
    margin: add % around bbox
    """

    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return None

    mx = int(bw * margin)
    my = int(bh * margin)

    xx1 = max(0, x1 - mx)
    yy1 = max(0, y1 - my)
    xx2 = min(w, x2 + mx)
    yy2 = min(h, y2 + my)

    if xx2 <= xx1 or yy2 <= yy1:
        return None

    roi = bgr[yy1:yy2, xx1:xx2]
    return roi
