import base64
import textwrap
from pathlib import Path
import streamlit as st
import numpy as np
import cv2

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def normalize_to_bgra(img: np.ndarray) -> np.ndarray | None:
    """Ensure image is BGRA (has alpha)."""
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        a = np.full(b.shape, 255, dtype=np.uint8)
        return cv2.merge([b, g, r, a])
    raise ValueError(f"Unexpected image shape: {img.shape}")


def _estimate_bg_color(bgr: np.ndarray) -> np.ndarray:
    """Estimate background color from 4 corners."""
    h, w = bgr.shape[:2]
    pts = np.array([
        bgr[0, 0], bgr[0, w - 1], bgr[h - 1, 0], bgr[h - 1, w - 1]
    ], dtype=np.float32)
    return np.mean(pts, axis=0)  # BGR


def remove_solid_bg_to_alpha(bgr: np.ndarray, tol: int = 35, feather: int = 3) -> np.ndarray:
    """
    Remove (mostly) solid background by color distance to estimated bg color.
    tol: higher => remove more aggressively
    feather: blur alpha edge for smoother look
    """
    bg = _estimate_bg_color(bgr).reshape(1, 1, 3)
    diff = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)  # 0..~441
    # diff small => background => alpha 0
    alpha = np.clip((diff - tol) / (tol), 0.0, 1.0) * 255.0
    alpha = alpha.astype(np.uint8)

    if feather and feather > 0:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    b, g, r = cv2.split(bgr)
    return cv2.merge([b, g, r, alpha])


def overlay_bgra_on_bgr(dst_bgr: np.ndarray, src_bgra: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """Overlay BGRA onto BGR in-place."""
    if src_bgra is None:
        return

    H, W = dst_bgr.shape[:2]
    h, w = src_bgra.shape[:2]
    if x >= W or y >= H:
        return

    x2 = min(x + w, W)
    y2 = min(y + h, H)
    w2 = x2 - x
    h2 = y2 - y
    if w2 <= 0 or h2 <= 0:
        return

    roi = dst_bgr[y:y2, x:x2].astype(np.float32)
    src = src_bgra[:h2, :w2].astype(np.float32)

    alpha = (src[:, :, 3:4] / 255.0) * float(opacity)
    fg = src[:, :, :3]
    out = alpha * fg + (1.0 - alpha) * roi
    dst_bgr[y:y2, x:x2] = out.astype(np.uint8)


def overlay_bgra_on_bgra(dst_bgra: np.ndarray, src_bgra: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """Overlay BGRA onto BGRA in-place, preserving alpha."""
    if src_bgra is None:
        return

    H, W = dst_bgra.shape[:2]
    h, w = src_bgra.shape[:2]
    if x >= W or y >= H:
        return

    x2 = min(x + w, W)
    y2 = min(y + h, H)
    w2 = x2 - x
    h2 = y2 - y
    if w2 <= 0 or h2 <= 0:
        return

    dst_roi = dst_bgra[y:y2, x:x2].astype(np.float32)
    src = src_bgra[:h2, :w2].astype(np.float32)

    src_a = (src[:, :, 3:4] / 255.0) * float(opacity)
    dst_a = (dst_roi[:, :, 3:4] / 255.0)

    # Porter-Duff "over"
    out_a = src_a + dst_a * (1.0 - src_a)
    # Avoid divide-by-zero
    out_rgb = (src[:, :, :3] * src_a + dst_roi[:, :, :3] * dst_a * (1.0 - src_a)) / np.clip(out_a, 1e-6, 1.0)

    dst_bgra[y:y2, x:x2, :3] = out_rgb.astype(np.uint8)
    dst_bgra[y:y2, x:x2, 3:4] = (out_a * 255.0).astype(np.uint8)


def make_badge(icon_bgra: np.ndarray, badge_size: int = 56, pad: int = 8, bg_alpha: int = 160) -> np.ndarray:
    """
    Create a circular badge background (semi-transparent) and place icon centered.
    Returns BGRA badge.
    """
    badge = np.zeros((badge_size, badge_size, 4), dtype=np.uint8)

    center = (badge_size // 2, badge_size // 2)
    radius = badge_size // 2
    # dark translucent background
    cv2.circle(badge, center, radius, (0, 0, 0, int(bg_alpha)), -1)

    inner = badge_size - 2 * pad
    ic = cv2.resize(icon_bgra, (inner, inner), interpolation=cv2.INTER_AREA)
    overlay_bgra_on_bgra(badge, ic, pad, pad, opacity=1.0)
    return badge


def load_icons_as_badges(
        icon_dir: str = "assets/icons",
        badge_size: int = 56,
        pad: int = 8,
        bg_alpha: int = 160,
        try_remove_white_bg: bool = True,
) -> dict[str, np.ndarray | None]:
    """
    Load emotion icons and normalize them into same-style badges (BGRA).
    """
    d = Path(icon_dir)
    icons: dict[str, np.ndarray | None] = {}

    for e in EMOTIONS:
        p = d / f"{e}.png"
        if not p.exists():
            icons[e] = None
            continue

        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # BGRA or BGR
        if img is None:
            icons[e] = None
            continue

        # If no alpha channel, optionally remove white bg then add alpha
        if img.ndim == 3 and img.shape[2] == 3:
            bgr = img
            img = remove_solid_bg_to_alpha(bgr, tol=35, feather=3)
        else:
            img = normalize_to_bgra(img)

        badge = make_badge(img, badge_size=badge_size, pad=pad, bg_alpha=bg_alpha)
        icons[e] = badge

    return icons


def draw_emotion_row_on_frame(
        frame_bgr: np.ndarray,
        badges: dict[str, np.ndarray | None],
        selected: str | None,
        anchor: str = "top-right",  # "bottom-left" | "bottom-right" | "top-left" | "top-right"
        margin: int = 12,
        gap: int = 8,
        dim_opacity: float = 0.18,
) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    any_badge = next((b for b in badges.values() if b is not None), None)
    if any_badge is None:
        return frame_bgr

    size = any_badge.shape[0]
    n = len(EMOTIONS)
    row_w = n * size + (n - 1) * gap

    # compute start x,y by anchor
    if "right" in anchor:
        x = W - margin - row_w
    else:
        x = margin

    if "top" in anchor:
        y = margin
    else:
        y = H - margin - size

    # clamp (avoid negative if window too small)
    x = max(0, x)
    y = max(0, y)

    for e in EMOTIONS:
        b = badges.get(e)
        if b is None:
            x += size + gap
            continue
        op = 1.0 if (selected == e) else dim_opacity
        overlay_bgra_on_bgr(frame_bgr, b, x, y, opacity=op)
        x += size + gap

    return frame_bgr
