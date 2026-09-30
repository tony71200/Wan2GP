"""Simple Reels + Motion Control app on top of Wan2GP (shared.api)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REELS_MODEL = "i2v_2_2_Enhanced_Lightning_v2"
EXTEND_MODEL = "i2v_2_2_Enhanced_Lightning_v2_svi2pro"
MOTION_MODEL = "animate2_distilled"

# quality -> (portrait, square, landscape)
RESOLUTIONS = {
    "480p": ("480x832", "720x720", "832x480"),
    "720p": ("720x1280", "960x960", "1280x720"),
}

# ponytail: filename heuristic for Wan 2.2 high/low noise LoRA pairs; replace with an explicit pair picker if names stop matching
_HIGH = re.compile(r"(?<![a-z])high(?:noise)?(?![a-z])|_hn(?![a-z])|_h\.", re.I)
_LOW = re.compile(r"(?<![a-z])low(?:noise)?(?![a-z])|_ln(?![a-z])|_l\.", re.I)


def pick_resolution(w, h, quality="480p"):
    portrait, square, landscape = RESOLUTIONS[quality]
    ratio = w / h
    if ratio < 0.9:
        return portrait
    if ratio > 1.1:
        return landscape
    return square


def round_frames(n):
    """Largest 4k+1 frame count not above n (minimum 5)."""
    return max(5, (int(n) - 1) // 4 * 4 + 1)


def frames_for(seconds, fps=16):
    """Nearest 4k+1 frame count for a duration, clamped to [17, 81]."""
    return min(81, max(17, round((seconds * fps - 1) / 4) * 4 + 1))


def pair_loras(names, high=1.0, low=0.0):
    """Return (activated_loras, loras_multipliers) with per-phase multipliers."""
    mults = []
    for name in names:
        if _HIGH.search(name):
            mults.append(f"{high:g};0")
        elif _LOW.search(name):
            mults.append(f"0;{low:g}")
        else:
            mults.append(f"{high:g};{low:g}")
    return list(names), " ".join(mults)


def parse_ideas(text):
    ideas = []
    for line in str(text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.):])\s*", "", line).strip().strip('"')
        if line:
            ideas.append(line)
    return ideas[:3]
