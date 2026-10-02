"""Simple Reels + Motion Control app on top of Wan2GP (shared.api)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REELS_MODEL = "i2v_2_2_Enhanced_Lightning_v2"
EXTEND_MODEL = "i2v_2_2_Enhanced_Lightning_v2_svi2pro"
MOTION_MODEL = "animate2_distilled"
FPS = 16  # Wan 2.2 i2v output rate; Motion Control resamples the driving video to it
ANIMATE_MODEL = "animate"  # Wan 2.2 Animate 14B
FUSIONX_LORA = "Wan2.1_I2V_14B_FusionX_LoRA.safetensors"  # loras/wan_i2v, 10-step accelerator
MOTION_MODELS = {"Animate 2 distilled": MOTION_MODEL, "Wan 2.2 Animate 14B": ANIMATE_MODEL}
MOTION_WINDOWS = {MOTION_MODEL: (65, 1), ANIMATE_MODEL: (81, 5)}  # (window size, overlap)

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


def motion_frames(src_fps, src_frames, limit=None):
    """Frames at FPS covering the whole driving video, or its first `limit` seconds."""
    seconds = src_frames / src_fps
    if limit:
        seconds = min(limit, seconds)
    return max(5, round((seconds * FPS - 1) / 4) * 4 + 1)


def sliding_windows(frames, size, overlap):
    """Frames per sliding window when `frames` are generated in windows of `size` sharing `overlap` frames."""
    count = 1 + max(0, -(-(frames - size) // (size - overlap)))
    return [size] * (count - 1) + [frames - (count - 1) * (size - overlap)]


def frames_for(seconds, fps=16):
    """Nearest 4k+1 frame count for a duration, clamped to [17, 81]."""
    return min(81, max(17, round((seconds * fps - 1) / 4) * 4 + 1))


def reel_plan(seconds):
    """(model_type, whole-second duration of each window) for a reel of `seconds`."""
    seconds = int(seconds)
    if seconds <= 5:
        return REELS_MODEL, [seconds]
    count = -(-seconds // 5)  # windows of at most 5 s (81 frames)
    base, extra = divmod(seconds, count)
    return EXTEND_MODEL, [base + 1] * extra + [base] * (count - extra)


def reel_windows(seconds):
    """(model_type, frames per window) used for the time estimate."""
    model, durations = reel_plan(seconds)
    if model == REELS_MODEL:
        return model, [frames_for(seconds)]
    return model, [d * FPS + 1 for d in durations]


# "(at 3 seconds: ...)", "(0-2 seconds: ...)", "(from 0 to 2 seconds: ...)", "0-2 giây: ..."
_TIMED = re.compile(
    r"^\(?\s*(?:at|from|từ)?\s*(\d+(?:\.\d+)?)\s*(?:(?:-|–|to|đến)\s*(\d+(?:\.\d+)?)\s*)?"
    r"(?:seconds?|giây|s)\s*:\s*(.*?)\)?$",
    re.I,
)


def split_timeline(prompt, durations):
    """One `[/duration=Ns]` paragraph per window; timed lines go to every window they overlap, re-timed from 0."""
    lines = [line.strip() for line in str(prompt or "").splitlines() if line.strip()]
    common, timed = [], []
    for line in lines:
        m = _TIMED.match(line)
        if m:
            timed.append((float(m[1]), float(m[2]) if m[2] else None, m[3].strip()))
        else:
            common.append(line)
    paragraphs, t0 = [], 0
    for d in durations:
        t1 = t0 + d
        body = list(common)
        hits = 0
        for start, end, text in timed:
            if start < t1 and (end if end is not None else start + 1) > t0:
                hits += 1
                if end is None:
                    body.append(f"(at {start - t0:g} seconds: {text})")
                else:
                    body.append(f"({max(start, t0) - t0:g}-{min(end, t1) - t0:g} seconds: {text})")
        earlier = [text for start, _, text in timed if start < t0]
        if timed and not hits and earlier:
            body.append(f"(at 0 seconds: {earlier[-1]})")
        paragraphs.append(f"[/duration={d}s] " + "\n".join(body))
        t0 = t1
    return "\n\n".join(paragraphs)


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


DETAIL_LORA = "detailz-wan.safetensors"  # loras/wan; Wan 2.1 detail enhancer, run on the low-noise phase only
DETAIL_MULT = "0;0.6"


def _lora_settings(loras, high, low, smooth, detail=False):
    names, mults = pair_loras(list(loras or []), high, low)
    if detail and DETAIL_LORA not in names:
        names.append(DETAIL_LORA)
        mults = f"{mults} {DETAIL_MULT}".strip()
    return {
        "activated_loras": names,
        "loras_multipliers": mults,
        "temporal_upsampling": "rife*2" if smooth else "",
    }


def reels_settings(defaults, *, image, prompt, resolution, seconds, seed, loras=(), high=1.0, low=0.0, smooth=True, detail=False):
    """Up to 5 s: one Lightning window. Longer: one SVI 2 Pro job, every window anchored on the original image."""
    model, durations = reel_plan(seconds)
    settings = {
        **defaults,
        "model_type": model,
        "prompt": prompt,
        "image_prompt_type": "S",
        "image_start": image,
        "resolution": resolution,
        "video_length": frames_for(seconds),
        "seed": int(seed),
        **_lora_settings(loras, high, low, smooth, detail),
    }
    if model == EXTEND_MODEL:
        settings.update(
            prompt=split_timeline(prompt, durations),
            multi_prompts_gen_type="PW",
            video_prompt_type="KI",
            image_refs=[image],
            video_length=int(seconds) * FPS + 1,  # Wan2GP replaces it with the sum of the [/duration] markers
        )
    return settings


def extend_settings(defaults, *, video, anchor, prompt, resolution, seconds, seed, loras=(), high=1.0, low=0.0, smooth=True, detail=False):
    """Continue `video` with SVI 2 Pro, anchoring every window on the original image."""
    return {
        **defaults,
        "model_type": EXTEND_MODEL,
        "prompt": prompt,
        "image_prompt_type": "V",
        "image_start": None,
        "video_source": video,
        "video_prompt_type": "KI",
        "image_refs": [anchor],
        "resolution": resolution,
        # 4 = SVI 2 Pro window overlap; asking for the full window spills a few frames into a second, wasted window
        "video_length": frames_for(seconds) - 4,
        "seed": int(seed),
        **_lora_settings(loras, high, low, smooth, detail),
    }


MAX_IMAGES = 6


def _one_line(*parts):
    """Join non-empty parts with '. ' on a single line (PW windows must not contain blank lines)."""
    cleaned = [" ".join(str(p).split()).rstrip(".") for p in parts if p and str(p).strip()]
    return ". ".join(c for c in cleaned if c)


def multi_settings(defaults, *, images, durations, prompts, common, resolution, seed, loras=(), high=1.0, low=0.0, smooth=True, detail=False):
    """One Lightning job through every image: window i starts at image i, lasts durations[i]
    and ends on image i+1; the last window has no end image."""
    paragraphs = [f"[/duration={int(d)}s] {_one_line(common, p)}" for d, p in zip(durations, prompts)]
    return {
        **defaults,
        "model_type": REELS_MODEL,
        "prompt": "\n\n".join(paragraphs),
        "multi_prompts_gen_type": "PW",
        "image_prompt_type": "SE" if len(images) > 1 else "S",
        "image_start": images[0],
        "image_end": list(images[1:]) or None,
        "resolution": resolution,
        "video_length": sum(int(d) for d in durations) * FPS + 1,
        "seed": int(seed),
        **_lora_settings(loras, high, low, smooth, detail),
    }


def aspect_mismatch(sizes, tolerance=0.1):
    """1-based positions of images whose aspect ratio differs from the first one by more than `tolerance`."""
    if not sizes or not sizes[0]:
        return []
    ref = sizes[0][0] / sizes[0][1]
    return [i for i, size in enumerate(sizes[1:], 2) if size and abs(size[0] / size[1] / ref - 1) > tolerance]


# "<image> :0–3s:" or "<image>: 3s:" heading each Story block
_STORY_HEAD = re.compile(
    r"^(?P<name>.+?\.(?:png|jpe?g|webp))\s*:\s*(?:(?P<start>\d+(?:\.\d+)?)\s*[–-]\s*)?(?P<end>\d+(?:\.\d+)?)\s*s?\s*:?$",
    re.I,
)


def parse_story(text):
    """[(image name, whole seconds, prompt on one line)] from a Story file, in file order."""
    story = []
    for block in re.split(r"\n\s*\n", str(text).lstrip("﻿").strip()):
        head, *body = block.strip().splitlines()
        m = _STORY_HEAD.match(head.strip())
        if not m:
            raise ValueError(f"Story: không đọc được dòng đầu khối: {head!r}")
        seconds = float(m["end"]) - float(m["start"] or 0)
        if seconds != int(seconds) or not 1 <= seconds <= 5:
            raise ValueError(f"Story: {m['name'].strip()}: thời lượng {seconds:g}s phải là số giây nguyên từ 1 đến 5")
        story.append((m["name"].strip(), int(seconds), _one_line(" ".join(body))))
    return story


# (steps, seconds per step, reference frames, reference pixels), measured 2026-09-30 on the RTX 5070 Ti Laptop 12 GB
SPEED = {
    REELS_MODEL: (4, 97, 81, 720 * 720),
    EXTEND_MODEL: (8, 106, 81, 720 * 720),
    MOTION_MODEL: (10, 94, 65, 480 * 848),
    ANIMATE_MODEL: (10, 94, 65, 480 * 848),  # not measured yet: borrowed from Animate 2, calibration corrects it
}
LOAD_SECONDS = 120  # model load + VAE decode
MODEL_LABELS = {REELS_MODEL: "Lightning v2", EXTEND_MODEL: "SVI 2 Pro", MOTION_MODEL: "Animate 2", ANIMATE_MODEL: "Animate 14B"}
TIMINGS = ROOT / "simple_app_timings.json"
HARNESS_LOG = ROOT / "logs" / "simple_app_harness.log"


def log_harness(entry, path=HARNESS_LOG):
    """Append one block to the harness log: a '=== time command' line, then 'key: value' lines."""
    import time

    entry = dict(entry)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {entry.pop('command', '')}"]
    lines += [f"{k}: {v}" for k, v in entry.items()]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def estimate_seconds(model, windows, resolution, calib=None):
    """Expected wall time of a job; `windows` lists the frame count of each sliding window."""
    steps, per_step, ref_frames, ref_pixels = SPEED[model]
    w, h = (int(v) for v in resolution.split("x"))
    # ponytail: linear in frames x pixels (attention grows faster); the calibration ratio absorbs the error for usual sizes
    work = sum(windows) * w * h / (ref_frames * ref_pixels)
    return (LOAD_SECONDS + steps * per_step * work) * (calib or {}).get(model, 1.0)


def fmt_minutes(seconds):
    minutes = max(1, round(seconds / 60))
    return f"{minutes} phút" if minutes < 60 else f"{minutes // 60} giờ {minutes % 60} phút"


def eta_text(model, windows, resolution, calib=None):
    steps = SPEED[model][0]
    eta = fmt_minutes(estimate_seconds(model, windows, resolution, calib))
    return f"Ước tính {eta} · {MODEL_LABELS[model]} · {len(windows)} cửa sổ × {steps} bước"


def load_calib(path=TIMINGS):
    """Per-model ratio actual / estimated time from previous runs; {} when missing or unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def save_calib(model, actual, raw_estimate, path=TIMINGS):
    calib = load_calib(path)
    calib[model] = round(actual / raw_estimate, 3)
    Path(path).write_text(json.dumps(calib, indent=1))
    return calib


# The model's own default prompt describes a demo character (a cat in a school uniform), so never fall back to it.
MOTION_DEFAULT_PROMPT = (
    "Character appearance: the same people as in the reference image, with the same faces, hairstyles and outfits. "
    "Background: the same place as in the reference image."
)


def motion_settings(defaults, *, image, video, description, resolution, frames, seed, model=MOTION_MODEL, smooth=False, audio=False, outfit=False):
    """Character image animated by the driving video; the image is the reference for every window."""
    settings = {
        **defaults,
        "model_type": model,
        "prompt": (description or "").strip() or MOTION_DEFAULT_PROMPT,
        "image_start": None,
        "image_refs": [image],
        "video_guide": video,
        "resolution": resolution,
        "video_length": int(frames),
        "force_fps": str(FPS),  # resample the driving video: half the frames of 30 fps, same duration
        "seed": int(seed),
        "temporal_upsampling": "rife*2" if smooth else "",
        # Wan2GP muxes the driving video's audio, trimmed to the generated length
        "postprocess_audio": "control" if audio else "",
    }
    if model == ANIMATE_MODEL:
        settings.update(
            image_prompt_type="",
            video_prompt_type="PVBKI",  # Animate Person in Reference Image using Motion of Whole Control Video
            activated_loras=[FUSIONX_LORA],
            loras_multipliers="1",
            # profiles/wan_i2v/Image2Video FusioniX - 10 Steps.json
            num_inference_steps=10,
            guidance_phases=1,
            guidance_scale=1,
            sliding_window_size=MOTION_WINDOWS[ANIMATE_MODEL][0],
        )
    elif outfit:
        # no character reference: later windows continue from the previous window, so outfit changes can carry over
        settings.update(image_prompt_type="S", image_start=image, video_prompt_type="UV", image_refs=None)
    else:
        settings.update(image_prompt_type="T", video_prompt_type="UVI")
    return settings


def ideas_instructions(seconds=5):
    return (
        "You are a creative director for short vertical social videos. Look at the image and propose "
        f"exactly 3 different short video ideas that start from this exact image. Each idea must work as one "
        f"continuous {seconds} second shot in the same place: no cuts, no change of location, lighting or time of day. "
        "Keep the same person, face, hairstyle and setting, and prefer small natural actions over big ones. "
        "The person keeps wearing exactly what they wear in the image. "
        "Write one idea per line, one sentence each, with no numbering and no extra text."
    )


# The model's own enhancer instructions ask for shot changes every second, which turns the person
# into someone else. These keep one continuous shot and restate the subject on every line.
def detail_instructions(seconds=5):
    return (
        "You are an AI vision model writing a prompt for an image-to-video model. Analyze the provided image and the user's goal.\n"
        "Describe ONE continuous shot that starts exactly from the image: no cuts, no scene change, no change of location, lighting, weather or time of day.\n"
        "The person must stay the same person for the whole video: same face, hairstyle, hair color, body, skin tone, outfit and accessories as in the image. Never describe a different look.\n"
        "Keep the motion natural and modest so the face stays recognizable. The camera is static or moves slowly.\n"
        "The camera keeps the same distance and framing as the image: no zoom, no push-in, no close-up on the face; the person may walk closer by themselves.\n"
        "Output rules:\n"
        f"Write one line per second, from 0 to {seconds - 1} seconds.\n"
        "Each line must begin with \"(at X seconds:\" where X is the second number, followed by a dense, comma-separated visual description, all inside a single set of parentheses.\n"
        "Every line repeats the same subject and setting description taken from the image and changes only the action and the camera movement.\n"
        "Do not include commentary, explanations, or meta-text."
    )


def enhance(session, model_type, text, image, instructions=None, progress=None):
    """Run Wan2GP's local prompt enhancer on (text, image) and return the generated text."""
    wgp = sys.modules["wgp"]
    from shared.utils.model_unload import wait_for_model_unload
    from shared.utils.process_locks import acquire_GPU_ressources, release_GPU_ressources

    # ponytail: session._state is private to WanGPSession; switch to the plugin-provided state if the API changes
    state = session._state
    model_def = wgp.get_model_def(model_type) if model_type else None
    wait_for_model_unload()
    acquire_GPU_ressources(state, "prompt_enhancer", "Prompt Enhancer")
    try:
        wgp.ensure_prompt_enhancer_loaded(progress=progress)
        out = wgp.process_prompt_enhancer(
            model_type, model_def, "TI", [text], image, None, False, False, -1,
            prompt_enhancer_instructions=instructions,
        )
    finally:
        wgp.unload_prompt_enhancer_runtime()
        if wgp.enhancer_offloadobj is not None:
            wgp.enhancer_offloadobj.unload_all()
        release_GPU_ressources(state, "prompt_enhancer")
    return str(out[0]) if out else ""


def _image_size(path):
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _video_info(path):
    """Return (fps, frame_count) of a video file."""
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


# Dark creative-tool palette (spec v3 §1): neutral surfaces, one orange accent for the primary action, state and focus.
COLORS = {
    "bg": "#0B0B0C", "surface": "#141416", "panel": "#1C1C1F", "border": "#27272A", "border_strong": "#3F3F46",
    "text": "#F4F4F5", "text2": "#A1A1AA", "accent": "#F97316", "accent_hover": "#FB923C", "on_accent": "#0B0B0C",
    "success": "#4ADE80", "warn": "#FBBF24", "error": "#F87171",
}
THEME_VARS = {
    "body_background_fill": COLORS["bg"],
    "body_text_color": COLORS["text"],
    "body_text_color_subdued": COLORS["text2"],
    "background_fill_primary": COLORS["surface"],
    "background_fill_secondary": COLORS["panel"],
    "border_color_primary": COLORS["border"],
    "border_color_accent": COLORS["accent"],
    "block_background_fill": COLORS["surface"],
    "block_border_color": COLORS["border"],
    "block_border_width": "1px",
    "block_shadow": "none",
    "block_label_background_fill": COLORS["surface"],
    "block_label_text_color": COLORS["text2"],
    "block_title_text_color": COLORS["text2"],
    "input_background_fill": COLORS["panel"],
    "input_border_color": COLORS["border"],
    "input_border_color_focus": COLORS["accent"],
    "input_shadow": "none",
    "input_shadow_focus": f"0 0 0 2px {COLORS['accent']}59",
    "button_primary_background_fill": COLORS["accent"],
    "button_primary_background_fill_hover": COLORS["accent_hover"],
    "button_primary_text_color": COLORS["on_accent"],
    "button_primary_border_color": COLORS["accent"],
    "button_primary_shadow": "none",
    "button_primary_shadow_hover": "none",
    "button_secondary_background_fill": COLORS["panel"],
    "button_secondary_background_fill_hover": COLORS["border"],
    "button_secondary_text_color": COLORS["text"],
    "button_secondary_border_color": COLORS["border_strong"],
    "button_secondary_shadow": "none",
    "button_secondary_shadow_hover": "none",
    "checkbox_background_color": COLORS["panel"],
    "checkbox_background_color_selected": COLORS["accent"],
    "checkbox_border_color": COLORS["border_strong"],
    "checkbox_label_shadow": "none",
    "slider_color": COLORS["accent"],
    "color_accent": COLORS["accent"],
    "color_accent_soft": COLORS["panel"],
    "link_text_color": COLORS["accent"],
    "panel_background_fill": COLORS["surface"],
    "panel_border_color": COLORS["border"],
    "shadow_drop": "none",
    "shadow_drop_lg": "none",
}
# The page is always dark, whatever the OS theme.
FORCE_DARK = "() => document.body.classList.add('dark')"

CSS = f"""
.gradio-container {{ max-width: 1280px !important; margin: 0 auto; }}
#header {{ align-items: center; border-bottom: 1px solid {COLORS['border']}; padding: 8px 0 16px; }}
#header h1 {{ font-size: 24px; font-weight: 600; letter-spacing: -0.02em; margin: 0; }}
#header p {{ color: {COLORS['text2']}; margin: 2px 0 0; }}
.eta {{ font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; color: {COLORS['text2']}; font-size: 14px; }}
.note {{ color: {COLORS['warn']}; font-size: 14px; }}
#guide {{ display: none; position: fixed; inset: 5vh 10vw; z-index: 1000; background: {COLORS['surface']};
          border: 1px solid {COLORS['border_strong']}; border-radius: 8px; padding: 24px; overflow: auto; }}
"""


def make_theme():
    """Gradio theme carrying COLORS for both light and dark variables."""
    import inspect

    import gradio as gr

    theme = gr.themes.Base(
        primary_hue="orange", neutral_hue="zinc", radius_size=gr.themes.sizes.radius_sm,
        font=[gr.themes.GoogleFont("Inter Tight"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )
    settable = set(inspect.signature(theme.set).parameters)
    return theme.set(**THEME_VARS, **{k + "_dark": v for k, v in THEME_VARS.items() if k + "_dark" in settable})


GUIDE = """
## Hướng dẫn sử dụng

### Reels: một ảnh → video 2–15 giây
- **Từ 5 giây trở xuống**: Wan 2.2 i2v *Enhanced Lightning v2*, 4 bước, một cửa sổ.
- **Trên 5 giây**: *SVI 2 Pro* (cùng model, thêm LoRA SVI), 8 bước mỗi cửa sổ, chia thành các cửa sổ tối đa 5 giây trong **một job**. Mọi cửa sổ neo về ảnh gốc nên nhân vật không trôi.
- Prompt có mốc giây như `(at 3 seconds: …)` hoặc `(0-2 seconds: …)` được tự tách cho từng cửa sổ. Dòng không có mốc (mô tả nhân vật) được giữ ở mọi cửa sổ.
- **3 ý tưởng / Viết chi tiết**: Qwen3.5-4B (có thị giác) đọc ảnh rồi viết prompt theo từng giây của thời lượng đang chọn.
- **Kéo dài video**: nối thêm một đoạn bằng SVI 2 Pro, neo về ảnh gốc.
- **Giữ da & tóc**: thêm LoRA chi tiết *Detailz* ở Pha cuối (cường độ 0.6) để da và tóc không bị bệt hay biến dạng. Ảnh cận mặt: bật *Giữ da & tóc*, nên chọn 720p. "Viết chi tiết" giữ máy quay ở khoảng cách cũ, không zoom sát mặt.

### Reels Multi Image: tối đa 6 ảnh của cùng một nhân vật
Mỗi ảnh là một mốc. Đoạn của ảnh *i* bắt đầu ở ảnh *i*, dài đúng số giây ghi trên ảnh đó và kết thúc ở ảnh *i+1*. Đoạn của ảnh cuối chạy tự do. Mô tả nhân vật chung được ghép vào đầu prompt của mọi đoạn. Độ phân giải theo ảnh 1.

### Motion Control
Nhân vật trong ảnh làm theo chuyển động của video mẫu. Video ra dài **đúng bằng video mẫu**, trừ khi tick *Giới hạn thời lượng*. Video mẫu được đưa về 16 fps để chạy nhanh gấp đôi; bật *Mượt x2* để ra 32 fps.
- **Animate 2 distilled**: 10 bước, cửa sổ 65 frame.
- **Wan 2.2 Animate 14B**: thêm LoRA FusionX, 10 bước, cửa sổ 81 frame.
- **Giữ âm thanh video mẫu**: video ra mang âm thanh của video mẫu, cắt đúng độ dài.
- **Trang phục theo video mẫu** (chỉ Animate 2): ảnh chỉ làm khung đầu, các đoạn sau nối tiếp nhau nên trang phục đổi được theo video mẫu (ví dụ cởi áo). Đánh đổi: mặt có thể trôi nhiều hơn.

### Video trung gian
Video dài được sinh từng cửa sổ. App chỉ giữ video cuối: mỗi cửa sổ mới thay file của cửa sổ trước, và báo số video trung gian đã xóa khi xong.

### Pha đầu và Pha cuối
Wan 2.2 khử nhiễu bằng **hai model nối tiếp**:
- **Pha đầu (high noise)**: các bước đầu, quyết định **bố cục và chuyển động lớn**.
- **Pha cuối (low noise)**: các bước sau, quyết định **chi tiết: mặt, da, tóc, vải**.

Hai thanh trượt là cường độ LoRA trong từng pha. Để **Pha cuối = 0** thì LoRA chỉ tác động lên chuyển động, khuôn mặt giữ theo ảnh gốc. File LoRA có chữ `high` chỉ chạy ở pha đầu, file có chữ `low` chỉ chạy ở pha cuối.

### Thời gian ước tính
Tính từ số đo trên RTX 5070 Ti Laptop 12 GB. Sau mỗi lần chạy, app ghi thời gian thật vào `simple_app_timings.json` để lần sau ước tính sát hơn.

### Upscale ×2
Phóng to video đang hiển thị gấp đôi bằng Lanczos: nhanh, không thêm chi tiết.

### LoRA tải thêm
Tải **cả cặp** high/low vào `loras/wan`, chọn cả hai file trong ô LoRA, để Pha cuối = 0.
- [lkzd7/WAN2.2_LoraSet_NSFW](https://huggingface.co/lkzd7/WAN2.2_LoraSet_NSFW): 48 file, phần lớn là cặp high/low.
- [Lythiga/WAN2.2_I2V_Lora](https://huggingface.co/Lythiga/WAN2.2_I2V_Lora)
- [rolechar/rc_wan2.2_i2v_loraset_nsfw_private](https://huggingface.co/rolechar/rc_wan2.2_i2v_loraset_nsfw_private): riêng tư, cần `HF_TOKEN`.
- [lopi999/Wan2.2-I2V_General-NSFW-LoRA](https://huggingface.co/lopi999/Wan2.2-I2V_General-NSFW-LoRA): cặp `NSFW-22-H-e8` / `NSFW-22-L-e8`.

### Nguồn model
- [DeepBeepMeep/Wan2.2](https://huggingface.co/DeepBeepMeep/Wan2.2): Enhanced Lightning v2, LoRA SVI 2 Pro, Animate 2, Wan 2.2 Animate 14B
- [DeepBeepMeep/Wan2.1](https://huggingface.co/DeepBeepMeep/Wan2.1): RIFE, LoRA FusionX
- [Enhanced Lightning v2 (civitai)](https://civitai.com/models/2053259)
- [Wan2GP](https://github.com/deepbeepmeep/Wan2GP)
"""


def build_ui(session):
    import contextvars
    import time

    import gradio as gr
    from shared.api import create_gradio_progress_callbacks

    active = {"job": None}

    def run(submit, progress):
        """submit(callbacks) -> job. Returns the last generated file, or None when cancelled."""
        # Wan2GP fires callbacks on its job thread, where Gradio has no event context and would drop the updates
        ctx = contextvars.copy_context()
        inner = create_gradio_progress_callbacks(progress)

        class Callbacks:
            def on_status(self, status):
                ctx.run(inner.on_status, status)

            def on_progress(self, update):
                ctx.run(inner.on_progress, update)

        try:
            job = submit(Callbacks())
        except RuntimeError as e:
            raise gr.Error(str(e))
        active["job"] = job
        try:
            result = job.result()
        finally:
            active["job"] = None
        if result.success and result.generated_files:
            return result.generated_files[-1]  # last file is the final video when sliding windows save intermediates
        if result.cancelled:
            return None
        raise gr.Error(str(result.errors[0]) if result.errors else "Wan2GP không trả về file nào.")

    def generate(settings, windows, progress):
        """Run a generation job and record actual / estimated time for its model. Returns (video, took_text)."""
        model = settings["model_type"]
        start = time.monotonic()
        video = run(lambda cb: session.submit_task(settings, callbacks=cb), progress)
        if video is None:
            return None, ""
        took = time.monotonic() - start
        save_calib(model, took, estimate_seconds(model, windows, settings["resolution"]))
        text = f"Thực tế {fmt_minutes(took)}"
        removed = len(windows) - 1  # keep_intermediate_sliding_windows = 0: each window replaced the previous file
        if removed > 0:
            gr.Info(f"Đã xóa {removed} video trung gian, chỉ giữ video cuối.")
            text += f" · đã xóa {removed} video trung gian"
        return video, text

    def cancel():
        job = active["job"]
        if job is not None and not job.done:
            job.cancel()

    def upscale(video, progress=gr.Progress()):
        if not video:
            raise gr.Error("Chưa có video.")
        out = run(lambda cb: session.submit_media_postprocessing(video, spatial_upsampling="lanczos*2", callbacks=cb), progress)
        return out or gr.update()

    def lora_names():
        return session.list_loras(REELS_MODEL)["loras"]

    def lora_choices():
        return gr.update(choices=lora_names())

    def results(extend=False):
        """Right column of a tab: actions, estimate, video, timing, follow-up buttons."""
        with gr.Row():
            cancel_btn = gr.Button("Hủy", variant="secondary")
            go = gr.Button("Tạo Video", variant="primary")
        eta = gr.Markdown(elem_classes="eta")
        video = gr.Video(label="Video ra", autoplay=True, interactive=False)
        took = gr.Markdown(elem_classes="eta")
        with gr.Row():
            ext = gr.Button("Kéo dài video (neo ảnh gốc)", variant="secondary", interactive=False) if extend else None
            up = gr.Button("Upscale ×2", variant="secondary", interactive=False)
        cancel_btn.click(cancel, queue=False)
        up.click(upscale, video, video)
        return go, eta, video, took, ext, up

    def image_resolution(image, quality):
        return pick_resolution(*_image_size(image), quality) if image else RESOLUTIONS[quality][0]

    # ---------- Reels ----------
    def reel_status(image, seconds, quality):
        info = ""
        if image:
            w, h = _image_size(image)
            info = f"**{Path(image).name}** · gốc {w}×{h} → video {pick_resolution(w, h, quality)}"
        model, windows = reel_windows(seconds)
        return info, eta_text(model, windows, image_resolution(image, quality), load_calib())

    def make_reel(image, prompt, seconds, quality, smooth, loras, high, low, detail, seed, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh đầu vào.")
        if not (prompt or "").strip():
            raise gr.Error("Chưa có prompt.")
        model, windows = reel_windows(seconds)
        settings = reels_settings(
            session.get_default_settings(model),
            image=image, prompt=prompt, resolution=image_resolution(image, quality),
            seconds=seconds, seed=seed, loras=loras, high=high, low=low, smooth=smooth, detail=detail,
        )
        video, took = generate(settings, windows, progress)
        if video is None:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        last = {"video": video, "image": image, "resolution": settings["resolution"]}
        return video, last, gr.update(interactive=True), gr.update(interactive=True), took

    def extend_reel(last, prompt, seconds, smooth, loras, high, low, detail, seed, progress=gr.Progress()):
        if not last:
            raise gr.Error("Chưa có video để kéo dài.")
        if not (prompt or "").strip():
            raise gr.Error("Chưa có prompt.")
        settings = extend_settings(
            session.get_default_settings(EXTEND_MODEL),
            video=last["video"], anchor=last["image"], prompt=prompt, resolution=last["resolution"],
            seconds=min(seconds, 5), seed=seed, loras=loras, high=high, low=low, smooth=smooth, detail=detail,
        )
        video, took = generate(settings, [settings["video_length"]], progress)
        if video is None:
            return gr.update(), gr.update(), gr.update()
        return video, {**last, "video": video}, took

    def suggest_ideas(image, hint, seconds, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh đầu vào.")
        if active["job"] is not None:
            raise gr.Error("Đang tạo video, chờ xong rồi gợi ý.")
        text = (hint or "").strip() or "Suggest short video ideas for this image."
        ideas = parse_ideas(enhance(session, "", text, image, ideas_instructions(int(seconds)), progress))
        if not ideas:
            raise gr.Error("Không nhận được ý tưởng nào, thử lại.")
        return gr.update(choices=ideas, value=None)

    def write_detail(image, hint, seconds, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh đầu vào.")
        if not (hint or "").strip():
            raise gr.Error("Nhập ý ngắn hoặc chọn một ý tưởng trước.")
        if active["job"] is not None:
            raise gr.Error("Đang tạo video, chờ xong rồi gợi ý.")
        return enhance(session, "", hint.strip(), image, detail_instructions(int(seconds)), progress)

    # ---------- Reels Multi Image ----------
    def multi_status(n, quality, *vals):
        images, durations = vals[:n], vals[MAX_IMAGES:MAX_IMAGES + n]
        count = f"**{n}/{MAX_IMAGES} ảnh · tổng {sum(durations):g} giây**"
        eta = eta_text(REELS_MODEL, [int(d) * FPS + 1 for d in durations], image_resolution(images[0], quality), load_calib())
        off = aspect_mismatch([_image_size(im) if im else None for im in images])
        if off:
            eta += f"\n\nLưu ý: ảnh {', '.join(map(str, off))} lệch tỉ lệ khung với ảnh 1 hơn 10%: Wan2GP sẽ co và cắt."
        return count, eta

    def multi_add(n):
        n = min(MAX_IMAGES, n + 1)
        return [n] + [gr.update(visible=i < n) for i in range(MAX_IMAGES)]

    def multi_remove(index):
        def remove(n, *vals):
            images, durations, prompts = list(vals[:MAX_IMAGES]), list(vals[MAX_IMAGES:2 * MAX_IMAGES]), list(vals[2 * MAX_IMAGES:])
            for values, blank in ((images, None), (durations, 3), (prompts, "")):
                del values[index]
                values.append(blank)
            n = max(1, n - 1)
            return [n, *images, *durations, *prompts] + [gr.update(visible=i < n) for i in range(MAX_IMAGES)]

        return remove

    # progress must come before *vals: Gradio only injects gr.Progress into positional parameters
    def make_multi(n, common, quality, smooth, loras, high, low, detail, seed, progress=gr.Progress(), *vals):
        images = list(vals[:n])
        durations = [int(d) for d in vals[MAX_IMAGES:MAX_IMAGES + n]]
        prompts = list(vals[2 * MAX_IMAGES:2 * MAX_IMAGES + n])
        if not all(images):
            raise gr.Error("Ô ảnh nào đang hiện cũng phải có ảnh.")
        if not (common or "").strip() and not all((p or "").strip() for p in prompts):
            raise gr.Error("Mỗi ảnh cần prompt riêng, hoặc điền Mô tả nhân vật chung.")
        settings = multi_settings(
            session.get_default_settings(REELS_MODEL),
            images=images, durations=durations, prompts=prompts, common=common,
            resolution=image_resolution(images[0], quality), seed=seed, loras=loras, high=high, low=low, smooth=smooth, detail=detail,
        )
        video, took = generate(settings, [d * FPS + 1 for d in durations], progress)
        if video is None:
            return gr.update(), gr.update(), gr.update()
        return video, gr.update(interactive=True), took

    # ---------- Motion Control ----------
    def motion_info(video):
        if not video:
            return "", gr.update()
        fps, total = _video_info(video)
        seconds = total / fps
        info = f"**Video mẫu:** {seconds:.1f} giây · {fps:g} fps → {motion_frames(fps, total)} frame ở {FPS} fps"
        return info, gr.update(maximum=max(1, min(15, int(seconds))))

    def motion_eta(image, video, limit_on, limit, quality, label, outfit):
        if not video:
            return "Chọn video mẫu để ước tính."
        model = MOTION_MODELS[label]
        fps, total = _video_info(video)
        frames = motion_frames(fps, total, limit if limit_on else None)
        windows = sliding_windows(frames, *MOTION_WINDOWS[model])
        text = eta_text(model, windows, image_resolution(image, quality), load_calib())
        if outfit and model == ANIMATE_MODEL:
            text += "\n\nLưu ý: Wan 2.2 Animate 14B luôn giữ trang phục theo ảnh."
        return text

    def make_motion(image, video, description, limit_on, limit, quality, smooth, label, audio, outfit, seed, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh nhân vật.")
        if not video:
            raise gr.Error("Chưa có video mẫu.")
        fps, total = _video_info(video)
        if total < 5:
            raise gr.Error("Không đọc được video mẫu.")
        model = MOTION_MODELS[label]
        frames = motion_frames(fps, total, limit if limit_on else None)
        settings = motion_settings(
            session.get_default_settings(model), image=image, video=video, description=description,
            resolution=image_resolution(image, quality), frames=frames, seed=seed, model=model, smooth=smooth,
            audio=audio, outfit=outfit,
        )
        out, took = generate(settings, sliding_windows(frames, *MOTION_WINDOWS[model]), progress)
        if out is None:
            return gr.update(), gr.update(), gr.update()
        return out, gr.update(interactive=True), took

    with gr.Blocks(title="Simple App", theme=make_theme(), css=CSS, js=FORCE_DARK) as demo:
        r_last = gr.State(None)
        with gr.Row(elem_id="header"):
            gr.HTML("<h1>Simple App</h1><p>Reels · Reels Multi Image · Motion Control</p>")
            guide_btn = gr.Button("Hướng dẫn", variant="secondary", size="sm", scale=0)
        with gr.Column(elem_id="guide"):
            gr.Markdown(GUIDE)
            close_btn = gr.Button("Đóng", variant="secondary")

        with gr.Tab("Reels"):
            with gr.Row():
                with gr.Column(scale=1):
                    r_image = gr.Image(type="filepath", label="Ảnh đầu vào", sources=["upload", "clipboard"])
                    r_info = gr.Markdown(elem_classes="eta")
                    r_hint = gr.Textbox(label="Ý ngắn", placeholder="chàng trai thong thả đi đến gần")
                    with gr.Row():
                        r_ideas_btn = gr.Button("3 ý tưởng", variant="secondary", scale=0)
                        r_ideas = gr.Dropdown(choices=[], label="Ý tưởng", scale=2)
                        r_detail_btn = gr.Button("Viết chi tiết", variant="secondary", scale=0)
                    r_prompt = gr.Textbox(label="Prompt", lines=8)
                with gr.Column(scale=1):
                    r_go, r_eta, r_video, r_took, r_extend, r_up = results(extend=True)
                    with gr.Accordion("Cài đặt nâng cao", open=True):
                        r_seconds = gr.Slider(2, 15, value=5, step=1, label="Thời lượng video (giây)")
                        r_quality = gr.Radio(["480p", "720p"], value="480p", label="Độ phân giải")
                        r_smooth = gr.Checkbox(value=True, label="Mượt x2 (32 fps)")
                        r_loras = gr.Dropdown(choices=lora_names(), multiselect=True, label="LoRA (loras/wan)")
                        with gr.Row():
                            r_high = gr.Slider(0, 2, value=1, step=0.1, label="Pha đầu (chuyển động)")
                            r_low = gr.Slider(0, 2, value=0, step=0.1, label="Pha cuối (chi tiết mặt)")
                        r_detail = gr.Checkbox(False, label="Giữ da & tóc (LoRA chi tiết, nên dùng cho ảnh cận mặt)")
                        r_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = ngẫu nhiên)")

        with gr.Tab("Reels Multi Image"):
            mm_n = gr.State(2)
            mm_images, mm_durations, mm_prompts, mm_cards, mm_removes = [], [], [], [], []
            with gr.Row():
                with gr.Column(scale=2):
                    mm_count = gr.Markdown()
                    mm_common = gr.Textbox(
                        label="Mô tả nhân vật chung (ghép vào đầu mọi prompt)",
                        placeholder="a young man with short black hair, white t-shirt, in a sunny park",
                    )
                    with gr.Row():
                        for i in range(MAX_IMAGES):
                            with gr.Column(visible=i < 2, min_width=220) as card:
                                mm_images.append(gr.Image(type="filepath", label=f"Ảnh {i + 1}", sources=["upload", "clipboard"], height=220))
                                mm_durations.append(gr.Slider(1, 5, value=3, step=1, label="Thời lượng đoạn (giây)"))
                                mm_prompts.append(gr.Textbox(label=f"Prompt {i + 1}", lines=3, placeholder="hành động trong đoạn này"))
                                mm_removes.append(gr.Button("Xóa", variant="secondary", size="sm"))
                            mm_cards.append(card)
                    mm_add = gr.Button("Thêm ảnh", variant="secondary")
                    with gr.Accordion("Cài đặt nâng cao", open=False):
                        mm_quality = gr.Radio(["480p", "720p"], value="480p", label="Độ phân giải (theo ảnh 1)")
                        mm_smooth = gr.Checkbox(value=True, label="Mượt x2 (32 fps)")
                        mm_loras = gr.Dropdown(choices=lora_names(), multiselect=True, label="LoRA (loras/wan)")
                        with gr.Row():
                            mm_high = gr.Slider(0, 2, value=1, step=0.1, label="Pha đầu (chuyển động)")
                            mm_low = gr.Slider(0, 2, value=0, step=0.1, label="Pha cuối (chi tiết mặt)")
                        mm_detail = gr.Checkbox(False, label="Giữ da & tóc (LoRA chi tiết)")
                        mm_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = ngẫu nhiên)")
                with gr.Column(scale=1):
                    mm_go, mm_eta, mm_video, mm_took, _, mm_up = results()

        with gr.Tab("Motion Control"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        m_image = gr.Image(type="filepath", label="Ảnh nhân vật", sources=["upload", "clipboard"])
                        m_video = gr.Video(label="Video tham chiếu", sources=["upload"])
                    m_info = gr.Markdown(elem_classes="eta")
                    m_desc = gr.Textbox(
                        label="Mô tả nhân vật, bối cảnh (không bắt buộc)", lines=2,
                        placeholder="Character appearance: ... Background: ...",
                    )
                    with gr.Accordion("Cài đặt nâng cao", open=True):
                        with gr.Row():
                            m_limit_on = gr.Checkbox(value=False, label="Giới hạn thời lượng", scale=0)
                            m_limit = gr.Slider(1, 15, value=5, step=1, label="Thời lượng (giây, cắt từ đầu video mẫu)", interactive=False)
                        m_quality = gr.Radio(["480p", "720p"], value="480p", label="Độ phân giải")
                        m_smooth = gr.Checkbox(value=True, label="Mượt x2 (16 → 32 fps)")
                        with gr.Row():
                            m_audio = gr.Checkbox(True, label="Giữ âm thanh video mẫu")
                            m_outfit = gr.Checkbox(False, label="Trang phục theo video mẫu")
                        m_model = gr.Dropdown(list(MOTION_MODELS), value="Animate 2 distilled", label="Chọn Model")
                        m_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = ngẫu nhiên)")
                    if not session.get_model_availability(MOTION_MODEL)["available"]:
                        gr.Markdown("Lần đầu dùng Animate 2 sẽ tải model (17 GB).")
                with gr.Column(scale=1):
                    m_go, m_eta, m_out, m_took, _, m_up = results()

        # purely visual toggle: run it in the browser, no server round trip
        guide_btn.click(None, js="() => { document.getElementById('guide').style.display = 'flex'; }")
        close_btn.click(None, js="() => { document.getElementById('guide').style.display = 'none'; }")

        reel_in = [r_image, r_seconds, r_quality]
        for comp in reel_in:
            comp.change(reel_status, reel_in, [r_info, r_eta])
        demo.load(reel_status, reel_in, [r_info, r_eta])
        r_loras.focus(lora_choices, outputs=r_loras)
        r_ideas_btn.click(suggest_ideas, [r_image, r_hint, r_seconds], r_ideas)
        r_ideas.change(lambda idea: idea or gr.update(), r_ideas, r_hint)
        r_detail_btn.click(write_detail, [r_image, r_hint, r_seconds], r_prompt)
        r_go.click(
            make_reel,
            [r_image, r_prompt, r_seconds, r_quality, r_smooth, r_loras, r_high, r_low, r_detail, r_seed],
            [r_video, r_last, r_extend, r_up, r_took],
        )
        r_extend.click(
            extend_reel,
            [r_last, r_prompt, r_seconds, r_smooth, r_loras, r_high, r_low, r_detail, r_seed],
            [r_video, r_last, r_took],
        )

        slots = [*mm_images, *mm_durations, *mm_prompts]
        status_in = [mm_n, mm_quality, *mm_images, *mm_durations]
        for comp in (*mm_images, *mm_durations, mm_quality):
            comp.change(multi_status, status_in, [mm_count, mm_eta])
        demo.load(multi_status, status_in, [mm_count, mm_eta])
        mm_add.click(multi_add, mm_n, [mm_n, *mm_cards]).then(multi_status, status_in, [mm_count, mm_eta])
        for i, rm in enumerate(mm_removes):
            rm.click(multi_remove(i), [mm_n, *slots], [mm_n, *slots, *mm_cards]).then(multi_status, status_in, [mm_count, mm_eta])
        mm_loras.focus(lora_choices, outputs=mm_loras)
        mm_go.click(
            make_multi,
            [mm_n, mm_common, mm_quality, mm_smooth, mm_loras, mm_high, mm_low, mm_detail, mm_seed, *slots],
            [mm_video, mm_up, mm_took],
        )

        m_limit_on.change(lambda on: gr.update(interactive=on), m_limit_on, m_limit)
        m_video.change(motion_info, m_video, [m_info, m_limit])
        motion_in = [m_image, m_video, m_limit_on, m_limit, m_quality, m_model, m_outfit]
        for comp in motion_in:
            comp.change(motion_eta, motion_in, m_eta)
        demo.load(motion_eta, motion_in, m_eta)
        m_go.click(
            make_motion,
            [m_image, m_video, m_desc, m_limit_on, m_limit, m_quality, m_smooth, m_model, m_audio, m_outfit, m_seed],
            [m_out, m_up, m_took],
        )
    return demo


def run_check(session, flag, args):
    """Headless smoke checks: python simple_app.py --check-<name> <args...>"""

    def generate(settings, label):
        result = session.submit_task(settings).result()
        assert result.success and result.generated_files, [str(e) for e in result.errors]
        path = result.generated_files[-1]
        print(f"{label}:", path)
        return path

    def check_duration(path, expected):
        fps, total = _video_info(path)
        got = total / fps
        print(f"duration {got:.2f}s ({total} frames @ {fps:g} fps), expected {expected:.2f}s")
        assert abs(got - expected) <= 0.2, "wrong duration"

    if flag in ("--check-enhancer", "--check-reel"):
        image, *rest = args
        seconds = int(rest[0]) if rest else 5
        print("enhancer_enabled =", sys.modules["wgp"].server_config.get("enhancer_enabled", 0))
        ideas = parse_ideas(enhance(session, "", "Suggest short video ideas for this image.", image, ideas_instructions(seconds)))
        print("IDEAS:", ideas)
        assert ideas, "no ideas returned"
        detail = enhance(session, "", ideas[0], image, detail_instructions(seconds))
        print("DETAIL:", detail)
        assert detail.strip(), "empty detailed prompt"
        print("enhancer check ok")
        if flag == "--check-reel":
            w, h = _image_size(image)
            model, _ = reel_plan(seconds)
            out = generate(reels_settings(
                session.get_default_settings(model),
                image=image, prompt=detail, resolution=pick_resolution(w, h), seconds=seconds, seed=-1,
            ), "REEL")
            check_duration(out, seconds)
    elif flag == "--check-extend":
        video, image, *prompt_words = args
        w, h = _image_size(image)
        generate(extend_settings(
            session.get_default_settings(EXTEND_MODEL),
            video=video, anchor=image, prompt=" ".join(prompt_words), resolution=pick_resolution(w, h), seconds=2, seed=-1,
        ), "EXTENDED")
    elif flag == "--check-motion":
        image, video, *rest = args
        model = ANIMATE_MODEL if rest and rest[0] == "animate" else MOTION_MODEL
        limit = float(rest[1]) if len(rest) > 1 else None
        fps, total = _video_info(video)
        w, h = _image_size(image)
        out = generate(motion_settings(
            session.get_default_settings(model), image=image, video=video, description="",
            resolution=pick_resolution(w, h), frames=motion_frames(fps, total, limit), seed=-1, model=model,
        ), "MOTION")
        check_duration(out, min(limit, total / fps) if limit else total / fps)
    else:
        raise SystemExit(f"unknown check: {flag}")


def main():
    import os

    flag = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("--check-") else None
    # resolve file arguments before chdir; other arguments (a prompt) pass through unchanged
    args = [str(Path(a).resolve()) if Path(a).exists() else a for a in sys.argv[2:]]
    os.chdir(ROOT)
    from shared.api import init

    session = init(root=ROOT, cli_args=["--profile", "4"])
    # in memory only (wgp_config.json is untouched): each new sliding window replaces the previous window's file
    sys.modules["wgp"].server_config["keep_intermediate_sliding_windows"] = 0
    if flag:
        run_check(session, flag, args)
        return
    build_ui(session).queue().launch(inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
