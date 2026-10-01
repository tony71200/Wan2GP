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


def _lora_settings(loras, high, low, smooth):
    names, mults = pair_loras(list(loras or []), high, low)
    return {
        "activated_loras": names,
        "loras_multipliers": mults,
        "temporal_upsampling": "rife*2" if smooth else "",
    }


def reels_settings(defaults, *, image, prompt, resolution, seconds, seed, loras=(), high=1.0, low=0.0, smooth=True):
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
        **_lora_settings(loras, high, low, smooth),
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


def extend_settings(defaults, *, video, anchor, prompt, resolution, seconds, seed, loras=(), high=1.0, low=0.0, smooth=True):
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
        **_lora_settings(loras, high, low, smooth),
    }


MAX_IMAGES = 6


def _one_line(*parts):
    """Join non-empty parts with '. ' on a single line (PW windows must not contain blank lines)."""
    cleaned = [" ".join(str(p).split()).rstrip(".") for p in parts if p and str(p).strip()]
    return ". ".join(c for c in cleaned if c)


def multi_settings(defaults, *, images, durations, prompts, common, resolution, seed, loras=(), high=1.0, low=0.0, smooth=True):
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
        **_lora_settings(loras, high, low, smooth),
    }


def aspect_mismatch(sizes, tolerance=0.1):
    """1-based positions of images whose aspect ratio differs from the first one by more than `tolerance`."""
    if not sizes or not sizes[0]:
        return []
    ref = sizes[0][0] / sizes[0][1]
    return [i for i, size in enumerate(sizes[1:], 2) if size and abs(size[0] / size[1] / ref - 1) > tolerance]


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
    return f"⏱ ≈ {eta} ({MODEL_LABELS[model]} · {len(windows)} cửa sổ × {steps} bước)"


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


def motion_settings(defaults, *, image, video, description, resolution, frames, seed, model=MOTION_MODEL, smooth=False):
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


def build_ui(session):
    import contextvars

    import gradio as gr
    from shared.api import create_gradio_progress_callbacks

    active = {"job": None}

    def run(settings, progress):
        # Wan2GP fires callbacks on its job thread, where Gradio has no event context and would drop the updates
        ctx = contextvars.copy_context()
        inner = create_gradio_progress_callbacks(progress)

        class Callbacks:
            def on_status(self, status):
                ctx.run(inner.on_status, status)

            def on_progress(self, update):
                ctx.run(inner.on_progress, update)

        try:
            job = session.submit_task(settings, callbacks=Callbacks())
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

    def cancel():
        job = active["job"]
        if job is not None and not job.done:
            job.cancel()

    def lora_choices():
        return gr.update(choices=session.list_loras(REELS_MODEL)["loras"])

    def make_reel(image, prompt, seconds, quality, smooth, loras, high, low, seed, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh đầu vào.")
        if not (prompt or "").strip():
            raise gr.Error("Chưa có prompt.")
        w, h = _image_size(image)
        settings = reels_settings(
            session.get_default_settings(REELS_MODEL),
            image=image, prompt=prompt, resolution=pick_resolution(w, h, quality),
            seconds=seconds, seed=seed, loras=loras, high=high, low=low, smooth=smooth,
        )
        video = run(settings, progress)
        if video is None:
            return gr.update(), gr.update(), gr.update()
        return video, {"video": video, "image": image, "resolution": settings["resolution"]}, gr.update(interactive=True)

    def extend_reel(last, prompt, seconds, smooth, loras, high, low, seed, progress=gr.Progress()):
        if not last:
            raise gr.Error("Chưa có video để kéo dài.")
        if not (prompt or "").strip():
            raise gr.Error("Chưa có prompt.")
        settings = extend_settings(
            session.get_default_settings(EXTEND_MODEL),
            video=last["video"], anchor=last["image"], prompt=prompt, resolution=last["resolution"],
            seconds=seconds, seed=seed, loras=loras, high=high, low=low, smooth=smooth,
        )
        video = run(settings, progress)
        if video is None:
            return gr.update(), gr.update()
        return video, {**last, "video": video}

    def make_motion(image, video, description, limit, quality, seed, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh nhân vật.")
        if not video:
            raise gr.Error("Chưa có video mẫu.")
        fps, total = _video_info(video)
        if total < 5:
            raise gr.Error("Không đọc được video mẫu.")
        w, h = _image_size(image)
        settings = motion_settings(
            session.get_default_settings(MOTION_MODEL), image=image, video=video, description=description,
            resolution=pick_resolution(w, h, quality),
            frames=round_frames(min(total, limit * fps)), seed=seed,
        )
        out = run(settings, progress)
        return out if out else gr.update()

    def suggest_ideas(image, hint, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh đầu vào.")
        if active["job"] is not None:
            raise gr.Error("Đang tạo video, chờ xong rồi gợi ý.")
        text = (hint or "").strip() or "Suggest short video ideas for this image."
        ideas = parse_ideas(enhance(session, "", text, image, ideas_instructions(), progress))
        if not ideas:
            raise gr.Error("Không nhận được ý tưởng nào, thử lại.")
        return gr.update(choices=ideas, value=None, visible=True)

    def write_detail(image, hint, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh đầu vào.")
        if not (hint or "").strip():
            raise gr.Error("Nhập ý ngắn hoặc chọn một ý tưởng trước.")
        if active["job"] is not None:
            raise gr.Error("Đang tạo video, chờ xong rồi gợi ý.")
        return enhance(session, "", hint.strip(), image, detail_instructions(), progress)

    with gr.Blocks(title="Simple App") as demo:
        r_last = gr.State(None)
        with gr.Tab("Reels"):
            with gr.Row():
                with gr.Column():
                    r_image = gr.Image(type="filepath", label="Ảnh đầu vào", sources=["upload", "clipboard"])
                    r_hint = gr.Textbox(label="Ý ngắn", placeholder="cô ấy quay lại và cười")
                    with gr.Row():
                        r_ideas_btn = gr.Button("3 ý tưởng")
                        r_detail_btn = gr.Button("Viết chi tiết")
                    r_ideas = gr.Radio(choices=[], label="Ý tưởng", visible=False)
                    r_prompt = gr.Textbox(label="Prompt", lines=5)
                    r_seconds = gr.Slider(2, 5, value=5, step=1, label="Thời lượng (giây)")
                    with gr.Row():
                        r_quality = gr.Radio(["480p", "720p"], value="480p", label="Chất lượng")
                        r_smooth = gr.Checkbox(value=True, label="Mượt x2 (32 fps)")
                    r_loras = gr.Dropdown(choices=session.list_loras(REELS_MODEL)["loras"], multiselect=True, label="LoRA (loras/wan)")
                    with gr.Row():
                        r_high = gr.Slider(0, 2, value=1, step=0.1, label="Pha đầu")
                        r_low = gr.Slider(0, 2, value=0, step=0.1, label="Pha sau")
                    r_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = ngẫu nhiên)")
                    with gr.Row():
                        r_go = gr.Button("Tạo video", variant="primary")
                        r_cancel = gr.Button("Hủy")
                with gr.Column():
                    r_video = gr.Video(label="Video kết quả", autoplay=True)
                    r_extend = gr.Button("Kéo dài video (neo ảnh gốc)", interactive=False)

        with gr.Tab("Motion control"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        m_image = gr.Image(type="filepath", label="Ảnh nhân vật", sources=["upload", "clipboard"])
                        m_video = gr.Video(label="Video mẫu", sources=["upload"])
                    m_desc = gr.Textbox(
                        label="Mô tả nhân vật (không bắt buộc)", lines=2,
                        placeholder="Character appearance: ... Background: ...",
                    )
                    m_limit = gr.Slider(2, 10, value=5, step=1, label="Giới hạn thời lượng (giây)")
                    m_quality = gr.Radio(["480p", "720p"], value="480p", label="Chất lượng")
                    m_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = ngẫu nhiên)")
                    if not session.get_model_availability(MOTION_MODEL)["available"]:
                        gr.Markdown("Lần đầu sẽ tải model Animate 2 (17 GB).")
                    with gr.Row():
                        m_go = gr.Button("Tạo video", variant="primary")
                        m_cancel = gr.Button("Hủy")
                with gr.Column():
                    m_out = gr.Video(label="Video kết quả", autoplay=True)
                    gr.Markdown("Nhân vật luôn tham chiếu ảnh gốc ở mọi đoạn.")

        r_loras.focus(lora_choices, outputs=r_loras)
        r_ideas_btn.click(suggest_ideas, [r_image, r_hint], r_ideas)
        r_ideas.change(lambda idea: idea or gr.update(), r_ideas, r_hint)
        r_detail_btn.click(write_detail, [r_image, r_hint], r_prompt)
        r_go.click(
            make_reel,
            [r_image, r_prompt, r_seconds, r_quality, r_smooth, r_loras, r_high, r_low, r_seed],
            [r_video, r_last, r_extend],
        )
        r_extend.click(
            extend_reel,
            [r_last, r_prompt, r_seconds, r_smooth, r_loras, r_high, r_low, r_seed],
            [r_video, r_last],
        )
        r_cancel.click(cancel, queue=False)
        m_go.click(make_motion, [m_image, m_video, m_desc, m_limit, m_quality, m_seed], m_out)
        m_cancel.click(cancel, queue=False)
    return demo


def run_check(session, flag, args):
    """Headless smoke checks: python simple_app.py --check-<name> <args...>"""

    def generate(settings, label):
        result = session.submit_task(settings).result()
        assert result.success and result.generated_files, [str(e) for e in result.errors]
        print(f"{label}:", result.generated_files[-1])

    if flag in ("--check-enhancer", "--check-reel"):
        (image,) = args
        print("enhancer_enabled =", sys.modules["wgp"].server_config.get("enhancer_enabled", 0))
        ideas = parse_ideas(enhance(session, "", "Suggest short video ideas for this image.", image, ideas_instructions()))
        print("IDEAS:", ideas)
        assert ideas, "no ideas returned"
        detail = enhance(session, "", ideas[0], image, detail_instructions())
        print("DETAIL:", detail)
        assert detail.strip(), "empty detailed prompt"
        print("enhancer check ok")
        if flag == "--check-reel":
            w, h = _image_size(image)
            generate(reels_settings(
                session.get_default_settings(REELS_MODEL),
                image=image, prompt=detail, resolution=pick_resolution(w, h), seconds=5, seed=-1,
            ), "REEL")
    elif flag == "--check-extend":
        video, image, *prompt_words = args
        w, h = _image_size(image)
        generate(extend_settings(
            session.get_default_settings(EXTEND_MODEL),
            video=video, anchor=image, prompt=" ".join(prompt_words), resolution=pick_resolution(w, h), seconds=2, seed=-1,
        ), "EXTENDED")
    elif flag == "--check-motion":
        image, video = args
        fps, total = _video_info(video)
        w, h = _image_size(image)
        generate(motion_settings(
            session.get_default_settings(MOTION_MODEL), image=image, video=video, description="",
            resolution=pick_resolution(w, h), frames=round_frames(min(total, 5 * fps)), seed=-1,
        ), "MOTION")
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
    if flag:
        run_check(session, flag, args)
        return
    build_ui(session).queue().launch(inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
