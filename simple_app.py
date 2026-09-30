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


def _lora_settings(loras, high, low, smooth):
    names, mults = pair_loras(list(loras or []), high, low)
    return {
        "activated_loras": names,
        "loras_multipliers": mults,
        "temporal_upsampling": "rife*2" if smooth else "",
    }


def reels_settings(defaults, *, image, prompt, resolution, seconds, seed, loras=(), high=1.0, low=0.0, smooth=True):
    return {
        **defaults,
        "model_type": REELS_MODEL,
        "prompt": prompt,
        "image_prompt_type": "S",
        "image_start": image,
        "resolution": resolution,
        "video_length": frames_for(seconds),
        "seed": int(seed),
        **_lora_settings(loras, high, low, smooth),
    }


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
        "video_length": frames_for(seconds),
        "seed": int(seed),
        **_lora_settings(loras, high, low, smooth),
    }


def motion_settings(defaults, *, image, video, description, resolution, frames, seed):
    """Animate 2 with the character image as a reference for every window."""
    return {
        **defaults,
        "model_type": MOTION_MODEL,
        "prompt": (description or "").strip() or defaults.get("prompt", ""),
        "image_prompt_type": "T",
        "image_start": None,
        "video_prompt_type": "UVI",
        "image_refs": [image],
        "video_guide": video,
        "resolution": resolution,
        "video_length": int(frames),
        "force_fps": "control",
        "seed": int(seed),
    }


IDEAS_INSTRUCTIONS = (
    "You are a creative director for short vertical social videos. Look at the image and propose "
    "exactly 3 different short video ideas that start from this exact image. Each idea must work as one "
    "continuous 5 second shot in the same place: no cuts, no change of location, lighting or time of day. "
    "Keep the same person, face, hairstyle and setting, and prefer small natural actions over big ones. "
    "The person keeps wearing exactly what they wear in the image. "
    "Write one idea per line, one sentence each, with no numbering and no extra text."
)

# The model's own enhancer instructions ask for shot changes every second, which turns the person
# into someone else. These keep one continuous shot and restate the subject on every line.
DETAIL_INSTRUCTIONS = (
    "You are an AI vision model writing a prompt for an image-to-video model. Analyze the provided image and the user's goal.\n"
    "Describe ONE continuous shot that starts exactly from the image: no cuts, no scene change, no change of location, lighting, weather or time of day.\n"
    "The person must stay the same person for the whole video: same face, hairstyle, hair color, body, skin tone, outfit and accessories as in the image. Never describe a different look.\n"
    "Keep the motion natural and modest so the face stays recognizable. The camera is static or moves slowly.\n"
    "Output rules:\n"
    "Write one line per second, from 0 to 5 seconds.\n"
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


def build_ui(session):
    import gradio as gr
    from PIL import Image
    from shared.api import create_gradio_progress_callbacks

    active = {"job": None}

    def run(settings, progress):
        try:
            job = session.submit_task(settings, callbacks=create_gradio_progress_callbacks(progress))
        except RuntimeError as e:
            raise gr.Error(str(e))
        active["job"] = job
        try:
            result = job.result()
        finally:
            active["job"] = None
        if result.success and result.generated_files:
            return result.generated_files[0]
        if result.cancelled:
            return None
        raise gr.Error(str(result.errors[0]) if result.errors else "Wan2GP không trả về file nào.")

    def cancel():
        job = active["job"]
        if job is not None and not job.done:
            job.cancel()

    def image_size(path):
        with Image.open(path) as im:
            return im.size

    def lora_choices():
        return gr.update(choices=session.list_loras(REELS_MODEL)["loras"])

    def make_reel(image, prompt, seconds, quality, smooth, loras, high, low, seed, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh đầu vào.")
        if not (prompt or "").strip():
            raise gr.Error("Chưa có prompt.")
        w, h = image_size(image)
        settings = reels_settings(
            session.get_default_settings(REELS_MODEL),
            image=image, prompt=prompt, resolution=pick_resolution(w, h, quality),
            seconds=seconds, seed=seed, loras=loras, high=high, low=low, smooth=smooth,
        )
        video = run(settings, progress)
        if video is None:
            return gr.update(), gr.update()
        return video, {"video": video, "image": image, "resolution": settings["resolution"]}

    with gr.Blocks(title="Simple App") as demo:
        r_last = gr.State(None)
        with gr.Tab("Reels"):
            with gr.Row():
                with gr.Column():
                    r_image = gr.Image(type="filepath", label="Ảnh đầu vào", sources=["upload", "clipboard"])
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

        r_loras.focus(lora_choices, outputs=r_loras)
        r_go.click(
            make_reel,
            [r_image, r_prompt, r_seconds, r_quality, r_smooth, r_loras, r_high, r_low, r_seed],
            [r_video, r_last],
        )
        r_cancel.click(cancel, queue=False)
    return demo


def main():
    import os

    flag = sys.argv[1] if len(sys.argv) == 3 else None
    check_image = str(Path(sys.argv[2]).resolve()) if flag in ("--check-enhancer", "--check-reel") else None
    os.chdir(ROOT)
    from shared.api import init

    session = init(root=ROOT, cli_args=["--profile", "4"])
    if check_image:
        print("enhancer_enabled =", sys.modules["wgp"].server_config.get("enhancer_enabled", 0))
        ideas = parse_ideas(enhance(session, "", "Suggest short video ideas for this image.", check_image, IDEAS_INSTRUCTIONS))
        print("IDEAS:", ideas)
        assert ideas, "no ideas returned"
        detail = enhance(session, "", ideas[0], check_image, DETAIL_INSTRUCTIONS)
        print("DETAIL:", detail)
        assert detail.strip(), "empty detailed prompt"
        print("enhancer check ok")
        if flag == "--check-reel":
            from PIL import Image

            with Image.open(check_image) as im:
                w, h = im.size
            settings = reels_settings(
                session.get_default_settings(REELS_MODEL),
                image=check_image, prompt=detail, resolution=pick_resolution(w, h), seconds=5, seed=-1,
            )
            result = session.submit_task(settings).result()
            assert result.success and result.generated_files, [str(e) for e in result.errors]
            print("REEL:", result.generated_files[0])
        return
    build_ui(session).queue().launch(inbrowser=True, show_error=True)


if __name__ == "__main__":
    main()
