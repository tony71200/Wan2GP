# Simple App (Reels + Motion Control) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Một app Gradio hai tab (`simple_app.py`) sinh video Reels từ ảnh và video Motion Control từ ảnh + video mẫu, chạy trên Wan2GP qua `shared.api`, ưu tiên giữ nguyên nhân vật trong ảnh.

**Architecture:** Một file `simple_app.py`: phần đầu là hàm thuần và hàm dựng settings (không import gradio/wgp, kiểm thử được không cần GPU); `enhance()` gọi prompt enhancer nội bộ của `wgp`; `build_ui(session)` dựng Gradio Blocks với các handler dạng closure; `main()` gọi `shared.api.init()` rồi launch. Không sửa file nào của Wan2GP.

**Tech Stack:** Python (env conda `wan2gp`), Gradio 5.29, `shared.api` của Wan2GP, Pillow, OpenCV (đều đã có trong `requirements.txt`).

**Spec:** `docs/superpowers/specs/2026-09-30-simple-app-design.md`

## Global Constraints

- Nhánh làm việc: `20260930_simple_app`. Không commit `run_wgp.bat` của người dùng.
- Không sửa `wgp.py` hay bất kỳ file nào dưới `shared/`, `models/`, `plugins/`, `defaults/`.
- Không thêm thư viện mới.
- Không xóa model hay file nào trong `ckpts/`, `loras/`.
- Không tự khởi động việc tải model lớn (Animate 2: 17 GB) khi chưa hỏi người dùng ở đúng bước đó.
- Tên model, chính xác: Reels `i2v_2_2_Enhanced_Lightning_v2`; kéo dài `i2v_2_2_Enhanced_Lightning_v2_svi2pro`; Motion Control `animate2_distilled`.
- Nhãn UI bằng tiếng Việt như trong spec.
- Đóng Wan2GP (`run_wgp.bat`) trước khi chạy app; hai tiến trình không dùng chung được GPU 12GB.
- Lệnh kiểm thử thuần (Git Bash): `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
- Lệnh chạy có GPU (Git Bash): `cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python simple_app.py <tham số>"`
- Mỗi commit kết thúc bằng dòng `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

| File | Trách nhiệm |
|---|---|
| `simple_app.py` (tạo) | Hằng số, hàm thuần, hàm dựng settings, `enhance()`, `build_ui()`, `main()` |
| `test_simple_app.py` (tạo) | Assert cho hàm thuần và hàm dựng settings; chạy bằng `python test_simple_app.py` |
| `run_simple_app.bat` (tạo) | Kích hoạt env `wan2gp` và chạy app |

---

### Task 1: Hàm thuần

**Files:**
- Create: `simple_app.py`
- Test: `test_simple_app.py`

**Interfaces:**
- Produces:
  - `REELS_MODEL`, `EXTEND_MODEL`, `MOTION_MODEL`: `str`
  - `pick_resolution(w: int, h: int, quality: str = "480p") -> str`
  - `frames_for(seconds: float, fps: int = 16) -> int`
  - `round_frames(n: float) -> int`
  - `pair_loras(names: list[str], high: float = 1.0, low: float = 0.0) -> tuple[list[str], str]`
  - `parse_ideas(text: str) -> list[str]`

- [ ] **Step 1: Viết test trước**

Tạo `test_simple_app.py`:

```python
from simple_app import (
    pick_resolution, frames_for, round_frames, pair_loras, parse_ideas,
)


def test_pick_resolution():
    assert pick_resolution(1080, 1920) == "480x832"
    assert pick_resolution(1920, 1080) == "832x480"
    assert pick_resolution(1000, 1000) == "720x720"
    assert pick_resolution(1000, 1050) == "720x720"
    assert pick_resolution(1080, 1920, "720p") == "720x1280"
    assert pick_resolution(1920, 1080, "720p") == "1280x720"
    assert pick_resolution(1024, 1024, "720p") == "960x960"


def test_frames_for():
    assert frames_for(2) == 33
    assert frames_for(3) == 49
    assert frames_for(4) == 65
    assert frames_for(5) == 81
    assert frames_for(0.5) == 17
    assert frames_for(10) == 81


def test_round_frames():
    assert round_frames(150) == 149
    assert round_frames(65) == 65
    assert round_frames(68) == 65
    assert round_frames(3) == 5


def test_pair_loras():
    names = [
        "Twerking_I2V_high_noise.safetensors",
        "Twerking_I2V_low_noise.safetensors",
        "slow_motion.safetensors",
        "transition_v2.0_H.safetensors",
        "transition_v2.0_L.safetensors",
        "wan2.2_i2v_lownoise_pov.safetensors",
        "x_HN.safetensors",
    ]
    activated, mults = pair_loras(names, 1.0, 0.5)
    assert activated == names
    assert mults == "1;0 0;0.5 1;0.5 1;0 0;0.5 0;0.5 1;0"
    assert pair_loras([], 1, 0) == ([], "")
    assert pair_loras(["a_high.safetensors"])[1] == "1;0"
    assert pair_loras(["a_low.safetensors"])[1] == "0;0"


def test_parse_ideas():
    text = "1. Wind blows her hair\n2) She turns around\n\n- She smiles\n4. extra"
    assert parse_ideas(text) == ["Wind blows her hair", "She turns around", "She smiles"]
    assert parse_ideas("") == []
    assert parse_ideas(None) == []
    assert parse_ideas('"Only one idea"') == ["Only one idea"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ModuleNotFoundError: No module named 'simple_app'`

- [ ] **Step 3: Viết phần cài đặt tối thiểu**

Tạo `simple_app.py`:

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận đạt**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add simple_app.py test_simple_app.py
git commit -m "Add pure helpers for simple app"
```

---

### Task 2: Hàm dựng settings

**Files:**
- Modify: `simple_app.py` (thêm sau `parse_ideas`)
- Test: `test_simple_app.py`

**Interfaces:**
- Consumes: `REELS_MODEL`, `EXTEND_MODEL`, `MOTION_MODEL`, `frames_for`, `pair_loras` (Task 1)
- Produces:
  - `reels_settings(defaults: dict, *, image: str, prompt: str, resolution: str, seconds: float, seed: int, loras=(), high=1.0, low=0.0, smooth=True) -> dict`
  - `extend_settings(defaults: dict, *, video: str, anchor: str, prompt: str, resolution: str, seconds: float, seed: int, loras=(), high=1.0, low=0.0, smooth=True) -> dict`
  - `motion_settings(defaults: dict, *, image: str, video: str, description: str, resolution: str, frames: int, seed: int) -> dict`

`defaults` là dict trả về từ `session.get_default_settings(model_type)`; các hàm không gọi session nên test được bằng dict thường.

- [ ] **Step 1: Viết test trước**

Trong `test_simple_app.py`, đổi dòng import thành:

```python
from simple_app import (
    REELS_MODEL, EXTEND_MODEL, MOTION_MODEL,
    pick_resolution, frames_for, round_frames, pair_loras, parse_ideas,
    reels_settings, extend_settings, motion_settings,
)
```

Thêm trước khối `if __name__ == "__main__":`:

```python
def test_reels_settings():
    s = reels_settings(
        {"num_inference_steps": 4, "prompt": "default", "model_type": "x"},
        image="a.png", prompt="p", resolution="480x832", seconds=5, seed=-1,
        loras=["x_high_noise.safetensors", "x_low_noise.safetensors"], high=1, low=0.3, smooth=True,
    )
    assert s["model_type"] == REELS_MODEL
    assert s["num_inference_steps"] == 4
    assert s["prompt"] == "p"
    assert s["image_prompt_type"] == "S" and s["image_start"] == "a.png"
    assert s["resolution"] == "480x832" and s["video_length"] == 81
    assert s["seed"] == -1
    assert s["activated_loras"] == ["x_high_noise.safetensors", "x_low_noise.safetensors"]
    assert s["loras_multipliers"] == "1;0 0;0.3"
    assert s["temporal_upsampling"] == "rife*2"
    assert reels_settings({}, image="a.png", prompt="p", resolution="480x832", seconds=2, seed=7, smooth=False)["temporal_upsampling"] == ""


def test_extend_settings():
    s = extend_settings(
        {"num_inference_steps": 8}, video="v.mp4", anchor="a.png", prompt="p",
        resolution="480x832", seconds=3, seed=5,
    )
    assert s["model_type"] == EXTEND_MODEL
    assert s["num_inference_steps"] == 8
    assert s["image_prompt_type"] == "V" and s["video_source"] == "v.mp4"
    assert s["video_prompt_type"] == "KI" and s["image_refs"] == ["a.png"]
    assert s["image_start"] is None
    assert s["video_length"] == 49 and s["seed"] == 5
    assert s["activated_loras"] == [] and s["loras_multipliers"] == ""
    assert s["temporal_upsampling"] == "rife*2"


def test_motion_settings():
    defaults = {"prompt": "Character appearance: default", "num_inference_steps": 10, "video_prompt_type": "UV", "image_prompt_type": "S"}
    s = motion_settings(defaults, image="c.png", video="d.mp4", description="", resolution="480x832", frames=149, seed=-1)
    assert s["model_type"] == MOTION_MODEL
    assert s["num_inference_steps"] == 10
    assert s["prompt"] == "Character appearance: default"
    assert s["image_prompt_type"] == "T" and s["image_start"] is None
    assert s["video_prompt_type"] == "UVI" and s["image_refs"] == ["c.png"]
    assert s["video_guide"] == "d.mp4" and s["force_fps"] == "control"
    assert s["video_length"] == 149 and s["resolution"] == "480x832"
    assert motion_settings(defaults, image="c.png", video="d.mp4", description=" red hair ", resolution="480x832", frames=65, seed=1)["prompt"] == "red hair"
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ImportError: cannot import name 'reels_settings' from 'simple_app'`

- [ ] **Step 3: Viết phần cài đặt**

Thêm vào cuối `simple_app.py`:

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận đạt**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add simple_app.py test_simple_app.py
git commit -m "Add settings builders for reels, extend and motion control"
```

---

### Task 3: Gợi ý prompt chạy độc lập (cổng quyết định)

Task này xác nhận rủi ro số 1 của spec: gọi prompt enhancer của `wgp` từ ngoài UI Wan2GP. Nếu thất bại, DỪNG và báo người dùng; phương án dự phòng là chuyển app thành plugin tab và cần sửa plan.

**Files:**
- Modify: `simple_app.py` (thêm `IDEAS_INSTRUCTIONS`, `enhance`, `main`)

**Interfaces:**
- Consumes: `REELS_MODEL`, `parse_ideas` (Task 1)
- Produces:
  - `IDEAS_INSTRUCTIONS: str`
  - `enhance(session, model_type: str, text: str, image: str, instructions: str | None = None, progress=None) -> str`. `model_type=""` nghĩa là không dùng chỉ dẫn của model. `image` là đường dẫn file.
  - `main() -> None`, hỗ trợ `python simple_app.py --check-enhancer <ảnh>`

- [ ] **Step 1: Thêm `enhance` và `main`**

Thêm vào cuối `simple_app.py`:

```python
IDEAS_INSTRUCTIONS = (
    "You are a creative director for short vertical social videos. Look at the image and propose "
    "exactly 3 different short video ideas that start from this exact image. Keep the same person, "
    "face, outfit and setting. Write one idea per line, one sentence each, with no numbering and no extra text."
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


def main():
    import os

    check_image = str(Path(sys.argv[2]).resolve()) if len(sys.argv) == 3 and sys.argv[1] == "--check-enhancer" else None
    os.chdir(ROOT)
    from shared.api import init

    session = init(root=ROOT, cli_args=["--profile", "4"])
    if check_image:
        print("enhancer_enabled =", sys.modules["wgp"].server_config.get("enhancer_enabled", 0))
        ideas = parse_ideas(enhance(session, "", "Suggest short video ideas for this image.", check_image, IDEAS_INSTRUCTIONS))
        print("IDEAS:", ideas)
        assert ideas, "no ideas returned"
        detail = enhance(session, REELS_MODEL, ideas[0], check_image)
        print("DETAIL:", detail)
        assert detail.strip(), "empty detailed prompt"
        print("enhancer check ok")
        return
    print("UI not built yet; run with --check-enhancer <image>")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Xác nhận test thuần vẫn đạt**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ok` (import `simple_app` không được kéo theo `wgp` hay `gradio`)

- [ ] **Step 3: Chạy kiểm tra enhancer trên GPU**

Đóng Wan2GP nếu đang mở. Dùng `favicon.png` ở gốc repo làm ảnh thử.

Run: `cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python simple_app.py --check-enhancer favicon.png"`

Expected, theo thứ tự:
- `enhancer_enabled = ` một số khác 0;
- `IDEAS: [...]` với 1 đến 3 chuỗi không rỗng;
- `DETAIL: ` một prompt nhiều dòng (model này dùng dạng `(at 0 seconds: ...)`);
- `enhancer check ok`.

Lần nạp Qwen đầu tiên mất vài phút. Nếu `enhancer_enabled = 0`, có ngoại lệ, hoặc assert thất bại: DỪNG, chép nguyên văn lỗi và báo người dùng. Không tự sửa config Wan2GP và không làm tiếp Task 4.

- [ ] **Step 4: Commit**

```bash
git add simple_app.py
git commit -m "Add headless prompt enhancer call with --check-enhancer"
```

---

### Task 4: Tab Reels (sinh video)

**Files:**
- Modify: `simple_app.py` (thêm `build_ui`, sửa `main`)
- Create: `run_simple_app.bat`

**Interfaces:**
- Consumes: `REELS_MODEL`, `pick_resolution`, `reels_settings` (Task 1, 2); `shared.api.create_gradio_progress_callbacks(progress)`; `session.submit_task`, `session.get_default_settings`, `session.list_loras(model_type)["loras"]`
- Produces (bên trong `build_ui`, các task sau dùng lại):
  - `build_ui(session) -> gr.Blocks`
  - closure `run(settings: dict, progress) -> str | None` (đường dẫn video, `None` nếu bị hủy, `gr.Error` nếu lỗi)
  - closure `cancel() -> None`
  - closure `image_size(path: str) -> tuple[int, int]`
  - `active: dict` với khóa `"job"`
  - component: `r_image`, `r_prompt`, `r_seconds`, `r_quality`, `r_smooth`, `r_loras`, `r_high`, `r_low`, `r_seed`, `r_go`, `r_cancel`, `r_video`, `r_last` (`gr.State`, dict `{"video", "image", "resolution"}`)

- [ ] **Step 1: Thêm `build_ui`**

Chèn vào `simple_app.py`, ngay trước `def main():`:

```python
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
```

- [ ] **Step 2: Cho `main` mở UI**

Trong `main()`, thay dòng cuối:

```python
    print("UI not built yet; run with --check-enhancer <image>")
```

bằng:

```python
    build_ui(session).queue().launch(inbrowser=True)
```

- [ ] **Step 3: Tạo `run_simple_app.bat`**

```bat
@echo off
cd /d "%~dp0"

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" wan2gp
if errorlevel 1 (
    echo [ERROR] Khong activate duoc env.
    pause
    exit /b 1
)

python simple_app.py
pause
```

- [ ] **Step 4: Xác nhận test thuần vẫn đạt**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ok`

- [ ] **Step 5: Chạy tay một video Reels**

Run: `cmd //c run_simple_app.bat` (hoặc bấm đúp file). Trình duyệt mở app.

Trong tab Reels: tải một ảnh chân dung dọc, nhập prompt `she smiles and slowly turns her head, camera static`, để 5 giây, 480p, Mượt x2 bật, không LoRA, bấm **Tạo video**.

Expected:
- thanh tiến độ chạy trên khung video;
- video hiện ra, dài khoảng 5 giây, khung dọc `480x832`, 32 fps (kiểm bằng `ffprobe` trong `ffmpeg_bins` hoặc thuộc tính file trong `outputs/`);
- khuôn mặt ở frame cuối vẫn là người trong ảnh gốc.

Kiểm thêm: bấm **Tạo video** khi chưa có ảnh → thông báo "Chưa có ảnh đầu vào."; bấm **Hủy** giữa chừng → dừng, không có hộp lỗi, video cũ giữ nguyên.

Nếu job báo lỗi về khóa settings, chép nguyên văn lỗi và DỪNG để báo người dùng.

- [ ] **Step 6: Commit**

```bash
git add simple_app.py run_simple_app.bat
git commit -m "Add Reels tab with generation, LoRA phases and cancel"
```

---

### Task 5: Nút gợi ý prompt

**Files:**
- Modify: `simple_app.py` (trong `build_ui`)

**Interfaces:**
- Consumes: `enhance`, `IDEAS_INSTRUCTIONS` (Task 3); `parse_ideas` (Task 1); `active`, `r_image`, `r_prompt` (Task 4)
- Produces: component `r_hint`, `r_ideas_btn`, `r_detail_btn`, `r_ideas`

- [ ] **Step 1: Thêm handler**

Trong `build_ui`, ngay sau hàm `make_reel`, thêm:

```python
    def suggest_ideas(image, hint, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh đầu vào.")
        if active["job"] is not None:
            raise gr.Error("Đang tạo video, chờ xong rồi gợi ý.")
        text = (hint or "").strip() or "Suggest short video ideas for this image."
        ideas = parse_ideas(enhance(session, "", text, image, IDEAS_INSTRUCTIONS, progress))
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
        return enhance(session, REELS_MODEL, hint.strip(), image, None, progress)
```

- [ ] **Step 2: Thêm component**

Trong cột trái của tab Reels, chèn giữa `r_image = ...` và `r_prompt = ...`:

```python
                    r_hint = gr.Textbox(label="Ý ngắn", placeholder="cô ấy quay lại và cười")
                    with gr.Row():
                        r_ideas_btn = gr.Button("3 ý tưởng")
                        r_detail_btn = gr.Button("Viết chi tiết")
                    r_ideas = gr.Radio(choices=[], label="Ý tưởng", visible=False)
```

- [ ] **Step 3: Nối sự kiện**

Ngay sau dòng `r_loras.focus(lora_choices, outputs=r_loras)`, thêm:

```python
        r_ideas_btn.click(suggest_ideas, [r_image, r_hint], r_ideas)
        r_ideas.change(lambda idea: idea or gr.update(), r_ideas, r_hint)
        r_detail_btn.click(write_detail, [r_image, r_hint], r_prompt)
```

- [ ] **Step 4: Xác nhận test thuần vẫn đạt**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ok`

- [ ] **Step 5: Chạy tay**

Run: `cmd //c run_simple_app.bat`

1. Tải ảnh, bấm **3 ý tưởng**. Expected: hiện radio với tối đa 3 ý tưởng liên quan tới ảnh.
2. Chọn một ý. Expected: ô "Ý ngắn" nhận nội dung ý đó.
3. Bấm **Viết chi tiết**. Expected: ô Prompt được điền prompt nhiều dòng theo giây, sửa được.
4. Bấm **Tạo video** với prompt đó. Expected: video sinh bình thường (xác nhận đổi model Qwen → model video không lỗi).
5. Sau khi có video, bấm **3 ý tưởng** lần nữa. Expected: vẫn chạy (xác nhận đổi model video → Qwen).

- [ ] **Step 6: Commit**

```bash
git add simple_app.py
git commit -m "Add idea suggestions and detailed prompt writer to Reels tab"
```

---

### Task 6: Kéo dài video bằng SVI 2 Pro

Lần chạy đầu Wan2GP tự tải hai LoRA SVI 2 Pro (tổng 2.45 GB). Báo người dùng trước khi chạy Step 4.

**Files:**
- Modify: `simple_app.py` (trong `build_ui`)

**Interfaces:**
- Consumes: `EXTEND_MODEL`, `extend_settings` (Task 1, 2); `run`, `r_last`, `r_prompt`, `r_seconds`, `r_smooth`, `r_loras`, `r_high`, `r_low`, `r_seed`, `r_video` (Task 4)
- Produces: component `r_extend`; `make_reel` trả thêm output thứ ba

- [ ] **Step 1: Thêm handler**

Trong `build_ui`, ngay sau hàm `make_reel`, thêm:

```python
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
```

- [ ] **Step 2: Bật nút sau khi có video**

Trong `make_reel`, thay hai dòng return cuối:

```python
        if video is None:
            return gr.update(), gr.update()
        return video, {"video": video, "image": image, "resolution": settings["resolution"]}
```

bằng:

```python
        if video is None:
            return gr.update(), gr.update(), gr.update()
        return video, {"video": video, "image": image, "resolution": settings["resolution"]}, gr.update(interactive=True)
```

- [ ] **Step 3: Thêm nút và nối sự kiện**

Trong cột phải của tab Reels, ngay sau `r_video = gr.Video(...)`, thêm:

```python
                    r_extend = gr.Button("Kéo dài video (neo ảnh gốc)", interactive=False)
```

Thay lời gọi `r_go.click(...)` bằng:

```python
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
```

- [ ] **Step 4: Chạy tay**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py` → `ok`, rồi `cmd //c run_simple_app.bat`.

1. Sinh một video Reels 5 giây (Mượt x2 bật). Expected: nút **Kéo dài video** sáng lên.
2. Sửa prompt cho đoạn tiếp theo, bấm **Kéo dài video**.

Expected:
- video mới dài hơn video đầu (khoảng 10 giây);
- đoạn đầu của video mới phát đúng tốc độ như video gốc, không bị chậm lại hay nhanh lên;
- điểm nối không giật;
- khuôn mặt ở cuối video vẫn là người trong ảnh gốc.

Nếu đoạn đầu phát sai tốc độ: đây là rủi ro số 4 của spec (video nguồn 32 fps). DỪNG và báo người dùng kèm mô tả; cách sửa đã ghi trong spec (giữ bản 16 fps làm nguồn nối) cần một lần sửa plan.
Nếu job báo lỗi về `image_prompt_type` / `video_prompt_type` / `image_refs`: chép nguyên văn lỗi, DỪNG và báo người dùng.

- [ ] **Step 5: Commit**

```bash
git add simple_app.py
git commit -m "Add SVI 2 Pro video extension anchored on the original image"
```

---

### Task 7: Tab Motion Control

Lần sinh đầu tiên Wan2GP tải `animate2_distilled` (17 GB). Hỏi người dùng trước khi chạy Step 4.

**Files:**
- Modify: `simple_app.py` (trong `build_ui`)

**Interfaces:**
- Consumes: `MOTION_MODEL`, `pick_resolution`, `round_frames`, `motion_settings` (Task 1, 2); `run`, `cancel`, `image_size` (Task 4); `session.get_model_availability(model_type)["available"]`
- Produces: tab "Motion control" hoàn chỉnh

- [ ] **Step 1: Thêm handler**

Trong `build_ui`, thêm `import cv2` ngay dưới `import gradio as gr`, rồi thêm sau hàm `make_reel` (và các handler Reels khác):

```python
    def make_motion(image, video, description, limit, quality, seed, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh nhân vật.")
        if not video:
            raise gr.Error("Chưa có video mẫu.")
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 16
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total < 5:
            raise gr.Error("Không đọc được video mẫu.")
        w, h = image_size(image)
        defaults = session.get_default_settings(MOTION_MODEL)
        settings = motion_settings(
            defaults, image=image, video=video, description=description,
            resolution=pick_resolution(w, h, quality),
            frames=round_frames(min(total, limit * fps)), seed=seed,
        )
        out = run(settings, progress)
        return out if out else gr.update()
```

- [ ] **Step 2: Thêm tab**

Trong khối `with gr.Blocks(...) as demo:`, ngay sau toàn bộ khối `with gr.Tab("Reels"):` và trước dòng `r_loras.focus(...)`, thêm:

```python
        with gr.Tab("Motion control"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        m_image = gr.Image(type="filepath", label="Ảnh nhân vật", sources=["upload", "clipboard"])
                        m_video = gr.Video(label="Video mẫu", sources=["upload"])
                    m_desc = gr.Textbox(
                        label="Mô tả nhân vật (không bắt buộc)", lines=2,
                        placeholder=session.get_default_settings(MOTION_MODEL).get("prompt", ""),
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
```

Và sau các dòng sự kiện của Reels, trước `return demo`, thêm:

```python
        m_go.click(make_motion, [m_image, m_video, m_desc, m_limit, m_quality, m_seed], m_out)
        m_cancel.click(cancel, queue=False)
```

- [ ] **Step 3: Xác nhận test thuần và UI mở được**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py` → `ok`
Run: `cmd //c run_simple_app.bat`
Expected: app mở, có hai tab; tab Motion control hiện dòng cảnh báo 17 GB (model chưa có); bấm **Tạo video** khi thiếu ảnh hoặc video → thông báo tương ứng.

- [ ] **Step 4: Chạy tay một video Motion Control**

Hỏi người dùng có đồng ý tải 17 GB không. Nếu đồng ý: tải một ảnh nhân vật rõ mặt và một video mẫu có người (khung hình gần giống ảnh), giới hạn 5 giây, 480p, bấm **Tạo video**.

Expected:
- Wan2GP tải model (thấy trong console), sau đó sinh video;
- video kết quả: nhân vật trong ảnh chuyển động theo video mẫu, fps bằng fps video mẫu, dài tối đa 5 giây;
- khuôn mặt ở frame đầu và frame cuối vẫn là người trong ảnh gốc.

Nếu job báo lỗi về `image_prompt_type` / `video_prompt_type` / `image_refs`: chép nguyên văn lỗi, DỪNG và báo người dùng (rủi ro số 3 của spec).

- [ ] **Step 5: Commit**

```bash
git add simple_app.py
git commit -m "Add Motion Control tab using Animate 2 with character reference"
```

---

### Task 8: Rà soát cuối

**Files:**
- Modify: `docs/superpowers/specs/2026-09-30-simple-app-design.md` (chỉ mục "Rủi ro")

- [ ] **Step 1: Chạy lại toàn bộ kiểm tra**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ok`

Run: `git status --short`
Expected: chỉ còn `?? run_wgp.bat` (file của người dùng, không commit).

- [ ] **Step 2: Đối chiếu tiêu chí trong spec**

Xác nhận từng dòng bằng kết quả đã thấy ở các task trước:
- Reels sinh được video từ ảnh + prompt (Task 4).
- 3 ý tưởng và Viết chi tiết chạy được, trước và sau khi sinh video (Task 5).
- Kéo dài video nối đúng và neo ảnh gốc (Task 6).
- Motion Control sinh được video, nhân vật tham chiếu ảnh gốc (Task 7).
- Không mở UI lớn của Wan2GP trong suốt quá trình.

- [ ] **Step 3: Ghi kết quả rủi ro vào spec**

Trong mục "Rủi ro và phương án dự phòng" của spec, thêm vào cuối mỗi rủi ro 1, 3, 4 một câu kết quả thực tế, ví dụ: `Kết quả 2026-09-30: chạy được ở chế độ độc lập.` Chỉ ghi điều đã quan sát; rủi ro nào chưa kiểm được thì ghi rõ là chưa kiểm.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-09-30-simple-app-design.md
git commit -m "Record verification results in simple app spec"
```

- [ ] **Step 5: Báo người dùng**

Báo: những gì đã chạy được, những gì chưa kiểm, mặt có còn trôi không (để quyết định giai đoạn 2: H3 Face Refiner, 21 GB), và nhắc ba ứng viên xóa trong spec do người dùng tự quyết.
