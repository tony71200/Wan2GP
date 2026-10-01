# Simple App v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nâng `simple_app.py` lên ba tab (Reels 2–15 s, Reels Multi Image, Motion Control đúng thời lượng video mẫu) theo mockup người dùng, có ước tính thời gian tự hiệu chỉnh và nút Upscale ×2.

**Architecture:** Vẫn một file `simple_app.py`: phần đầu là hằng số và hàm thuần (không import gradio/wgp, test được không cần GPU); `build_ui(session)` dựng Gradio Blocks với handler dạng closure; `run_check()` là harness không UI. Mọi việc sinh video đi qua `shared.api` của Wan2GP (`submit_task`, `submit_media_postprocessing`); sliding window, `[/duration]` và ảnh cuối theo cửa sổ đều là tính năng có sẵn của Wan2GP, app chỉ dựng settings.

**Tech Stack:** Python 3.11 (env conda `wan2gp`), Gradio 5.29, `shared.api` của Wan2GP, Pillow, OpenCV, stdlib `json`/`tempfile`/`time`.

**Spec:** `docs/superpowers/specs/2026-10-01-simple-app-v2-design.md` (nối tiếp `docs/superpowers/specs/2026-09-30-simple-app-design.md`)

## Global Constraints

- Nhánh: `20260930_simple_app`. Không commit `run_wgp.bat` (file chưa track của người dùng).
- Không sửa `wgp.py` hay file nào dưới `shared/`, `models/`, `plugins/`, `defaults/`, `profiles/`. Không thêm thư viện.
- Không tải model mới. Upscale chỉ dùng `lanczos*2`.
- Tên model chính xác: `i2v_2_2_Enhanced_Lightning_v2` (Reels ≤ 5 s, Multi), `i2v_2_2_Enhanced_Lightning_v2_svi2pro` (Reels > 5 s, Kéo dài), `animate2_distilled`, `animate` (Wan 2.2 Animate 14B).
- LoRA FusionX: `Wan2.1_I2V_14B_FusionX_LoRA.safetensors` (đã có trong `loras/wan_i2v`).
- Trong chế độ `multi_prompts_gen_type="PW"`, không có dòng trống bên trong một đoạn; mỗi đoạn mở đầu `[/duration=Ns]`.
- Nhãn UI bằng tiếng Việt như spec.
- Mỗi lúc một tiến trình dùng GPU: dừng app đang nghe cổng 7860 và Wan2GP trước khi chạy harness.
- Test thuần (Git Bash): `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py` → in `ok`.
- Lệnh có GPU (Git Bash, chạy nền, ghi log): `cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python simple_app.py <tham số>" > <log> 2>&1`. Đọc log bằng `tr '\r' '\n' < <log> | tail`.
- Mỗi commit kết thúc bằng dòng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Structure

| File | Trách nhiệm |
|---|---|
| `simple_app.py` (sửa) | Hằng số, hàm thuần mới (kế hoạch Reels, tách prompt, settings Multi/Motion, ước tính), `build_ui()` viết lại theo mockup, `run_check()` có assert thời lượng |
| `test_simple_app.py` (sửa) | Assert cho mọi hàm thuần mới; cập nhật test Motion (`force_fps` = `"16"`) |
| `docs/superpowers/specs/2026-10-01-simple-app-v2-design.md` (sửa ở Task 5, 6, 7) | Chỉnh hai chi tiết UI/harness, ghi kết quả đo |

`.gitignore` đã có `simple_app_timings.json` (commit `8dc16eb5`).

---

### Task 1: Kế hoạch Reels 2–15 s và tách prompt theo cửa sổ

**Files:**
- Modify: `simple_app.py` (hằng số đầu file; `reels_settings`; thay hai hằng `IDEAS_INSTRUCTIONS`, `DETAIL_INSTRUCTIONS` bằng hàm; chỗ dùng chúng trong `build_ui` và `run_check`)
- Test: `test_simple_app.py`

**Interfaces:**
- Produces:
  - `FPS = 16`
  - `reel_plan(seconds: int) -> tuple[str, list[int]]`: model và thời lượng nguyên giây của từng cửa sổ
  - `reel_windows(seconds: int) -> tuple[str, list[int]]`: model và số frame từng cửa sổ (dùng cho ước tính)
  - `split_timeline(prompt: str, durations: list[int]) -> str`
  - `reels_settings(defaults, *, image, prompt, resolution, seconds, seed, loras=(), high=1.0, low=0.0, smooth=True) -> dict` (giữ chữ ký v1; model tự chọn theo `reel_plan`)
  - `ideas_instructions(seconds: int = 5) -> str`, `detail_instructions(seconds: int = 5) -> str`

- [ ] **Step 1: Viết test trước**

Trong `test_simple_app.py`, sửa khối import đầu file thành:

```python
from simple_app import (
    REELS_MODEL, EXTEND_MODEL, MOTION_MODEL, MOTION_DEFAULT_PROMPT,
    pick_resolution, frames_for, round_frames, pair_loras, parse_ideas,
    reels_settings, extend_settings, motion_settings,
    reel_plan, reel_windows, split_timeline, ideas_instructions, detail_instructions,
)
```

Thêm các test sau (trước khối `if __name__ == "__main__":`):

```python
def test_reel_plan():
    assert reel_plan(2) == (REELS_MODEL, [2])
    assert reel_plan(5) == (REELS_MODEL, [5])
    assert reel_plan(6) == (EXTEND_MODEL, [3, 3])
    assert reel_plan(7) == (EXTEND_MODEL, [4, 3])
    assert reel_plan(11) == (EXTEND_MODEL, [4, 4, 3])
    assert reel_plan(15) == (EXTEND_MODEL, [5, 5, 5])


def test_reel_windows():
    assert reel_windows(3) == (REELS_MODEL, [49])
    assert reel_windows(5) == (REELS_MODEL, [81])
    assert reel_windows(15) == (EXTEND_MODEL, [81, 81, 81])


def test_split_timeline():
    prompt = (
        "A young man in a white shirt, in a park.\n"
        "(at 0 seconds: he stands)\n\n"
        "(at 4 seconds: he waves)\n"
        "(at 5 seconds: he smiles)\n"
        "(3-7 seconds: camera pushes in)"
    )
    first, second = split_timeline(prompt, [5, 5]).split("\n\n")
    assert first == (
        "[/duration=5s] A young man in a white shirt, in a park.\n"
        "(at 0 seconds: he stands)\n(at 4 seconds: he waves)\n(3-5 seconds: camera pushes in)"
    )
    assert second == (
        "[/duration=5s] A young man in a white shirt, in a park.\n"
        "(at 0 seconds: he smiles)\n(0-2 seconds: camera pushes in)"
    )
    # other time formats, Vietnamese unit, no parentheses
    assert split_timeline("(from 0 to 2 seconds: walks)\n(at 2s: stops)", [2, 1]) == (
        "[/duration=2s] (0-2 seconds: walks)\n\n[/duration=1s] (at 0 seconds: stops)"
    )
    assert split_timeline("0-2 giây: đi tới", [2]) == "[/duration=2s] (0-2 seconds: đi tới)"
    assert split_timeline("(at 0–2 seconds: turns)", [2]) == "[/duration=2s] (0-2 seconds: turns)"
    # no timed line: every window gets the whole prompt, blank lines removed
    assert split_timeline("he walks\n\nslowly", [3, 3]) == (
        "[/duration=3s] he walks\nslowly\n\n[/duration=3s] he walks\nslowly"
    )
    # a window with no timed line carries on the latest earlier action
    assert split_timeline("(at 0 seconds: x)\n(at 6 seconds: y)", [4, 4, 3]) == (
        "[/duration=4s] (at 0 seconds: x)\n\n[/duration=4s] (at 2 seconds: y)\n\n[/duration=3s] (at 0 seconds: y)"
    )


def test_reels_settings_long():
    s = reels_settings(
        {"num_inference_steps": 8}, image="a.png", prompt="(at 0 seconds: x)\n(at 6 seconds: y)",
        resolution="480x832", seconds=11, seed=3,
    )
    assert s["model_type"] == EXTEND_MODEL and s["num_inference_steps"] == 8
    assert s["multi_prompts_gen_type"] == "PW"
    assert s["image_prompt_type"] == "S" and s["image_start"] == "a.png"
    assert s["video_prompt_type"] == "KI" and s["image_refs"] == ["a.png"]
    assert s["video_length"] == 177
    assert s["prompt"] == split_timeline("(at 0 seconds: x)\n(at 6 seconds: y)", [4, 4, 3])
    assert s["seed"] == 3 and s["temporal_upsampling"] == "rife*2"


def test_instructions():
    assert "one continuous 15 second shot" in ideas_instructions(15)
    assert "from 0 to 14 seconds" in detail_instructions(15)
    assert "from 0 to 4 seconds" in detail_instructions()
```

- [ ] **Step 2: Chạy test, xác nhận lỗi**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ImportError: cannot import name 'reel_plan' from 'simple_app'`

- [ ] **Step 3: Cài đặt**

Trong `simple_app.py`, ngay sau dòng `MOTION_MODEL = "animate2_distilled"` thêm:

```python
FPS = 16  # Wan 2.2 i2v output rate; Motion Control resamples the driving video to it
```

Ngay sau hàm `frames_for` thêm:

```python
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
```

Thay toàn bộ hàm `reels_settings` bằng:

```python
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
```

Thay hai hằng `IDEAS_INSTRUCTIONS`, `DETAIL_INSTRUCTIONS` và hai dòng comment phía trên `DETAIL_INSTRUCTIONS` bằng (comment được giữ trong khối dưới):

```python
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
```

Sửa chỗ dùng cũ để file vẫn chạy (Task 5 sẽ viết lại `build_ui` nên đây chỉ là giữ cho đúng):
- Trong `build_ui`, `suggest_ideas`: `IDEAS_INSTRUCTIONS` → `ideas_instructions()`; `write_detail`: `DETAIL_INSTRUCTIONS` → `detail_instructions()`.
- Trong `run_check`: `IDEAS_INSTRUCTIONS` → `ideas_instructions()`; `DETAIL_INSTRUCTIONS` → `detail_instructions()`.

- [ ] **Step 4: Chạy test, xác nhận qua**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ok`

Run: `grep -n "IDEAS_INSTRUCTIONS\|DETAIL_INSTRUCTIONS" simple_app.py`
Expected: không có dòng nào.

- [ ] **Step 5: Commit**

```bash
git add simple_app.py test_simple_app.py
git commit -m "Plan reels up to 15 s as one SVI 2 Pro job with per-window prompts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Settings cho Reels Multi Image

**Files:**
- Modify: `simple_app.py` (thêm sau `extend_settings`)
- Test: `test_simple_app.py`

**Interfaces:**
- Consumes: `REELS_MODEL`, `FPS`, `_lora_settings(loras, high, low, smooth) -> dict` (có sẵn từ v1)
- Produces:
  - `MAX_IMAGES = 6`
  - `multi_settings(defaults, *, images: list[str], durations: list[int], prompts: list[str], common: str, resolution: str, seed, loras=(), high=1.0, low=0.0, smooth=True) -> dict`
  - `aspect_mismatch(sizes: list[tuple[int, int] | None], tolerance: float = 0.1) -> list[int]`: số thứ tự (từ 1) các ảnh lệch tỉ lệ so với ảnh 1

- [ ] **Step 1: Viết test trước**

Thêm `multi_settings, aspect_mismatch, MAX_IMAGES` vào khối import của `test_simple_app.py`, rồi thêm:

```python
def test_multi_settings():
    s = multi_settings(
        {"num_inference_steps": 4}, images=["a.png", "b.png", "c.png"], durations=[3, 2, 4],
        prompts=["walks in", "", "sits.\n\ndown"], common="A man in a red coat.",
        resolution="480x832", seed=1, loras=["x_high_noise.safetensors"], high=1, low=0, smooth=False,
    )
    assert s["model_type"] == REELS_MODEL and s["num_inference_steps"] == 4
    assert s["image_prompt_type"] == "SE"
    assert s["image_start"] == "a.png" and s["image_end"] == ["b.png", "c.png"]
    assert s["multi_prompts_gen_type"] == "PW"
    assert s["prompt"] == (
        "[/duration=3s] A man in a red coat. walks in\n\n"
        "[/duration=2s] A man in a red coat\n\n"
        "[/duration=4s] A man in a red coat. sits. down"
    )
    assert s["video_length"] == 145 and s["resolution"] == "480x832" and s["seed"] == 1
    assert s["loras_multipliers"] == "1;0" and s["temporal_upsampling"] == ""
    one = multi_settings({}, images=["a.png"], durations=[2], prompts=["waves"], common="", resolution="480x832", seed=-1)
    assert one["image_prompt_type"] == "S" and one["image_end"] is None
    assert one["prompt"] == "[/duration=2s] waves" and one["video_length"] == 33
    assert MAX_IMAGES == 6


def test_aspect_mismatch():
    assert aspect_mismatch([(1000, 1250), (800, 1000), (1000, 1000), None, (1080, 1920)]) == [3, 5]
    assert aspect_mismatch([None, (1000, 1000)]) == []
    assert aspect_mismatch([]) == []
```

- [ ] **Step 2: Chạy test, xác nhận lỗi**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ImportError: cannot import name 'multi_settings' from 'simple_app'`

- [ ] **Step 3: Cài đặt**

Thêm vào `simple_app.py` ngay sau hàm `extend_settings`:

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận qua**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add simple_app.py test_simple_app.py
git commit -m "Add multi-image keyframe settings for one Lightning job

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Motion Control đúng thời lượng, hai model

**Files:**
- Modify: `simple_app.py` (hằng số; thêm `motion_frames`, `sliding_windows`; thay `motion_settings`)
- Test: `test_simple_app.py` (thêm test; sửa `test_motion_settings`)

**Interfaces:**
- Consumes: `FPS`, `MOTION_MODEL`, `MOTION_DEFAULT_PROMPT`
- Produces:
  - `ANIMATE_MODEL = "animate"`, `FUSIONX_LORA = "Wan2.1_I2V_14B_FusionX_LoRA.safetensors"`
  - `MOTION_MODELS: dict[str, str]` = `{"Animate 2 distilled": MOTION_MODEL, "Wan 2.2 Animate 14B": ANIMATE_MODEL}` (nhãn dropdown → model_type)
  - `MOTION_WINDOWS: dict[str, tuple[int, int]]` = `{MOTION_MODEL: (65, 1), ANIMATE_MODEL: (81, 5)}` (cỡ cửa sổ, chồng lấn)
  - `motion_frames(src_fps: float, src_frames: int, limit: float | None = None) -> int`
  - `sliding_windows(frames: int, size: int, overlap: int) -> list[int]`
  - `motion_settings(defaults, *, image, video, description, resolution, frames, seed, model=MOTION_MODEL, smooth=False) -> dict`

- [ ] **Step 1: Viết test trước**

Thêm `ANIMATE_MODEL, FUSIONX_LORA, MOTION_MODELS, MOTION_WINDOWS, motion_frames, sliding_windows` vào khối import. Thay toàn bộ `test_motion_settings` bằng:

```python
def test_motion_settings():
    defaults = {"prompt": "Character appearance: default", "num_inference_steps": 10, "video_prompt_type": "UV", "image_prompt_type": "S"}
    s = motion_settings(defaults, image="c.png", video="d.mp4", description="", resolution="480x832", frames=149, seed=-1)
    assert s["model_type"] == MOTION_MODEL
    assert s["num_inference_steps"] == 10
    assert s["prompt"] == MOTION_DEFAULT_PROMPT
    assert s["image_prompt_type"] == "T" and s["image_start"] is None
    assert s["video_prompt_type"] == "UVI" and s["image_refs"] == ["c.png"]
    assert s["video_guide"] == "d.mp4" and s["force_fps"] == "16"
    assert s["video_length"] == 149 and s["resolution"] == "480x832"
    assert s["temporal_upsampling"] == ""
    assert motion_settings(defaults, image="c.png", video="d.mp4", description=" red hair ", resolution="480x832", frames=65, seed=1)["prompt"] == "red hair"

    a = motion_settings(
        {"num_inference_steps": 30, "guidance_scale": 5}, image="c.png", video="d.mp4", description="",
        resolution="480x832", frames=433, seed=2, model=ANIMATE_MODEL, smooth=True,
    )
    assert a["model_type"] == "animate"
    assert a["image_prompt_type"] == "" and a["video_prompt_type"] == "PVBKI"
    assert a["image_refs"] == ["c.png"] and a["video_guide"] == "d.mp4"
    assert a["activated_loras"] == [FUSIONX_LORA] and a["loras_multipliers"] == "1"
    assert a["num_inference_steps"] == 10 and a["guidance_phases"] == 1 and a["guidance_scale"] == 1
    assert a["sliding_window_size"] == 81 and a["force_fps"] == "16"
    assert a["video_length"] == 433 and a["temporal_upsampling"] == "rife*2"
    assert MOTION_MODELS == {"Animate 2 distilled": MOTION_MODEL, "Wan 2.2 Animate 14B": ANIMATE_MODEL}


def test_motion_frames():
    assert motion_frames(30, 813) == 433        # 27.1 s at 30 fps -> 27.06 s at 16 fps
    assert motion_frames(30, 813, 5) == 81
    assert motion_frames(30, 813, 40) == 433    # a limit above the video length changes nothing
    assert motion_frames(30, 60) == 33
    assert motion_frames(30, 3) == 5


def test_sliding_windows():
    assert sliding_windows(433, *MOTION_WINDOWS[MOTION_MODEL]) == [65] * 6 + [49]
    assert sliding_windows(433, *MOTION_WINDOWS[ANIMATE_MODEL]) == [81] * 5 + [53]
    assert sliding_windows(49, 65, 1) == [49]
```

- [ ] **Step 2: Chạy test, xác nhận lỗi**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ImportError: cannot import name 'ANIMATE_MODEL' from 'simple_app'`

- [ ] **Step 3: Cài đặt**

Ngay sau dòng `FPS = 16 ...` thêm:

```python
ANIMATE_MODEL = "animate"  # Wan 2.2 Animate 14B
FUSIONX_LORA = "Wan2.1_I2V_14B_FusionX_LoRA.safetensors"  # loras/wan_i2v, 10-step accelerator
MOTION_MODELS = {"Animate 2 distilled": MOTION_MODEL, "Wan 2.2 Animate 14B": ANIMATE_MODEL}
MOTION_WINDOWS = {MOTION_MODEL: (65, 1), ANIMATE_MODEL: (81, 5)}  # (window size, overlap)
```

Ngay sau hàm `round_frames` thêm:

```python
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
```

Thay toàn bộ hàm `motion_settings` bằng:

```python
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
```

Trong `build_ui` (`make_motion`) và `run_check` (`--check-motion`), lời gọi `motion_settings(...)` cũ vẫn chạy (tham số mới có mặc định). Không sửa gì thêm ở bước này.

- [ ] **Step 4: Chạy test, xác nhận qua**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add simple_app.py test_simple_app.py
git commit -m "Run Motion Control at 16 fps for the full driving video, add Wan 2.2 Animate 14B

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Ước tính thời gian và hiệu chỉnh

**Files:**
- Modify: `simple_app.py` (thêm `import json` ở đầu file; thêm khối ước tính sau `aspect_mismatch`)
- Test: `test_simple_app.py`

**Interfaces:**
- Consumes: `REELS_MODEL`, `EXTEND_MODEL`, `MOTION_MODEL`, `ANIMATE_MODEL`, `ROOT`
- Produces:
  - `SPEED: dict[str, tuple[int, float, int, int]]` (bước, giây/bước, frame mốc, điểm ảnh mốc), `LOAD_SECONDS = 120`
  - `MODEL_LABELS: dict[str, str]`
  - `TIMINGS: Path` = `ROOT / "simple_app_timings.json"`
  - `estimate_seconds(model: str, windows: list[int], resolution: str, calib: dict | None = None) -> float`
  - `fmt_minutes(seconds: float) -> str`
  - `eta_text(model: str, windows: list[int], resolution: str, calib: dict | None = None) -> str`
  - `load_calib(path: Path = TIMINGS) -> dict[str, float]`
  - `save_calib(model: str, actual: float, raw_estimate: float, path: Path = TIMINGS) -> dict[str, float]`

- [ ] **Step 1: Viết test trước**

Thêm `ANIMATE_MODEL` (nếu chưa có), `estimate_seconds, fmt_minutes, eta_text, load_calib, save_calib` vào khối import, rồi thêm:

```python
def test_estimate_seconds():
    assert estimate_seconds(REELS_MODEL, [81], "720x720") == 120 + 4 * 97
    assert estimate_seconds(REELS_MODEL, [81], "720x720", {REELS_MODEL: 2}) == 2 * (120 + 4 * 97)
    assert estimate_seconds(REELS_MODEL, [81], "720x720", {EXTEND_MODEL: 2}) == 120 + 4 * 97
    assert estimate_seconds(EXTEND_MODEL, [81, 81], "720x720") == 120 + 2 * 8 * 106
    assert estimate_seconds(MOTION_MODEL, [65], "480x848") == 120 + 10 * 94
    assert estimate_seconds(ANIMATE_MODEL, [65], "480x848") == 120 + 10 * 94
    assert estimate_seconds(REELS_MODEL, [81], "960x960") > estimate_seconds(REELS_MODEL, [81], "720x720")


def test_fmt_minutes():
    assert fmt_minutes(419) == "7 phút"
    assert fmt_minutes(10) == "1 phút"
    assert fmt_minutes(6266) == "1 giờ 44 phút"


def test_eta_text():
    assert eta_text(EXTEND_MODEL, [81, 81, 81], "480x832") == "⏱ ≈ 35 phút (SVI 2 Pro · 3 cửa sổ × 8 bước)"
    assert eta_text(REELS_MODEL, [81], "480x832") == "⏱ ≈ 7 phút (Lightning v2 · 1 cửa sổ × 4 bước)"


def test_calib_roundtrip():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "timings.json"
        assert load_calib(p) == {}
        assert save_calib(REELS_MODEL, 1000, 500, p) == {REELS_MODEL: 2.0}
        assert save_calib(MOTION_MODEL, 300, 400, p) == {REELS_MODEL: 2.0, MOTION_MODEL: 0.75}
        assert load_calib(p) == {REELS_MODEL: 2.0, MOTION_MODEL: 0.75}
        p.write_text("not json")
        assert load_calib(p) == {}
```

- [ ] **Step 2: Chạy test, xác nhận lỗi**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ImportError: cannot import name 'estimate_seconds' from 'simple_app'`

- [ ] **Step 3: Cài đặt**

Đầu file, sửa khối import thành:

```python
import json
import re
import sys
from pathlib import Path
```

Thêm sau hàm `aspect_mismatch`:

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận qua**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add simple_app.py test_simple_app.py
git commit -m "Add self-calibrating time estimate for every model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Giao diện ba tab theo mockup

**Files:**
- Modify: `simple_app.py` (thêm `CSS`, `FORCE_LIGHT`, `GUIDE` trước `build_ui`; thay toàn bộ hàm `build_ui`)
- Modify: `docs/superpowers/specs/2026-10-01-simple-app-v2-design.md` (thẻ thông tin video mẫu không có độ phân giải)

**Interfaces:**
- Consumes: mọi hàm của Task 1–4; từ v1: `pick_resolution`, `RESOLUTIONS`, `frames_for`, `parse_ideas`, `extend_settings`, `enhance`, `_image_size`, `_video_info`; `shared.api.create_gradio_progress_callbacks`; `session.submit_task`, `session.submit_media_postprocessing`, `session.get_default_settings`, `session.list_loras`, `session.get_model_availability`
- Produces: `build_ui(session) -> gr.Blocks` (chữ ký không đổi; `main()` không đổi)

Gradio 5.29 chỉ chèn `gr.Progress` cho tham số **đứng trước** `*args`, nên `make_multi` đặt `progress` trước `*vals`.

- [ ] **Step 1: Thêm CSS, FORCE_LIGHT, GUIDE**

Thêm ngay trước `def build_ui(session):`:

```python
CSS = """
.gradio-container { background: #F9BE7E !important; }
#header { background: #F9A710; padding: 8px 16px; border-radius: 10px; align-items: center; }
#header h2 { margin: 0; font-family: monospace; }
#guide { position: fixed; inset: 5vh 10vw; z-index: 1000; background: white; border-radius: 12px;
         box-shadow: 0 10px 40px rgba(0, 0, 0, .35); padding: 20px; overflow: auto; }
"""
# The orange palette is designed for light mode; Gradio otherwise follows the OS theme.
FORCE_LIGHT = "() => document.body.classList.remove('dark')"

GUIDE = """
## Hướng dẫn sử dụng

### Reels: một ảnh → video 2–15 giây
- **Từ 5 giây trở xuống**: Wan 2.2 i2v *Enhanced Lightning v2*, 4 bước, một cửa sổ.
- **Trên 5 giây**: *SVI 2 Pro* (cùng model, thêm LoRA SVI), 8 bước mỗi cửa sổ, chia thành các cửa sổ tối đa 5 giây trong **một job**. Mọi cửa sổ neo về ảnh gốc nên nhân vật không trôi.
- Prompt có mốc giây như `(at 3 seconds: …)` hoặc `(0-2 seconds: …)` được tự tách cho từng cửa sổ. Dòng không có mốc (mô tả nhân vật) được giữ ở mọi cửa sổ.
- **3 ý tưởng / Viết chi tiết**: Qwen3.5-4B (có thị giác) đọc ảnh rồi viết prompt theo từng giây của thời lượng đang chọn.
- **Kéo dài video**: nối thêm một đoạn bằng SVI 2 Pro, neo về ảnh gốc.

### Reels Multi Image: tối đa 6 ảnh của cùng một nhân vật
Mỗi ảnh là một mốc. Đoạn của ảnh *i* bắt đầu ở ảnh *i*, dài đúng số giây ghi trên ảnh đó và kết thúc ở ảnh *i+1*. Đoạn của ảnh cuối chạy tự do. Mô tả nhân vật chung được ghép vào đầu prompt của mọi đoạn. Độ phân giải theo ảnh 1.

### Motion Control
Nhân vật trong ảnh làm theo chuyển động của video mẫu. Video ra dài **đúng bằng video mẫu**, trừ khi tick *Giới hạn thời lượng*. Video mẫu được đưa về 16 fps để chạy nhanh gấp đôi; bật *Mượt x2* để ra 32 fps.
- **Animate 2 distilled**: 10 bước, cửa sổ 65 frame.
- **Wan 2.2 Animate 14B**: thêm LoRA FusionX, 10 bước, cửa sổ 81 frame.

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
```

- [ ] **Step 2: Thay toàn bộ `build_ui`**

```python
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
        return video, f"Thực tế: {fmt_minutes(took)}"

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
            cancel_btn = gr.Button("Hủy")
            go = gr.Button("Tạo Video", variant="primary")
        eta = gr.Markdown()
        video = gr.Video(label="Video ra", autoplay=True)
        took = gr.Markdown()
        with gr.Row():
            ext = gr.Button("Kéo dài video (neo ảnh gốc)", interactive=False) if extend else None
            up = gr.Button("Upscale ×2", interactive=False)
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

    def make_reel(image, prompt, seconds, quality, smooth, loras, high, low, seed, progress=gr.Progress()):
        if not image:
            raise gr.Error("Chưa có ảnh đầu vào.")
        if not (prompt or "").strip():
            raise gr.Error("Chưa có prompt.")
        model, windows = reel_windows(seconds)
        settings = reels_settings(
            session.get_default_settings(model),
            image=image, prompt=prompt, resolution=image_resolution(image, quality),
            seconds=seconds, seed=seed, loras=loras, high=high, low=low, smooth=smooth,
        )
        video, took = generate(settings, windows, progress)
        if video is None:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        last = {"video": video, "image": image, "resolution": settings["resolution"]}
        return video, last, gr.update(interactive=True), gr.update(interactive=True), took

    def extend_reel(last, prompt, seconds, smooth, loras, high, low, seed, progress=gr.Progress()):
        if not last:
            raise gr.Error("Chưa có video để kéo dài.")
        if not (prompt or "").strip():
            raise gr.Error("Chưa có prompt.")
        settings = extend_settings(
            session.get_default_settings(EXTEND_MODEL),
            video=last["video"], anchor=last["image"], prompt=prompt, resolution=last["resolution"],
            seconds=min(seconds, 5), seed=seed, loras=loras, high=high, low=low, smooth=smooth,
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
            eta += f"\n\n⚠ Ảnh {', '.join(map(str, off))} lệch tỉ lệ khung với ảnh 1 hơn 10%: Wan2GP sẽ co và cắt."
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
    def make_multi(n, common, quality, smooth, loras, high, low, seed, progress=gr.Progress(), *vals):
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
            resolution=image_resolution(images[0], quality), seed=seed, loras=loras, high=high, low=low, smooth=smooth,
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

    def motion_eta(image, video, limit_on, limit, quality, label):
        if not video:
            return "⏱ Chọn video mẫu để ước tính."
        model = MOTION_MODELS[label]
        fps, total = _video_info(video)
        frames = motion_frames(fps, total, limit if limit_on else None)
        windows = sliding_windows(frames, *MOTION_WINDOWS[model])
        return eta_text(model, windows, image_resolution(image, quality), load_calib())

    def make_motion(image, video, description, limit_on, limit, quality, smooth, label, seed, progress=gr.Progress()):
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
        )
        out, took = generate(settings, sliding_windows(frames, *MOTION_WINDOWS[model]), progress)
        if out is None:
            return gr.update(), gr.update(), gr.update()
        return out, gr.update(interactive=True), took

    with gr.Blocks(title="Simple App", theme=gr.themes.Soft(primary_hue="orange"), css=CSS, js=FORCE_LIGHT) as demo:
        r_last = gr.State(None)
        with gr.Row(elem_id="header"):
            gr.Markdown("## Simple WebApp: Reels and Motion Control")
            guide_btn = gr.Button("Hướng dẫn", size="sm", scale=0)
        with gr.Column(visible=False, elem_id="guide") as guide:
            gr.Markdown(GUIDE)
            close_btn = gr.Button("Đóng")

        with gr.Tab("Reels"):
            with gr.Row():
                with gr.Column(scale=1):
                    r_image = gr.Image(type="filepath", label="Ảnh đầu vào", sources=["upload", "clipboard"])
                    r_info = gr.Markdown()
                    r_hint = gr.Textbox(label="Ý ngắn", placeholder="chàng trai thong thả đi đến gần")
                    with gr.Row():
                        r_ideas_btn = gr.Button("3 ý tưởng", scale=0)
                        r_ideas = gr.Dropdown(choices=[], label="Ý tưởng", scale=2)
                        r_detail_btn = gr.Button("Viết chi tiết", variant="primary", scale=0)
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
                                mm_removes.append(gr.Button("Xóa", size="sm"))
                            mm_cards.append(card)
                    mm_add = gr.Button("+ Thêm ảnh")
                    with gr.Accordion("Cài đặt nâng cao", open=False):
                        mm_quality = gr.Radio(["480p", "720p"], value="480p", label="Độ phân giải (theo ảnh 1)")
                        mm_smooth = gr.Checkbox(value=True, label="Mượt x2 (32 fps)")
                        mm_loras = gr.Dropdown(choices=lora_names(), multiselect=True, label="LoRA (loras/wan)")
                        with gr.Row():
                            mm_high = gr.Slider(0, 2, value=1, step=0.1, label="Pha đầu (chuyển động)")
                            mm_low = gr.Slider(0, 2, value=0, step=0.1, label="Pha cuối (chi tiết mặt)")
                        mm_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = ngẫu nhiên)")
                with gr.Column(scale=1):
                    mm_go, mm_eta, mm_video, mm_took, _, mm_up = results()

        with gr.Tab("Motion Control"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        m_image = gr.Image(type="filepath", label="Ảnh nhân vật", sources=["upload", "clipboard"])
                        m_video = gr.Video(label="Video tham chiếu", sources=["upload"])
                    m_info = gr.Markdown()
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
                        m_model = gr.Dropdown(list(MOTION_MODELS), value="Animate 2 distilled", label="Chọn Model")
                        m_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = ngẫu nhiên)")
                    if not session.get_model_availability(MOTION_MODEL)["available"]:
                        gr.Markdown("Lần đầu dùng Animate 2 sẽ tải model (17 GB).")
                with gr.Column(scale=1):
                    m_go, m_eta, m_out, m_took, _, m_up = results()

        guide_btn.click(lambda: gr.update(visible=True), outputs=guide, queue=False)
        close_btn.click(lambda: gr.update(visible=False), outputs=guide, queue=False)

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
            [r_image, r_prompt, r_seconds, r_quality, r_smooth, r_loras, r_high, r_low, r_seed],
            [r_video, r_last, r_extend, r_up, r_took],
        )
        r_extend.click(
            extend_reel,
            [r_last, r_prompt, r_seconds, r_smooth, r_loras, r_high, r_low, r_seed],
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
            [mm_n, mm_common, mm_quality, mm_smooth, mm_loras, mm_high, mm_low, mm_seed, *slots],
            [mm_video, mm_up, mm_took],
        )

        m_limit_on.change(lambda on: gr.update(interactive=on), m_limit_on, m_limit)
        m_video.change(motion_info, m_video, [m_info, m_limit])
        motion_in = [m_image, m_video, m_limit_on, m_limit, m_quality, m_model]
        for comp in motion_in:
            comp.change(motion_eta, motion_in, m_eta)
        demo.load(motion_eta, motion_in, m_eta)
        m_go.click(
            make_motion,
            [m_image, m_video, m_desc, m_limit_on, m_limit, m_quality, m_smooth, m_model, m_seed],
            [m_out, m_up, m_took],
        )
    return demo
```

- [ ] **Step 3: Sửa spec cho khớp thẻ video mẫu**

Trong `docs/superpowers/specs/2026-10-01-simple-app-v2-design.md`, mục "Tab Motion Control › Giao diện", thay `thẻ chỉ đọc \`27.1 giây · 30 fps · 720×1280 → 433 frame ở 16 fps\`` bằng `thẻ chỉ đọc \`27.1 giây · 30 fps → 433 frame ở 16 fps\``.

- [ ] **Step 4: Test thuần vẫn qua, file biên dịch được**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py && "$USERPROFILE/anaconda3/envs/wan2gp/python.exe" -m py_compile simple_app.py && echo compiled`
Expected: `ok` rồi `compiled`

- [ ] **Step 5: Chạy app và kiểm tra giao diện**

Dừng tiến trình đang nghe cổng 7860 nếu có:

```bash
for p in $(netstat -ano | grep ':7860 ' | grep LISTENING | awk '{print $5}' | sort -u); do taskkill //PID $p //F; done
```

Chạy nền (Git Bash, `run_in_background`):

```bash
cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python simple_app.py" > "$TEMP/simple_app_ui.log" 2>&1
```

Khi log có `Running on local URL`, mở `http://127.0.0.1:7860` trong browser pane và kiểm tra (đọc bằng `read_page` / `find`, chụp màn hình khi cần):
1. Có ba tab `Reels`, `Reels Multi Image`, `Motion Control`; header màu cam, nền sáng.
2. Bấm `Hướng dẫn`: dialog hiện có mục "Pha đầu và Pha cuối" và "LoRA tải thêm"; bấm `Đóng` thì ẩn.
3. Reels: dòng ước tính hiện `⏱ ≈ 7 phút (Lightning v2 · 1 cửa sổ × 4 bước)` (hoặc số khác nếu đã có `simple_app_timings.json`); kéo thời lượng lên 15 thì đổi thành `SVI 2 Pro · 3 cửa sổ × 8 bước`.
4. Multi: hiện `2/6 ảnh · tổng 6 giây`; bấm `+ Thêm ảnh` → `3/6 ảnh · tổng 9 giây` và hiện ô Ảnh 3; bấm `Xóa` ở ô 1 → còn 2 ô.
5. Motion: thanh thời lượng bị khóa; tick `Giới hạn thời lượng` thì mở; dòng ước tính hiện `⏱ Chọn video mẫu để ước tính.`
6. Log không có traceback: `grep -n "Traceback" "$TEMP/simple_app_ui.log"` không ra dòng nào.

Nếu có bước nào sai: sửa `build_ui`, khởi động lại app, kiểm tra lại. Xong thì dừng app (lệnh `taskkill` ở trên).

- [ ] **Step 6: Commit**

```bash
git add simple_app.py docs/superpowers/specs/2026-10-01-simple-app-v2-design.md
git commit -m "Rebuild the UI as three tabs with guide dialog, estimates and upscale

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Harness có assert thời lượng, chạy Reels 15 s

**Files:**
- Modify: `simple_app.py` (thay toàn bộ `run_check`)
- Modify: `docs/superpowers/specs/2026-10-01-simple-app-v2-design.md` (chữ ký harness Motion có thêm `[giây]`; ghi kết quả)

**Interfaces:**
- Consumes: `reel_plan`, `reels_settings`, `ideas_instructions`, `detail_instructions`, `motion_frames`, `motion_settings`, `ANIMATE_MODEL`, `MOTION_MODEL`, `_video_info`, `_image_size`, `enhance`, `parse_ideas`, `pick_resolution`, `extend_settings`, `EXTEND_MODEL`
- Produces: CLI
  - `python simple_app.py --check-enhancer <ảnh>`
  - `python simple_app.py --check-reel <ảnh> [giây]` (mặc định 5)
  - `python simple_app.py --check-extend <video> <ảnh> <prompt ...>` (như v1)
  - `python simple_app.py --check-motion <ảnh> <video> [animate2|animate] [giây]`

- [ ] **Step 1: Thay `run_check`**

```python
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
```

- [ ] **Step 2: Sửa spec**

Trong `docs/superpowers/specs/2026-10-01-simple-app-v2-design.md`, mục "Kiểm thử", thay dòng `` - `--check-motion <ảnh> <video> [animate2|animate]`: toàn bộ video mẫu; ...`` bằng:

```markdown
- `--check-motion <ảnh> <video> [animate2|animate] [giây]`: mặc định toàn bộ video mẫu; mong đợi 27.06 s so với 27.10 s, khoảng 1.8 giờ, chạy sau cùng. Thêm `[giây]` để chạy thử ngắn (dùng cho Animate 14B).
```

- [ ] **Step 3: Test thuần vẫn qua**

Run: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py && "$USERPROFILE/anaconda3/envs/wan2gp/python.exe" -m py_compile simple_app.py && echo compiled`
Expected: `ok` rồi `compiled`

- [ ] **Step 4: Chạy harness Reels 15 s (GPU, khoảng 35–40 phút)**

Dừng mọi tiến trình đang nghe cổng 7860 (lệnh ở Task 5 Step 5), rồi chạy nền:

```bash
cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python simple_app.py --check-reel D:\001_Personal_Proj\Comfy\PostIG\day_53_1\Ryo\ZImage_2026_06_06_101818_60_00001.png 15" > "$TEMP/check_reel15.log" 2>&1
```

Khi xong: `tr '\r' '\n' < "$TEMP/check_reel15.log" | grep -n "IDEAS\|DETAIL\|REEL:\|duration\|Error\|Traceback\|assert" | tail -20`

Expected: dòng `REEL: ...mp4`, dòng `duration 15.0Xs (... frames @ 32 fps), expected 15.00s`, không có `Traceback`.

Nếu lỗi: dùng superpowers:systematic-debugging; nguyên nhân hay gặp là Wan2GP từ chối tổ hợp `S` + `KI` + `PW` (đọc thông báo lỗi validate trong log, đối chiếu `wgp.py:1440-1520` và `shared/utils/frame_scheduler.py`).

Mở video ra, so mặt ở giây 0, 5, 10, 15 với ảnh gốc (trích frame bằng `cv2` hoặc xem trực tiếp). Ghi nhận xét.

- [ ] **Step 5: Ghi kết quả vào spec**

Thêm vào cuối `docs/superpowers/specs/2026-10-01-simple-app-v2-design.md`:

```markdown
## Kết quả đo

| Việc | Kết quả |
|---|---|
| `--check-reel` Ryo 15 s, `480x832`, SVI 2 Pro 3 cửa sổ | <thời lượng ra> s, <tổng thời gian chạy>, ước tính trước khi hiệu chỉnh 35 phút; mặt: <nhận xét> |
```

Điền số thật lấy từ log (tổng thời gian: hiệu giữa dòng đầu và dòng `REEL:`; hoặc đo bằng `time` khi chạy).

- [ ] **Step 6: Commit**

```bash
git add simple_app.py docs/superpowers/specs/2026-10-01-simple-app-v2-design.md
git commit -m "Assert output duration in headless checks, record 15 s reel run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Motion Control đúng thời lượng trên GPU

**Files:**
- Modify: `docs/superpowers/specs/2026-10-01-simple-app-v2-design.md` (bảng Kết quả đo)

**Interfaces:**
- Consumes: CLI `--check-motion` từ Task 6

- [ ] **Step 1: Chạy thử ngắn Animate 14B + FusionX (khoảng 10–20 phút)**

Dừng tiến trình trên cổng 7860, rồi chạy nền:

```bash
cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python simple_app.py --check-motion D:\001_Personal_Proj\Comfy\srcs\260914\motionControl\day_27_1\Klein_I2I_260920_77_00001_.png D:\001_Personal_Proj\Comfy\srcs\260914\motionControl\day_27_1\vietnameseboy__1788933546_3982185246061537055_27770465737.mp4 animate 5" > "$TEMP/check_motion_animate5.log" 2>&1
```

Expected: `MOTION: ...mp4`, `duration 5.0Xs ..., expected 5.00s`, không có `Traceback`. Ghi giây/bước từ log (thanh tiến độ khử nhiễu) để so với mốc 94 giây/bước đang mượn.

Nếu lỗi nạp LoRA FusionX (tên file không nhận) hoặc lỗi chế độ `PVBKI`: dùng superpowers:systematic-debugging; đối chiếu `models/wan/wan_handler.py:769-778` và `profiles/wan_i2v/Image2Video FusioniX - 10 Steps.json`. Không đổi model âm thầm; báo người dùng.

- [ ] **Step 2: Chạy Motion đủ 27 giây với Animate 2 (khoảng 1.8 giờ)**

```bash
cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python simple_app.py --check-motion D:\001_Personal_Proj\Comfy\srcs\260914\motionControl\day_27_1\Klein_I2I_260920_77_00001_.png D:\001_Personal_Proj\Comfy\srcs\260914\motionControl\day_27_1\vietnameseboy__1788933546_3982185246061537055_27770465737.mp4" > "$TEMP/check_motion_full.log" 2>&1
```

Expected: `duration 27.06s (433 frames @ 16 fps), expected 27.10s`, không có `Traceback`. So mặt và trang phục ở đầu, giữa, cuối với ảnh nhân vật.

- [ ] **Step 3: Ghi kết quả vào spec**

Thêm hai dòng vào bảng "Kết quả đo" của spec v2:

```markdown
| `--check-motion` Animate 14B + FusionX, 5 s | <thời lượng ra> s, <thời gian>, <giây/bước> giây/bước (mốc đang mượn: 94) |
| `--check-motion` Animate 2, đủ 27.1 s, 433 frame, 7 cửa sổ | <thời lượng ra> s, <thời gian>, ước tính 1 giờ 44 phút; mặt: <nhận xét> |
```

Nếu giây/bước của Animate 14B lệch hơn 30% so với 94, sửa giá trị `ANIMATE_MODEL` trong `SPEED` của `simple_app.py` theo số đo, chạy lại `test_simple_app.py` (test `estimate_seconds` cho `ANIMATE_MODEL` cũng phải đổi theo số mới), và đưa cả hai file vào commit.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-10-01-simple-app-v2-design.md
git commit -m "Record full-length Motion Control and Animate 14B runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Thêm `simple_app.py test_simple_app.py` vào `git add` nếu Step 3 đã sửa `SPEED`.)

---

## Ngoài plan này

- Harness Multi Image: viết khi người dùng có bộ ảnh mẫu cùng nhân vật (spec: "viết sau").
- Chạy tay của người dùng: Upscale ×2 một video, một video Multi 2–3 ảnh.
