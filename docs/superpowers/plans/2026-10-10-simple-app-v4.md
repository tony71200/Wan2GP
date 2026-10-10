# Simple App v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reels Multi Image chạy Segment 1–15 s với prompt theo giây, có "Viết chi tiết" biết ảnh đầu, ảnh cuối và Segment trước; cả hai tab Reels hiện prompt đã gửi; log `logs/simple_app.log`; harness `--check-multi` đọc Story VI/EN và chạy được enhancer.

**Architecture:** Vẫn một file `simple_app.py`. Hàm thuần mới (`multi_prompt`, `parse_story` trả thêm mô tả chung, `segment_instructions`, `segment_request`, `strip_think`, `log_block`) test không cần GPU. `enhance()` thêm `think`/`purpose`/`tab` và tự ghi log; `write_segment()` ở mức module để UI và harness dùng chung. Giao diện: slider 1–15, dòng trạng thái từng card, nút Viết chi tiết, ô Prompt đã gửi trong `results()`.

**Tech Stack:** Python 3.11 (env `wan2gp`), Gradio 5.29, `shared.api`, Wan2GP prompt enhancer (Qwen3.5-4B), OpenCV, Pillow, `ffmpeg_bins/ffprobe.exe`.

**Spec:** `docs/superpowers/specs/2026-10-10-simple-app-v4-design.md`

## Global Constraints

- Nhánh `20261010_Simple_3_1`. Không commit `run_wgp.bat`, `docs/superpowers/PR_*.md`, `logs/`.
- Không sửa code Wan2GP (`wgp.py`, `shared/`, `models/`), không thêm thư viện, không tải model.
- Enhancer là Qwen3.5-4B đang có (`enhancer_enabled = 3`); chế độ Think = chữ `K` qua `wgp.build_prompt_enhancer_value("TI", True)` → `"TIK"`.
- Một Segment = một cửa sổ (Wan2GP tiêu thụ ảnh kết thúc theo thứ tự cửa sổ); không chia nhỏ Segment.
- Prompt cửa sổ: dòng `[/duration=Ns] <mô tả chung>` rồi các dòng Segment nguyên văn, xuống dòng đơn; cửa sổ cách nhau đúng một dòng trống; không có dòng trống bên trong cửa sổ.
- Chữ giao diện tiếng Việt, không emoji. Class CSS có sẵn: `eta` (xám mono), `note` (vàng).
- Test thuần: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py` → in `ok`.
- GPU: `cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python -u simple_app.py <tham số>" > <log> 2>&1`, chạy nền, mỗi lúc một job, dừng app ở cổng 7860 trước; đọc log bằng `tr '\r' '\n'`.
- Commit kết thúc bằng `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: Hàm thuần

**Files:**
- Modify: `simple_app.py` (các hàm `multi_settings`, `_one_line`, `parse_story`, `detail_instructions`, `estimate_seconds`, `log_harness`, hằng `HARNESS_LOG`)
- Test: `test_simple_app.py`

**Interfaces:**
- Produces:
  - `_lines(text) -> list[str]`: các dòng không trống, đã strip.
  - `multi_prompt(common, prompts, durations) -> str`.
  - `multi_settings(...)` dùng `multi_prompt`; chữ ký không đổi.
  - `parse_story(text, lang="en") -> tuple[str, list[tuple[str, int, str]]]` (mô tả chung, [(tên ảnh, giây, prompt nhiều dòng)]); thời lượng 1–15.
  - `VEED_RULES: str`; `detail_instructions(seconds=5)` chứa `VEED_RULES` và "at most 25 words".
  - `segment_instructions(seconds, has_end=False, has_prev=False) -> str`.
  - `segment_request(common, hint, seconds, caption=None, previous=None) -> str`.
  - `strip_think(text) -> str`.
  - `estimate_seconds(model, windows, resolution, calib=None)`: công việc mỗi cửa sổ `f * (f / ref_frames) ** 0.5`.
  - `APP_LOG = ROOT / "logs" / "simple_app.log"`; `log_block(entry, path=APP_LOG)`; `HARNESS_LOG` giữ nguyên. `log_harness` và `_one_line` bị xóa.

- [ ] **Step 1: Test trước.** Trong `test_simple_app.py` sửa dòng import: bỏ `log_harness`, thêm `multi_prompt, segment_instructions, segment_request, strip_think, log_block, VEED_RULES`. Thay `test_multi_settings`, `test_parse_story`, `test_log_harness` và thêm test mới:

```python
def test_multi_prompt():
    assert multi_prompt("A man in a red coat.", ["(at 0 seconds: he waves)\n\n(at 1 seconds: he smiles)", "", "sits.\ndown"], [3, 2, 4]) == (
        "[/duration=3s] A man in a red coat.\n(at 0 seconds: he waves)\n(at 1 seconds: he smiles)\n\n"
        "[/duration=2s] A man in a red coat.\n\n"
        "[/duration=4s] A man in a red coat.\nsits.\ndown"
    )
    assert multi_prompt("", ["waves"], [2]) == "[/duration=2s] waves"
    assert multi_prompt("  two   words ", [" x \n\n y "], [1]) == "[/duration=1s] two words\nx\ny"
    assert multi_prompt("", [""], [1]) == "[/duration=1s]"


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
        "[/duration=3s] A man in a red coat.\nwalks in\n\n"
        "[/duration=2s] A man in a red coat.\n\n"
        "[/duration=4s] A man in a red coat.\nsits.\ndown"
    )
    assert s["video_length"] == 145 and s["resolution"] == "480x832" and s["seed"] == 1
    assert s["loras_multipliers"] == "1;0" and s["temporal_upsampling"] == ""
    one = multi_settings({}, images=["a.png"], durations=[2], prompts=["waves"], common="", resolution="480x832", seed=-1)
    assert one["image_prompt_type"] == "S" and one["image_end"] is None
    assert one["prompt"] == "[/duration=2s] waves" and one["video_length"] == 33
    long = multi_settings({}, images=["a.png", "b.png"], durations=[10, 5], prompts=["x", "y"], common="", resolution="480x832", seed=1)
    assert long["video_length"] == 241
    assert MAX_IMAGES == 6


def test_parse_story():
    text = (
        "\ufeff(VI) Mô tả chung.\n(EN) Common description.\n\n"
        "A %x% .png :0–10s:\n- (VI) Anh cười.\n- (EN) He smiles.\nHe waves.\n\n"
        "b.png: 10-15s:\n(EN) (at 0 seconds: he sits)\n(at 1 seconds: down)\n\n"
        "c.jpg : 2s :\n• (vi) Anh ngồi.\n"
    )
    assert parse_story(text) == ("Common description.", [
        ("A %x% .png", 10, "He smiles.\nHe waves."),
        ("b.png", 5, "(at 0 seconds: he sits)\n(at 1 seconds: down)"),
        ("c.jpg", 2, ""),
    ])
    assert parse_story(text, "vi") == ("Mô tả chung.", [
        ("A %x% .png", 10, "Anh cười.\nHe waves."),
        ("b.png", 5, "(at 1 seconds: down)"),
        ("c.jpg", 2, "Anh ngồi."),
    ])
    assert parse_story("a.png : 3s :\nx\ny") == ("", [("a.png", 3, "x\ny")])
    for bad in ("a.png :0–16s:\nx", "a.png :3–3s:\nx", "a.png 0-3s\nx", "a.png : 2s :\nx\n\nno header here\ny"):
        try:
            parse_story(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_segment_instructions():
    s8 = segment_instructions(8)
    assert "in order: 0, 1, 2, 3, 4, 5, 6, 7." in s8
    assert "in order: 0, 2, 4, 6, 8." in segment_instructions(9)
    assert "in order: 0, 2, 4, 6, 8, 10, 12, 14." in segment_instructions(15)
    assert "end pose" not in s8 and "previous clip" not in s8
    assert "end pose" in segment_instructions(5, has_end=True)
    assert "previous clip" in segment_instructions(5, has_prev=True)
    assert "at most 25 words" in s8 and "no zoom" in s8 and VEED_RULES in s8
    assert "static on the first line and on the last line" in s8
    assert VEED_RULES in detail_instructions(5) and "at most 25 words" in detail_instructions(5)
    assert "Write in English" in VEED_RULES


def test_segment_request():
    assert segment_request("A man.", "he waves", 5, "sitting on the edge", "(at 3 seconds: a)\n(at 4 seconds: b)\n") == (
        "Character and setting: A man.\nGoal: he waves\nThe clip lasts 5 seconds.\n"
        "It must end exactly in this pose: sitting on the edge\nThe previous clip ended with: (at 4 seconds: b)"
    )
    assert segment_request("", "", 3) == "Goal: Continue the scene naturally\nThe clip lasts 3 seconds.\nThe clip ends freely."
    assert segment_request(" a  b ", " c\nd ", 2, previous="") == "Character and setting: a b\nGoal: c d\nThe clip lasts 2 seconds.\nThe clip ends freely."


def test_strip_think():
    assert strip_think("<think>\nplan\n</think>\n(at 0 seconds: x)\n") == "(at 0 seconds: x)"
    assert strip_think("(at 0 seconds: x)") == "(at 0 seconds: x)"
    assert strip_think("") == ""


def test_estimate_long_window():
    assert estimate_seconds(REELS_MODEL, [161], "720x720") > estimate_seconds(REELS_MODEL, [81, 81], "720x720")
    assert estimate_seconds(REELS_MODEL, [81], "720x720") == 120 + 4 * 97


def test_log_block():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "logs" / "h.log"
        log_block({"command": "generate", "prompt": "a\nb", "ok": True}, p)
        log_block({"command": "--check-multi x", "ok": False}, p)
        text = p.read_text(encoding="utf-8")
        assert text.count("=== ") == 2
        assert " generate\nprompt: a\n  b\nok: True\n\n" in text
        assert "--check-multi x\nok: False\n" in text
```

- [ ] **Step 2: Chạy test, phải fail** vì thiếu import: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py` → `ImportError: cannot import name 'multi_prompt'`.

- [ ] **Step 3: Viết code.** Trong `simple_app.py`:

Thay `_one_line` và `multi_settings` (dòng 205–227) bằng:

```python
def _lines(text):
    """Non-empty stripped lines of `text` (PW windows must not contain blank lines)."""
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def multi_prompt(common, prompts, durations):
    """One `[/duration=Ns]` paragraph per segment: the common description first, then the segment's lines verbatim."""
    paragraphs = []
    for d, p in zip(durations, prompts):
        body = [" ".join(str(common or "").split())] + _lines(p)
        paragraphs.append(" ".join([f"[/duration={int(d)}s]", "\n".join(b for b in body if b)]).rstrip())
    return "\n\n".join(paragraphs)


def multi_settings(defaults, *, images, durations, prompts, common, resolution, seed, loras=(), high=1.0, low=0.0, smooth=True, detail=False):
    """One Lightning job through every image: window i starts at image i, lasts durations[i]
    and ends on image i+1; the last window has no end image."""
    return {
        **defaults,
        "model_type": REELS_MODEL,
        "prompt": multi_prompt(common, prompts, durations),
        "multi_prompts_gen_type": "PW",
        "image_prompt_type": "SE" if len(images) > 1 else "S",
        "image_start": images[0],
        "image_end": list(images[1:]) or None,
        "resolution": resolution,
        "video_length": sum(int(d) for d in durations) * FPS + 1,
        "seed": int(seed),
        **_lora_settings(loras, high, low, smooth, detail),
    }
```

Thay `parse_story` (dòng 245–257) bằng:

```python
_LANG = re.compile(r"^\((vi|en)\)\s*", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.):])\s*")
_IMAGE_EXT = re.compile(r"\.(?:png|jpe?g|webp)\b", re.I)


def _story_lines(lines, lang):
    """Lines of a Story block for `lang`: bullets dropped, `(VI)`/`(EN)` prefixes filtered, unprefixed lines kept."""
    out = []
    for line in lines:
        line = _BULLET.sub("", line).strip()
        m = _LANG.match(line)
        if m:
            if m[1].lower() != lang:
                continue
            line = line[m.end():].strip()
        if line:
            out.append(line)
    return out


def parse_story(text, lang="en"):
    """(common description, [(image name, whole seconds, prompt lines)]) from a Story file, in file order.
    A first block without an image heading is the common description."""
    common, story = "", []
    for i, block in enumerate(re.split(r"\n\s*\n", str(text).lstrip("\ufeff").strip())):
        head, *body = block.strip().splitlines()
        m = _STORY_HEAD.match(head.strip())
        if not m:
            if i == 0 and not _IMAGE_EXT.search(head):
                common = " ".join(_story_lines(block.splitlines(), lang))
                continue
            raise ValueError(f"Story: không đọc được dòng đầu khối: {head!r}")
        seconds = float(m["end"]) - float(m["start"] or 0)
        if seconds != int(seconds) or not 1 <= seconds <= 15:
            raise ValueError(f"Story: {m['name'].strip()}: thời lượng {seconds:g}s phải là số giây nguyên từ 1 đến 15")
        story.append((m["name"].strip(), int(seconds), "\n".join(_story_lines(body, lang))))
    return common, story
```

Thay `HARNESS_LOG` và `log_harness` (dòng 270–283) bằng:

```python
HARNESS_LOG = ROOT / "logs" / "simple_app_harness.log"
APP_LOG = ROOT / "logs" / "simple_app.log"


def log_block(entry, path=APP_LOG):
    """Append one block: a '=== time command' line, then 'key: value' lines; later lines of a value are indented."""
    import time

    entry = dict(entry)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {entry.pop('command', '')}"]
    lines += [f"{k}: " + str(v).replace("\n", "\n  ") for k, v in entry.items()]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")
```

Trong `estimate_seconds` thay dòng `work = ...` bằng:

```python
    # ponytail: f^1.5 sits between linear layers and quadratic attention; the calibration ratio absorbs the rest
    work = sum(f * (f / ref_frames) ** 0.5 for f in windows) * w * h / (ref_frames * ref_pixels)
```

Thay `detail_instructions` (dòng 378–390) bằng, và thêm `segment_instructions`, `segment_request`, `strip_think` ngay sau:

```python
VEED_RULES = (
    "Each line describes exactly one main movement of the person and one concrete camera move (static, pan, tilt, dolly, slow push): "
    "say how things move, do not re-describe the image.\n"
    "Build the motion progressively: start subtle and grow through the clip.\n"
    "Write in English even if the goal is written in another language.\n"
)

_SHOT_RULES = (
    "Describe ONE continuous shot that starts exactly from the image: no cuts, no scene change, no change of location, lighting, weather or time of day.\n"
    "The person must stay the same person for the whole video: same face, hairstyle, hair color, body, skin tone, outfit and accessories as in the image, "
    "unless the goal says they put on, take off or pick up something. Never describe a different look.\n"
    "Keep the motion natural and modest so the face stays recognizable. The camera is static or moves slowly.\n"
    "The camera keeps the same distance and framing as the image: no zoom, no push-in, no close-up on the face; the person may walk closer by themselves.\n"
)

_LINE_RULES = (
    "Each line must begin with \"(at X seconds:\" where X is that second number, followed by a dense, comma-separated visual description "
    "of at most 25 words, all inside a single set of parentheses.\n"
    "Do not include commentary, explanations, or meta-text."
)


# The model's own enhancer instructions ask for shot changes every second, which turns the person
# into someone else. These keep one continuous shot and restate the subject on every line.
def detail_instructions(seconds=5):
    return (
        "You are an AI vision model writing a prompt for an image-to-video model. Analyze the provided image and the user's goal.\n"
        + _SHOT_RULES + VEED_RULES +
        "Output rules:\n"
        f"Write one line per second, from 0 to {seconds - 1} seconds.\n"
        "Every line repeats the same subject and setting description taken from the image and changes only the action and the camera movement.\n"
        + _LINE_RULES
    )


def segment_instructions(seconds, has_end=False, has_prev=False):
    """Per-second (or per-2-second above 8 s, for the 512-token window limit) lines for one Multi Image segment."""
    step = 1 if seconds <= 8 else 2
    marks = ", ".join(str(t) for t in range(0, int(seconds), step))
    text = (
        "You are an AI vision model writing a prompt for an image-to-video model. The image is the first frame of a clip; analyze it and the user's goal.\n"
        + _SHOT_RULES + VEED_RULES +
        "The camera is static on the first line and on the last line.\n"
    )
    if has_prev:
        text += "The first line continues exactly the action the previous clip ended with, with no pause or reset.\n"
    if has_end:
        text += "The last line brings the person exactly into the end pose given by the user; pace the earlier lines so this pose is reachable in time.\n"
    return text + "Output rules:\n" + f"Write one line for each of these seconds, in order: {marks}.\n" + _LINE_RULES


def segment_request(common, hint, seconds, caption=None, previous=None):
    """The user message for `segment_instructions`: setting, goal, length, end pose and the previous segment's last line."""
    parts = []
    if (common or "").strip():
        parts.append(f"Character and setting: {' '.join(common.split())}")
    parts.append(f"Goal: {' '.join(hint.split()) if (hint or '').strip() else 'Continue the scene naturally'}")
    parts.append(f"The clip lasts {int(seconds)} seconds.")
    parts.append(f"It must end exactly in this pose: {caption.strip()}" if caption else "The clip ends freely.")
    last = _lines(previous)
    if last:
        parts.append(f"The previous clip ended with: {last[-1]}")
    return "\n".join(parts)


def strip_think(text):
    """Drop a leaked Qwen `<think>…</think>` block."""
    return re.sub(r"<think>.*?</think>\s*", "", str(text or ""), flags=re.S).strip()
```

Trong `run_check` tạm đổi `log_harness(entry)` → `log_block(entry, HARNESS_LOG)` và `log_harness({...})` của audio check → `log_block({...}, HARNESS_LOG)`; `--check-multi` tạm đổi `story = parse_story(...)` → `common, story = parse_story(...)` và `common=""` → `common=common` (Task 4 viết lại nhánh này).

- [ ] **Step 4: Chạy test, phải qua**: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py` → `ok`. Kiểm tra thêm `grep -n "_one_line\|log_harness" simple_app.py test_simple_app.py` không còn kết quả.

- [ ] **Step 5: Commit**

```bash
git add simple_app.py test_simple_app.py
git commit -m "Multi prompt keeps per-second lines, story with VI/EN and common block, segment instructions, log_block

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `enhance()` có Think và log, `write_segment()`

**Files:**
- Modify: `simple_app.py` (`enhance` dòng 393–415; `suggest_ideas`, `write_detail` trong `build_ui`; nhánh `--check-enhancer` trong `run_check`)

**Interfaces:**
- Consumes: `segment_instructions`, `segment_request`, `strip_think`, `log_block`, `_lines` (Task 1).
- Produces:
  - `enhance(session, model_type, text, image, instructions=None, progress=None, think=False, purpose="", tab="") -> str`: ghi khối `enhance` (hoặc `error`) vào `APP_LOG`, lọc think.
  - `POSE_INSTRUCTIONS: str`.
  - `write_segment(session, image, end_image, seconds, common, hint, previous=None, progress=None, tab="multi", index=0) -> str`.

- [ ] **Step 1: Viết `enhance` và `write_segment`.** Thay `enhance` bằng:

```python
def enhance(session, model_type, text, image, instructions=None, progress=None, think=False, purpose="", tab=""):
    """Run Wan2GP's local prompt enhancer on (text, image) and return the generated text; every call is logged."""
    wgp = sys.modules["wgp"]
    from shared.utils.model_unload import wait_for_model_unload
    from shared.utils.process_locks import acquire_GPU_ressources, release_GPU_ressources

    # ponytail: session._state is private to WanGPSession; switch to the plugin-provided state if the API changes
    state = session._state
    model_def = wgp.get_model_def(model_type) if model_type else None
    mode = wgp.build_prompt_enhancer_value("TI", think)  # "TIK" = text + image + Qwen thinking
    wait_for_model_unload()
    acquire_GPU_ressources(state, "prompt_enhancer", "Prompt Enhancer")
    try:
        wgp.ensure_prompt_enhancer_loaded(progress=progress)
        out = wgp.process_prompt_enhancer(
            model_type, model_def, mode, [text], image, None, False, False, -1,
            prompt_enhancer_instructions=instructions,
        )
    except Exception as e:
        log_block({"command": "error", "tab": tab, "purpose": purpose, "prompt": text, "error": repr(e)})
        raise
    finally:
        wgp.unload_prompt_enhancer_runtime()
        if wgp.enhancer_offloadobj is not None:
            wgp.enhancer_offloadobj.unload_all()
        release_GPU_ressources(state, "prompt_enhancer")
    result = strip_think(out[0]) if out else ""
    log_block({"command": "enhance", "tab": tab, "purpose": purpose, "image": image, "think": think, "input": text, "output": result})
    return result


POSE_INSTRUCTIONS = (
    "You are an AI vision model. Describe the provided image in ONE English sentence: the person's pose, where they are in the frame, "
    "what they wear or hold, and the camera framing (distance and angle). No commentary, no list, no meta-text."
)


def write_segment(session, image, end_image, seconds, common, hint, previous=None, progress=None, tab="multi", index=0):
    """Timed lines for one Multi Image segment: the start image, the end image's pose (captioned first),
    the common description, the user's hint and the previous segment's last line."""
    caption = None
    if end_image:
        caption = enhance(session, "", "Describe this image.", end_image, POSE_INSTRUCTIONS, progress, purpose=f"caption {index + 1}", tab=tab)
    request = segment_request(common, hint, seconds, caption, previous)
    return enhance(session, "", request, image, segment_instructions(int(seconds), bool(end_image), bool(_lines(previous))), progress,
                   think=True, purpose=f"segment {index + 1}", tab=tab)
```

- [ ] **Step 2: Cập nhật các chỗ gọi.** Trong `build_ui`:

```python
        ideas = parse_ideas(enhance(session, "", text, image, ideas_instructions(int(seconds)), progress, purpose="ideas", tab="reels"))
```

```python
        return enhance(session, "", hint.strip(), image, detail_instructions(int(seconds)), progress, think=True, purpose="detail", tab="reels")
```

Trong `run_check`, nhánh `--check-enhancer`:

```python
            ideas = parse_ideas(enhance(session, "", "Suggest short video ideas for this image.", image, ideas_instructions(seconds), purpose="ideas", tab=flag))
            print("IDEAS:", ideas)
            assert ideas, "no ideas returned"
            prompt = enhance(session, "", ideas[0], image, detail_instructions(seconds), think=True, purpose="detail", tab=flag)
```

- [ ] **Step 3: Kiểm tra tĩnh**: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" -c "import ast,sys; ast.parse(open('simple_app.py',encoding='utf-8').read()); print('syntax ok')"` và chạy lại `test_simple_app.py` → `ok`. (Chạy thật ở Task 5, bước 7.)

- [ ] **Step 4: Commit**

```bash
git add simple_app.py
git commit -m "Enhancer: Think mode, per-call log, write_segment with end-image caption and previous segment

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Giao diện

**Files:**
- Modify: `simple_app.py` (`CSS`, `GUIDE`, `build_ui`: `results`, `generate`, `make_reel`, `extend_reel`, `multi_status`, `make_multi`, `make_motion`, tab Multi, phần nối sự kiện)

**Interfaces:**
- Consumes: `write_segment`, `log_block`, `fmt_minutes`, `save_calib`, `estimate_seconds`.
- Produces: `results()` trả `(go, eta, video, took, sent, ext, up)`; `generate(settings, windows, progress, tab)` trả `(video, took_text, prompt)`.

- [ ] **Step 1: CSS và Hướng dẫn.** Thêm vào cuối `CSS` (trước `"""`):

```python
.mono textarea {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; }}
```

Thay mục "Reels Multi Image" trong `GUIDE` bằng:

```markdown
### Reels Multi Image: tối đa 6 ảnh của cùng một nhân vật
Mỗi ảnh là một mốc. Đoạn của ảnh *i* bắt đầu ở ảnh *i*, dài 1–15 giây theo slider và kết thúc ở ảnh *i+1*; đoạn của ảnh cuối chạy tự do. Trên 8 giây có cảnh báo: model huấn luyện ở 5 giây, đoạn dài hay lặp động tác và tốn thời gian hơn nhiều. Độ phân giải theo ảnh 1.
- **Prompt theo giây**: mỗi dòng `(at N seconds: …)` là định dạng model hiểu; mốc kiểu `(2s-4s)` trong một đoạn không có tác dụng. Mô tả nhân vật chung được ghép vào đầu mọi đoạn.
- **Viết chi tiết**: enhancer đọc ảnh đầu, tư thế ảnh cuối, ý ngắn bạn gõ và dòng cuối của đoạn trước, rồi viết các dòng theo giây. *Viết chi tiết tất cả* đi lần lượt từ đoạn 1. Mỗi lần gọi mất khoảng một phút.
- **Mối nối**: hai đoạn kề nhau chỉ chung 1 frame, nên giây cuối đoạn trước và giây đầu đoạn sau nên tả cùng một động tác, máy quay đứng yên ở ranh giới, không mở đầu đoạn bằng hiệu ứng mới.
- **Ngôn ngữ**: viết prompt tiếng Anh; ý ngắn tiếng Việt thì Viết chi tiết sẽ dịch. Prompt tiếng Việt gửi thẳng chạy được nhưng bám yếu hơn.
- **Prompt đã gửi**: sau khi tạo xong, ô này hiện đúng chuỗi đã gửi cho Wan2GP; `logs/simple_app.log` ghi lại mọi prompt và mọi lần gọi enhancer.
```

- [ ] **Step 2: `results()` và `generate()`.** Thay hai hàm trong `build_ui`:

```python
    def generate(settings, windows, progress, tab):
        """Run a generation job, log it and record actual / estimated time for its model. Returns (video, took_text, prompt)."""
        model = settings["model_type"]
        start = time.monotonic()
        try:
            video = run(lambda cb: session.submit_task(settings, callbacks=cb), progress)
        except Exception as e:
            log_block({"command": "error", "tab": tab, "prompt": settings["prompt"], "error": str(e)})
            raise
        if video is None:
            return None, "", settings["prompt"]
        took = time.monotonic() - start
        calib = save_calib(model, took, estimate_seconds(model, windows, settings["resolution"]))
        log_block({
            "command": "generate", "tab": tab, "model": model, "resolution": settings["resolution"], "seed": settings["seed"],
            "windows": len(windows), "prompt": settings["prompt"], "video": video, "took": fmt_minutes(took), "calib": calib[model],
        })
        text = f"Thực tế {fmt_minutes(took)}"
        removed = len(windows) - 1  # keep_intermediate_sliding_windows = 0: each window replaced the previous file
        if removed > 0:
            gr.Info(f"Đã xóa {removed} video trung gian, chỉ giữ video cuối.")
            text += f" · đã xóa {removed} video trung gian"
        return video, text, settings["prompt"]
```

```python
    def results(extend=False):
        """Right column of a tab: actions, estimate, video, timing, follow-up buttons, the prompt sent."""
        with gr.Row():
            cancel_btn = gr.Button("Hủy", variant="secondary")
            go = gr.Button("Tạo Video", variant="primary")
        eta = gr.Markdown(elem_classes="eta")
        video = gr.Video(label="Video ra", autoplay=True, interactive=False)
        took = gr.Markdown(elem_classes="eta")
        with gr.Row():
            ext = gr.Button("Kéo dài video (neo ảnh gốc)", variant="secondary", interactive=False) if extend else None
            up = gr.Button("Upscale ×2", variant="secondary", interactive=False)
        sent = gr.Textbox(label="Prompt đã gửi", lines=8, interactive=False, elem_classes="mono")
        cancel_btn.click(cancel, queue=False)
        up.click(upscale, video, video)
        return go, eta, video, took, sent, ext, up
```

- [ ] **Step 3: Các handler tạo video.** Trong bốn hàm dưới, `...` là các dòng hiện có (kiểm tra đầu vào và dựng `settings`) giữ nguyên; chỉ thay dòng gọi `generate` và hai dòng `return`:

```python
    def make_reel(image, prompt, seconds, quality, smooth, loras, high, low, detail, seed, progress=gr.Progress()):
        ...
        video, took, sent = generate(settings, windows, progress, "reels")
        if video is None:
            return (gr.update(),) * 6
        last = {"video": video, "image": image, "resolution": settings["resolution"]}
        return video, last, gr.update(interactive=True), gr.update(interactive=True), took, sent

    def extend_reel(last, prompt, seconds, smooth, loras, high, low, detail, seed, progress=gr.Progress()):
        ...
        video, took, sent = generate(settings, [settings["video_length"]], progress, "extend")
        if video is None:
            return (gr.update(),) * 4
        return video, {**last, "video": video}, took, sent
```

```python
    def make_multi(n, common, quality, smooth, loras, high, low, detail, seed, progress=gr.Progress(), *vals):
        ...
        video, took, sent = generate(settings, [d * FPS + 1 for d in durations], progress, "multi")
        if video is None:
            return (gr.update(),) * 4
        return video, gr.update(interactive=True), took, sent
```

```python
    def make_motion(image, video, description, limit_on, limit, quality, smooth, label, audio, outfit, seed, progress=gr.Progress()):
        ...
        out, took, sent = generate(settings, sliding_windows(frames, *MOTION_WINDOWS[model]), progress, "motion")
        if out is None:
            return (gr.update(),) * 4
        return out, gr.update(interactive=True), took, sent
```

- [ ] **Step 4: Trạng thái card và nút Viết chi tiết.** Thay `multi_status` và thêm ba hàm trong phần `# ---------- Reels Multi Image ----------`:

```python
    def card_status(n, i, seconds):
        if seconds > 8:
            return gr.update(value="Lưu ý: hơn 8 s, có thể lặp động tác và thời gian tăng nhanh", elem_classes="note")
        return gr.update(value="Đoạn cuối, chạy tự do" if i == n - 1 else f"Kết thúc ở ảnh {i + 2}", elem_classes="eta")

    def multi_status(n, quality, *vals):
        images, durations = vals[:n], vals[MAX_IMAGES:MAX_IMAGES + n]
        count = f"**{n}/{MAX_IMAGES} ảnh · tổng {sum(durations):g} giây**"
        eta = eta_text(REELS_MODEL, [int(d) * FPS + 1 for d in durations], image_resolution(images[0], quality), load_calib())
        off = aspect_mismatch([_image_size(im) if im else None for im in images])
        if off:
            eta += f"\n\nLưu ý: ảnh {', '.join(map(str, off))} lệch tỉ lệ khung với ảnh 1 hơn 10%: Wan2GP sẽ co và cắt."
        statuses = [card_status(n, i, durations[i]) if i < n else gr.update(value="") for i in range(MAX_IMAGES)]
        return [count, eta, *statuses]

    # progress must come before *vals: Gradio only injects gr.Progress into positional parameters
    def write_card(index):
        def write(n, common, progress=gr.Progress(), *vals):
            images, durations, prompts = vals[:MAX_IMAGES], vals[MAX_IMAGES:2 * MAX_IMAGES], vals[2 * MAX_IMAGES:]
            if index >= n or not images[index]:
                raise gr.Error(f"Ảnh {index + 1} chưa có.")
            if active["job"] is not None:
                raise gr.Error("Đang tạo video, chờ xong rồi viết.")
            end = images[index + 1] if index + 1 < n else None
            return write_segment(session, images[index], end, int(durations[index]), common, prompts[index],
                                 prompts[index - 1] if index else None, progress, index=index)

        return write

    def write_all(n, common, progress=gr.Progress(), *vals):
        images, durations, prompts = list(vals[:MAX_IMAGES]), vals[MAX_IMAGES:2 * MAX_IMAGES], list(vals[2 * MAX_IMAGES:])
        if not all(images[:n]):
            raise gr.Error("Ô ảnh nào đang hiện cũng phải có ảnh.")
        if active["job"] is not None:
            raise gr.Error("Đang tạo video, chờ xong rồi viết.")
        for i in range(n):
            end = images[i + 1] if i + 1 < n else None
            prompts[i] = write_segment(session, images[i], end, int(durations[i]), common, prompts[i], prompts[i - 1] if i else None, progress, index=i)
        return prompts
```

- [ ] **Step 5: Bố cục tab Multi.** Thay khối `with gr.Tab("Reels Multi Image"):` bằng:

```python
        with gr.Tab("Reels Multi Image"):
            mm_n = gr.State(2)
            mm_images, mm_durations, mm_status, mm_prompts, mm_cards, mm_writes, mm_removes = [], [], [], [], [], [], []
            with gr.Row():
                with gr.Column(scale=2):
                    mm_count = gr.Markdown()
                    mm_common = gr.Textbox(
                        label="Mô tả nhân vật chung (ghép vào đầu mọi đoạn)",
                        placeholder="a young man with short black hair, white t-shirt, in a sunny park",
                    )
                    with gr.Row():
                        for i in range(MAX_IMAGES):
                            with gr.Column(visible=i < 2, min_width=220) as card:
                                mm_images.append(gr.Image(type="filepath", label=f"Ảnh {i + 1}", sources=["upload", "clipboard"], height=220))
                                mm_durations.append(gr.Slider(1, 15, value=3, step=1, label="Thời lượng đoạn (giây)"))
                                mm_status.append(gr.Markdown(elem_classes="eta"))
                                mm_prompts.append(gr.Textbox(label=f"Prompt {i + 1}", lines=5, placeholder="ý ngắn, hoặc các dòng (at N seconds: …)"))
                                with gr.Row():
                                    mm_writes.append(gr.Button("Viết chi tiết", variant="secondary", size="sm"))
                                    mm_removes.append(gr.Button("Xóa", variant="secondary", size="sm"))
                            mm_cards.append(card)
                    with gr.Row():
                        mm_add = gr.Button("Thêm ảnh", variant="secondary")
                        mm_write_all = gr.Button("Viết chi tiết tất cả", variant="secondary")
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
                    mm_go, mm_eta, mm_video, mm_took, mm_sent, _, mm_up = results()
```

Sửa hai dòng gọi `results` còn lại:

```python
                    r_go, r_eta, r_video, r_took, r_sent, r_extend, r_up = results(extend=True)
```

```python
                    m_go, m_eta, m_out, m_took, m_sent, _, m_up = results()
```

- [ ] **Step 6: Nối sự kiện.** Thay phần nối của Reels, Multi và Motion:

```python
        r_go.click(
            make_reel,
            [r_image, r_prompt, r_seconds, r_quality, r_smooth, r_loras, r_high, r_low, r_detail, r_seed],
            [r_video, r_last, r_extend, r_up, r_took, r_sent],
        )
        r_extend.click(
            extend_reel,
            [r_last, r_prompt, r_seconds, r_smooth, r_loras, r_high, r_low, r_detail, r_seed],
            [r_video, r_last, r_took, r_sent],
        )

        slots = [*mm_images, *mm_durations, *mm_prompts]
        status_in = [mm_n, mm_quality, *mm_images, *mm_durations]
        status_out = [mm_count, mm_eta, *mm_status]
        for comp in (*mm_images, *mm_durations, mm_quality):
            comp.change(multi_status, status_in, status_out)
        demo.load(multi_status, status_in, status_out)
        mm_add.click(multi_add, mm_n, [mm_n, *mm_cards]).then(multi_status, status_in, status_out)
        for i, rm in enumerate(mm_removes):
            rm.click(multi_remove(i), [mm_n, *slots], [mm_n, *slots, *mm_cards]).then(multi_status, status_in, status_out)
        for i, btn in enumerate(mm_writes):
            btn.click(write_card(i), [mm_n, mm_common, *slots], mm_prompts[i])
        mm_write_all.click(write_all, [mm_n, mm_common, *slots], mm_prompts)
        mm_loras.focus(lora_choices, outputs=mm_loras)
        mm_go.click(
            make_multi,
            [mm_n, mm_common, mm_quality, mm_smooth, mm_loras, mm_high, mm_low, mm_detail, mm_seed, *slots],
            [mm_video, mm_up, mm_took, mm_sent],
        )
```

```python
        m_go.click(
            make_motion,
            [m_image, m_video, m_desc, m_limit_on, m_limit, m_quality, m_smooth, m_model, m_audio, m_outfit, m_seed],
            [m_out, m_up, m_took, m_sent],
        )
```

- [ ] **Step 7: Kiểm tra tĩnh và mở app.** Chạy `test_simple_app.py` → `ok`. Dừng app cũ ở cổng 7860 nếu có (`netstat -ano | findstr :7860` rồi `taskkill /PID <pid> /F`). Chạy nền `cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python -u simple_app.py" > logs/ui.log 2>&1`, mở `http://127.0.0.1:7860` bằng browser pane. Kiểm tra: tab Multi có 2 card với slider tới 15, dòng trạng thái "Kết thúc ở ảnh 2" / "Đoạn cuối, chạy tự do"; kéo slider card 1 lên 10 → dòng đổi thành cảnh báo vàng; Thêm ảnh → card 3 hiện và trạng thái card 2 đổi thành "Kết thúc ở ảnh 3"; ô "Prompt đã gửi" có ở ba tab; dialog Hướng dẫn hiện mục mới. Chưa bấm Viết chi tiết (Task 5). Tắt app.

- [ ] **Step 8: Commit**

```bash
git add simple_app.py
git commit -m "Multi tab: 1-15 s segments with status line, per-card and all-card Viết chi tiết, Prompt đã gửi on every tab

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Harness `--check-multi` với `--enhance`, `--lang`, log, và file story

**Files:**
- Modify: `simple_app.py` (`run_check`: argparse, `check`, nhánh `--check-multi`)
- Create: `D:\001_Personal_Proj\Comfy\result\story4_test20261010\story.txt` (ngoài repo, không commit)

**Interfaces:**
- Consumes: `parse_story(text, lang)`, `write_segment`, `multi_settings`, `log_block`, `HARNESS_LOG`.

- [ ] **Step 1: argparse.** Sau `--seed` thêm:

```python
    parser.add_argument("--enhance", action="store_true", help="Multi: rewrite every segment with the enhancer before generating")
    parser.add_argument("--lang", choices=["en", "vi"], default="en", help="Multi: which (VI)/(EN) lines of story.txt to use")
```

- [ ] **Step 2: `check()` ghi cả hai log.** Sau `log_harness`-cũ (nay `log_block(entry, HARNESS_LOG)`) thêm khối app:

```python
        log_block(entry, HARNESS_LOG)
        log_block({
            "command": "generate", "tab": flag, "model": settings["model_type"], "resolution": settings["resolution"], "seed": settings["seed"],
            "windows": len(windows), "prompt": settings["prompt"], "video": out, "took": fmt_minutes(took),
        })
```

- [ ] **Step 3: Nhánh `--check-multi`.** Thay bằng:

```python
    elif flag == "--check-multi":
        (folder,) = opts.inputs
        folder = Path(folder)
        common, story = parse_story((folder / "story.txt").read_text(encoding="utf-8"), opts.lang)
        images = [str(folder / name) for name, _, _ in story]
        durations = [seconds for _, seconds, _ in story]
        prompts = [prompt for *_, prompt in story]
        if opts.enhance:
            for i, (image, seconds) in enumerate(zip(images, durations)):
                end = images[i + 1] if i + 1 < len(images) else None
                prompts[i] = write_segment(session, image, end, seconds, common, prompts[i], prompts[i - 1] if i else None, tab=flag, index=i)
                print(f"SEGMENT {i + 1}:\n{prompts[i]}")
        w, h = _image_size(images[0])
        settings = multi_settings(
            session.get_default_settings(REELS_MODEL), images=images, durations=durations,
            prompts=prompts, common=common, resolution=pick_resolution(w, h, opts.quality),
            seed=opts.seed, detail=opts.detail,
        )
        starts = [sum(durations[:i]) for i in range(len(durations))]
        check("multi", settings, [d * FPS + 1 for d in durations], sum(durations), (*starts, sum(durations) - 0.2), images[0],
              {"segments": ", ".join(f"{name} {d}s" for name, d, _ in story), "lang": opts.lang, "enhance": opts.enhance})
```

- [ ] **Step 4: Viết `story.txt`** tại `D:\001_Personal_Proj\Comfy\result\story4_test20261010\story.txt` (UTF-8):

```
(VI) Nhân vật giữ nguyên diện mạo: hơi thở tự nhiên, chớp mắt nhẹ, đầu cử động thư thái.
(EN) The character keeps his natural appearance: realistic breathing, gentle blinking, relaxed subtle head movements.

R3_1_Ivan_Ryo_20261009_00001.png : 0-10s :
(VI) Anh từ từ cầm lấy một chiếc khăn trắng. Anh đứng dậy, quay lưng và mông về camera, rồi quấn chiếc khăn trắng quanh người. Anh bước sang trái, đi dọc thành bể onsen rồi ngồi xuống.
(EN) He slowly picks up a white towel. He stands up, turns his back and hips toward the camera, then wraps the white towel around himself. He steps to the left, moves along the edge of the onsen pool, and sits down.

R1_4_Ivan_Ryo_20261012_00001.png : 10-15s :
(VI) Anh vuốt ngược mái tóc ướt của mình.
(EN) He slicks back his wet hair.

R3_3_Ivan_Ryo_20261011_00001.png : 15-20s :
(VI) Anh đứng dậy, chuẩn bị đi về.
(EN) He stands up and gets ready to leave.

R4_4_Ivan_Ryo_20261012_00001.png : 20-23s :
(VI) Anh đi ngang qua camera, camera ngước dần lên trên.
(EN) He walks past the camera, and the camera tilts upwards.
```

- [ ] **Step 5: Kiểm tra parser trên file thật** (không GPU):

```bash
"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" -c "from simple_app import parse_story; c,s=parse_story(open(r'D:\001_Personal_Proj\Comfy\result\story4_test20261010\story.txt',encoding='utf-8').read()); print(c); [print(x) for x in s]; c,s=parse_story(open(r'D:\001_Personal_Proj\Comfy\result\story4_test20261010\story.txt',encoding='utf-8').read(),'vi'); print(c); [print(x) for x in s]; c,s=parse_story(open(r'D:\001_Personal_Proj\Comfy\PostIG\day_24_1\Ryo\story.txt',encoding='utf-8').read()); print(repr(c), [(n,d) for n,d,_ in s])"
```

Mong đợi: EN: 4 Segment (10, 5, 5, 3) với câu tiếng Anh; VI: cùng 4 Segment với câu tiếng Việt; story v3 cũ: `''` và `[(…, 3), (…, 4), (…, 4)]`. Chạy `test_simple_app.py` → `ok`.

- [ ] **Step 6: Commit**

```bash
git add simple_app.py
git commit -m "Harness --check-multi: --enhance, --lang, common block, generate log

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Chạy GPU, soát kết quả, ghi bảng đo

**Files:**
- Modify: `docs/superpowers/specs/2026-10-10-simple-app-v4-design.md` (bảng "Kết quả đo")
- Đọc: `logs/simple_app.log`, `logs/simple_app_harness.log`, `logs/*_multi.jpg`

Mỗi lệnh chạy nền, một lúc một lệnh; trước khi chạy, bảo đảm không app nào ở cổng 7860.

- [ ] **Step 1: `--check-enhancer`** (khoảng 3 phút):

```bash
cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python -u simple_app.py --check-enhancer D:\001_Personal_Proj\Comfy\PostIG\day_77_1\Ryo\ZImage_2026_05_27_224819_21_672251849304681_00001.png 5" > logs/check_enhancer.log 2>&1
```

Kiểm tra `tr '\r' '\n' < logs/check_enhancer.log | tail -30`: `IDEAS:` có 3 ý; `DETAIL:` có đúng 5 dòng bắt đầu `(at 0 seconds:` … `(at 4 seconds:`, mỗi dòng ≤ 25 từ, không có `<think>`; `enhancer check ok`. `logs/simple_app.log` có hai khối `enhance` (purpose `ideas` với `think: False`, `detail` với `think: True`). Nếu dòng dài hơn 25 từ nhiều hoặc thiếu dòng: siết câu chữ trong `_LINE_RULES`, chạy lại.

- [ ] **Step 2: `--check-multi --enhance`** (khoảng 50–60 phút):

```bash
cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python -u simple_app.py --check-multi D:\001_Personal_Proj\Comfy\result\story4_test20261010 --enhance --seed 20261010" > logs/check_multi_enhance.log 2>&1
```

Trong lúc chờ, đọc 4 khối `SEGMENT i:` trong log: Segment 1 có dòng ở 0, 2, 4, 6, 8; dòng cuối Segment 1 tả ngồi quấn khăn hướng camera (khớp ảnh 2); dòng đầu Segment 2 tiếp nối dòng cuối Segment 1. Khi xong: `duration 23.0x s`, `ok: True`. Xem `logs/<stamp>_multi.jpg` (Read ảnh) và trích thêm frame 2–6 s:

```bash
./ffmpeg_bins/ffmpeg.exe -v error -y -i "<video>" -vf "select='not(mod(n\,16))',scale=180:-1,tile=14x1" -frames:v 1 logs/<stamp>_multi_0_7s.jpg
```

Ghi nhận: khăn quấn sau khi quay lưng hay chưa; mối nối 10, 15, 20 s có giật không; Segment 10 s có lặp động tác không.

- [ ] **Step 3: `--check-multi --lang vi`** (không enhance, khoảng 45 phút):

```bash
cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python -u simple_app.py --check-multi D:\001_Personal_Proj\Comfy\result\story4_test20261010 --lang vi --seed 20261010" > logs/check_multi_vi.log 2>&1
```

Khối `generate` trong `logs/simple_app.log` phải có prompt tiếng Việt. So contact sheet với bước 2: Wan có làm theo câu tiếng Việt không (cầm khăn, quay lưng, vuốt tóc).

- [ ] **Step 4: Thử UI.** Chạy app, mở tab Multi, nạp 2 ảnh `R3_1` và `R1_4`, đoạn 2 + 2 s, gõ ý ngắn "he picks up the towel" ở Prompt 1, bấm **Viết chi tiết** card 1 → Prompt 1 thành 2 dòng `(at 0 seconds:` `(at 1 seconds:`; bấm **Viết chi tiết tất cả** → cả hai ô đổi. Tạo Video (khoảng 6 phút) → ô Prompt đã gửi hiện hai cửa sổ `[/duration=2s]`. Tab Reels: Viết chi tiết rồi Tạo Video 2 s → Prompt đã gửi hiện prompt. `logs/simple_app.log` có các khối tương ứng. Tắt app.

- [ ] **Step 5: Ghi bảng "Kết quả đo"** trong spec: thay ba dòng "chưa chạy" bằng kết quả của bước 1–3 (thời lượng, số frame, thời gian chạy, nhận xét về thứ tự động tác, mối nối, Segment 10 s, tiếng Việt), thêm dòng cho bước 4. Nếu bước 2 cho thấy Segment 10 s hỏng, ghi rõ và đề xuất chia Keyframe dày hơn (Rủi ro 1).

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-10-10-simple-app-v4-design.md
git commit -m "Record v4 harness runs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
