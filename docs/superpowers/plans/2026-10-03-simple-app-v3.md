# Simple App v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Giao diện tối theo các quy ước thiết kế, LoRA giữ da & tóc, harness Reels Multi Image đọc Story, âm thanh và tùy chọn trang phục cho Motion Control, không để lại Intermediate Video, log harness.

**Architecture:** Vẫn một file `simple_app.py`. Hàm thuần mới (`parse_story`, tham số `detail` của `_lora_settings`, `audio`/`outfit` của `motion_settings`, `log_harness`) được test không cần GPU. Giao diện chỉ đổi theme, CSS, chữ và thêm checkbox. Harness dùng `argparse` cho cờ `--detail`, `--outfit`, `--seed`.

**Tech Stack:** Python 3.11 (env `wan2gp`), Gradio 5.29, `shared.api`, OpenCV, Pillow, `ffmpeg_bins/ffprobe.exe`.

**Spec:** `docs/superpowers/specs/2026-10-03-simple-app-v3-design.md`

## Global Constraints

- Nhánh `20261003_simple_app_v3`. Không commit `run_wgp.bat`, `docs/superpowers/PR_20261001_simple_app_v2.md`.
- Không sửa code Wan2GP, không thêm thư viện. Không tải thêm model.
- LoRA chi tiết: `detailz-wan.safetensors` trong `loras/wan`, hệ số `0;0.6`.
- Bảng màu đúng bảng trong spec mục 1; không emoji trong chữ giao diện.
- Test thuần: `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" test_simple_app.py` → `ok`.
- GPU: `cmd //c "call %USERPROFILE%\anaconda3\Scripts\activate.bat wan2gp && python -u simple_app.py <tham số>" > <log> 2>&1`, mỗi lúc một job, dừng app ở cổng 7860 trước.
- Commit kết thúc bằng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

### Task 1: Hàm thuần

**Files:** Modify `simple_app.py`, `test_simple_app.py`

**Produces:**
- `DETAIL_LORA = "detailz-wan.safetensors"`, `DETAIL_MULT = "0;0.6"`
- `_lora_settings(loras, high, low, smooth, detail=False)`; `reels_settings`, `extend_settings`, `multi_settings` nhận thêm `detail=False` và chuyển xuống
- `parse_story(text) -> list[tuple[str, int, str]]`
- `motion_settings(..., audio=False, outfit=False)`
- `detail_instructions(seconds)` có câu cấm zoom
- `eta_text` ra dạng `"Ước tính 35 phút · SVI 2 Pro · 3 cửa sổ × 8 bước"`
- `log_harness(entry: dict, path=HARNESS_LOG)`; `HARNESS_LOG = ROOT / "logs" / "simple_app_harness.log"`

- [ ] **Step 1: Test trước** (thêm vào `test_simple_app.py`, import thêm `DETAIL_LORA, parse_story, log_harness`)

```python
def test_detail_lora():
    s = reels_settings({}, image="a.png", prompt="p", resolution="480x832", seconds=5, seed=1,
                       loras=["x_high_noise.safetensors"], high=1, low=0, detail=True)
    assert s["activated_loras"] == ["x_high_noise.safetensors", DETAIL_LORA]
    assert s["loras_multipliers"] == "1;0 0;0.6"
    assert reels_settings({}, image="a.png", prompt="p", resolution="480x832", seconds=5, seed=1, detail=True)["loras_multipliers"] == "0;0.6"
    own = reels_settings({}, image="a.png", prompt="p", resolution="480x832", seconds=5, seed=1, loras=[DETAIL_LORA], detail=True)
    assert own["activated_loras"] == [DETAIL_LORA]
    assert DETAIL_LORA in multi_settings({}, images=["a.png"], durations=[2], prompts=["w"], common="", resolution="480x832", seed=1, detail=True)["activated_loras"]
    assert DETAIL_LORA in extend_settings({}, video="v.mp4", anchor="a.png", prompt="p", resolution="480x832", seconds=3, seed=1, detail=True)["activated_loras"]


def test_parse_story():
    text = (
        "\ufeffA %x% .png :0–3s:\nAnimate the photo.\nHe smiles.\n\n"
        "b.png: 3-7s:\nHe waves.\n\n"
        "c.jpg : 2s :\nHe sits.\n"
    )
    assert parse_story(text) == [("A %x% .png", 3, "Animate the photo. He smiles"), ("b.png", 4, "He waves"), ("c.jpg", 2, "He sits")]
    for bad in ("a.png :0–6s:\nx", "a.png :3–3s:\nx", "no header here\nx"):
        try:
            parse_story(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_motion_audio_outfit():
    base = dict(image="c.png", video="d.mp4", description="", resolution="480x832", frames=65, seed=1)
    assert motion_settings({}, **base)["postprocess_audio"] == ""
    assert motion_settings({}, **base, audio=True)["postprocess_audio"] == "control"
    o = motion_settings({}, **base, outfit=True)
    assert o["image_prompt_type"] == "S" and o["image_start"] == "c.png"
    assert o["video_prompt_type"] == "UV" and o["image_refs"] is None
    a = motion_settings({}, **base, model=ANIMATE_MODEL, outfit=True)
    assert a["video_prompt_type"] == "PVBKI" and a["image_refs"] == ["c.png"]


def test_detail_instructions_no_zoom():
    assert "no zoom" in detail_instructions(10)


def test_log_harness():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "logs" / "h.log"
        log_harness({"command": "--check-reel a.png 10", "got": "10.06s", "ok": True}, p)
        log_harness({"command": "--check-multi x", "ok": False}, p)
        text = p.read_text(encoding="utf-8")
        assert text.count("=== ") == 2
        assert "--check-reel a.png 10\ngot: 10.06s\nok: True\n" in text
```

Sửa `test_eta_text` thành:

```python
def test_eta_text():
    assert eta_text(EXTEND_MODEL, [81, 81, 81], "480x832") == "Ước tính 35 phút · SVI 2 Pro · 3 cửa sổ × 8 bước"
    assert eta_text(REELS_MODEL, [81], "480x832") == "Ước tính 7 phút · Lightning v2 · 1 cửa sổ × 4 bước"
```

- [ ] **Step 2: Chạy, xác nhận lỗi import** (`DETAIL_LORA`).

- [ ] **Step 3: Cài đặt**

```python
DETAIL_LORA = "detailz-wan.safetensors"  # loras/wan; Wan 2.1 detail enhancer, run on the low-noise phase only
DETAIL_MULT = "0;0.6"


def _lora_settings(loras, high, low, smooth, detail=False):
    names, mults = pair_loras(list(loras or []), high, low)
    if detail and DETAIL_LORA not in names:
        names.append(DETAIL_LORA)
        mults = f"{mults} {DETAIL_MULT}".strip()
    return {"activated_loras": names, "loras_multipliers": mults, "temporal_upsampling": "rife*2" if smooth else ""}
```

`reels_settings`, `extend_settings`, `multi_settings`: thêm tham số `detail=False`, gọi `_lora_settings(loras, high, low, smooth, detail)`.

```python
# "<image> :0–3s:" or "<image>: 3s:" heading each Story block
_STORY_HEAD = re.compile(
    r"^(?P<name>.+?\.(?:png|jpe?g|webp))\s*:\s*(?:(?P<start>\d+(?:\.\d+)?)\s*[–-]\s*)?(?P<end>\d+(?:\.\d+)?)\s*s?\s*:?$",
    re.I,
)


def parse_story(text):
    """[(image name, whole seconds, prompt on one line)] from a Story file, in file order."""
    story = []
    for block in re.split(r"\n\s*\n", str(text).lstrip("\ufeff").strip()):
        head, *body = block.strip().splitlines()
        m = _STORY_HEAD.match(head.strip())
        if not m:
            raise ValueError(f"Story: không đọc được dòng đầu khối: {head!r}")
        seconds = float(m["end"]) - float(m["start"] or 0)
        if seconds != int(seconds) or not 1 <= seconds <= 5:
            raise ValueError(f"Story: {m['name'].strip()}: thời lượng {seconds:g}s phải là số giây nguyên từ 1 đến 5")
        story.append((m["name"].strip(), int(seconds), _one_line(" ".join(body))))
    return story
```

`motion_settings`: thêm `audio=False, outfit=False`; khóa `"postprocess_audio": "control" if audio else ""`; nhánh không phải `ANIMATE_MODEL`:

```python
    elif outfit:
        # no character reference: later windows continue from the previous window, so outfit changes can carry over
        settings.update(image_prompt_type="S", image_start=image, video_prompt_type="UV", image_refs=None)
    else:
        settings.update(image_prompt_type="T", video_prompt_type="UVI")
```

`detail_instructions`: thêm dòng `"The camera keeps the same distance and framing as the image: no zoom, no push-in, no close-up on the face; the person may walk closer by themselves.\n"` sau dòng "Keep the motion natural…".

`eta_text`: `return f"Ước tính {eta} · {MODEL_LABELS[model]} · {len(windows)} cửa sổ × {steps} bước"`.

```python
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
```

- [ ] **Step 4: Test qua.** - [ ] **Step 5: Commit** "Add detail LoRA, story parser, motion audio/outfit options and harness log".

---

### Task 2: Giao diện tối và các checkbox mới

**Files:** Modify `simple_app.py` (thay `CSS`, `FORCE_LIGHT`, thêm `COLORS`, `THEME`; sửa `GUIDE`, `build_ui`, `main`)

- [ ] **Step 1:** Thay `CSS`/`FORCE_LIGHT` bằng `COLORS`, `THEME_VARS`, `THEME`, `FORCE_DARK`, `CSS` đúng như `mockup_v3.py` (bỏ phần swatch; giữ `#guide` ẩn mặc định, nền `surface`, viền `border_strong`, bo 8px).
- [ ] **Step 2:** `GUIDE`: thêm mục "Giữ da & tóc" (Detailz, Pha cuối 0.6, ảnh cận nên 720p), "Âm thanh và trang phục" ở Motion Control, câu Intermediate Video; bỏ mọi emoji.
- [ ] **Step 3:** `build_ui`:
  - `gr.Blocks(title="Simple App", theme=THEME, css=CSS, js=FORCE_DARK)`; header `gr.HTML("<h1>Simple App</h1><p>Reels · Reels Multi Image · Motion Control</p>")` + nút "Hướng dẫn" `variant="secondary"`.
  - Nút phụ dùng `variant="secondary"`; dòng ước tính, thông tin ảnh/video, thời gian thực tế dùng `elem_classes="eta"`; cảnh báo dùng tiền tố "Lưu ý:" và `elem_classes="note"`; `gr.Video(label="Video ra", interactive=False)`.
  - Reels và Multi: `gr.Checkbox(False, label="Giữ da & tóc (LoRA chi tiết, nên dùng cho ảnh cận mặt)")` truyền `detail=` cho `reels_settings`, `extend_settings`, `multi_settings`.
  - Motion: `m_audio = gr.Checkbox(True, label="Giữ âm thanh video mẫu")`, `m_outfit = gr.Checkbox(False, label="Trang phục theo video mẫu")`; `make_motion` truyền `audio=m_audio, outfit=m_outfit`; `motion_eta` thêm "Lưu ý: Wan 2.2 Animate 14B luôn giữ trang phục theo ảnh." khi tick với model 14B.
  - `generate()`: sau thành công, `removed = len(windows) - 1`; nếu `removed > 0` gọi `gr.Info(f"Đã xóa {removed} video trung gian, chỉ giữ video cuối.")` và thêm `· đã xóa {removed} video trung gian` vào dòng thực tế.
- [ ] **Step 4:** `main()`: sau `init(...)`, `sys.modules["wgp"].server_config["keep_intermediate_sliding_windows"] = 0` (comment: chỉ trong bộ nhớ, Wan2GP tự thay file Window trước).
- [ ] **Step 5:** Test thuần + `py_compile`; chạy app, kiểm bằng screenshot headless Edge 1440px ba tab, dialog, checkbox; log không có traceback.
- [ ] **Step 6: Commit** "Restyle the app with the dark theme and add detail, audio and outfit options".

---

### Task 3: Harness mới và chạy GPU

**Files:** Modify `simple_app.py` (`run_check`, thêm `_has_audio`, `contact_sheet`), spec (bảng Kết quả đo)

- [ ] **Step 1:** `run_check` dùng `argparse` (`inputs` nargs="*", `--detail`, `--outfit`, `--seed` int mặc định -1):
  - `--check-reel <ảnh> [giây] [file prompt]`: có file prompt thì đọc làm prompt, không gọi Qwen; mốc ảnh ghép `0,3,6,9` lọc `< giây`.
  - `--check-multi <thư mục>`: `parse_story(story.txt)`, ảnh = thư mục / tên, `multi_settings(..., common="", detail=args.detail)`; mốc = đầu mỗi Segment + cuối.
  - `--check-motion <ảnh> <video> [animate2|animate] [giây]`: `audio=True`, `outfit=args.outfit`; mốc `0,8,11,13,15` lọc `< giây`; kiểm `_has_audio(out) == _has_audio(video)`.
  - Mỗi lần: đo thời gian, `check_duration`, `contact_sheet` (ảnh gốc + frame, lưu `logs/<stamp>_<loại>.jpg`, kèm PNG từng mốc), `log_harness` với model, resolution, seed, detail_lora/outfit, expected, got, ok, took, intermediate_removed (= số Window − 1), audio, video, sheet.
- [ ] **Step 2:** Test thuần + `py_compile`; commit "Harness: prompt file, story, outfit, audio check and log".
- [ ] **Step 3:** Chạy GPU theo bảng Kiểm thử của spec (1)→(4), đọc ảnh ghép, so da/tóc (1) vs (2), áo ở (4).
- [ ] **Step 4:** Ghi bảng "Kết quả đo" vào spec; commit "Record v3 harness runs".
