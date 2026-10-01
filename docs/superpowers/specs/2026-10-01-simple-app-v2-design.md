# Simple App v2: Reels 2–15 s, Reels Multi Image, Motion Control đúng thời lượng

Ngày: 2026-10-01. Nhánh: `20260930_simple_app`. Nối tiếp [spec v1](2026-09-30-simple-app-design.md); phần nào không nhắc ở đây thì giữ như v1 (giữ nhân vật, gợi ý prompt, tiến độ, xử lý lỗi).

## Mục tiêu

1. Giao diện ba tab theo mockup người dùng (`Wangp (1).zip`): **Reels**, **Reels Multi Image**, **Motion Control**, header có nút **Hướng dẫn**.
2. Reels dài 2–15 giây, có ước tính thời gian chạy.
3. Tab mới: tối đa 6 ảnh của cùng một nhân vật, mỗi ảnh có prompt và thời lượng riêng, ra một video liên tục.
4. Motion Control ra video dài đúng bằng video mẫu, trừ khi người dùng tick "Giới hạn thời lượng".

Tiêu chí xong: hai harness ở mục Kiểm thử chạy qua trên máy người dùng, video ra đúng thời lượng (±0.2 s).

## Giới hạn

- Vẫn một file `simple_app.py`, test trong `test_simple_app.py`. Không sửa `wgp.py`, không thêm thư viện.
- Repo không có `AGENTS.md` / `CLAUDE.md`. Theo `wangp-agent/SKILL.md`: chỉ đổi settings cần đổi, giữ mỗi cửa sổ trong `metadata.frames_maximum`, trong chế độ `PW` không để dòng trống bên trong một cửa sổ, báo lỗi ra ngoài chứ không lặng lẽ đổi model.
- Máy: RTX 5070 Ti Laptop 12 GB, profile 4, mỗi lúc một job GPU. Không tải model mới (Upscale chỉ dùng Lanczos).

## Ngoài phạm vi

Upscale bằng AI (SeedVR2 7.3 GB, FlashVSR thiếu `triton`), nút tải LoRA trong app, "Viết chi tiết" cho từng ảnh ở Multi, sắp xếp lại ảnh bằng kéo thả, harness cho Multi (viết sau), nút bánh răng.

## Giao diện chung

- Theme `gr.themes.Soft(primary_hue="orange")`, nền cam nhạt, header cam; ép chế độ sáng bằng `js` của `gr.Blocks` (bảng màu chỉ thiết kế cho nền sáng).
- Header: tiêu đề và nút **Hướng dẫn**. Bấm mở một `gr.Column` ẩn được CSS đặt nổi giữa màn hình (Gradio 5.29 không có modal), có nút Đóng. Nội dung (Markdown, tiếng Việt):
  - cách mỗi tab hoạt động và model dùng;
  - **Pha đầu / Pha cuối**: Wan 2.2 khử nhiễu bằng hai model nối tiếp. Pha đầu (high noise) là các bước đầu, quyết định bố cục và chuyển động lớn. Pha cuối (low noise) là các bước sau, quyết định chi tiết: mặt, da, tóc, vải. Thanh trượt là cường độ LoRA trong từng pha; Pha cuối = 0 thì LoRA chỉ tác động lên chuyển động, mặt giữ theo ảnh gốc;
  - nguồn model (mục Tham khảo);
  - LoRA tải được (lấy từ `D:\001_Personal_Proj\wan2-2-i2v-v3\lora_loader.py`), liệt kê theo repo: `lkzd7/WAN2.2_LoraSet_NSFW`, `Lythiga/WAN2.2_I2V_Lora`, `rolechar/rc_wan2.2_i2v_loraset_nsfw_private` (riêng tư, cần `HF_TOKEN`), `lopi999/Wan2.2-I2V_General-NSFW-LoRA`. Cách cài: tải cả cặp high/low vào `loras/wan`, chọn cả hai file trong dropdown, để Pha cuối = 0.
- Mỗi tab: cột trái nhập liệu; cột phải gồm hàng [Hủy] [Tạo Video], dòng **⏱ ước tính**, video ra, tiến độ, nút phụ. Cài đặt nâng cao nằm trong `gr.Accordion`.
- Nút **Upscale ×2** ở cả ba tab: `session.submit_media_postprocessing(video, spatial_upsampling="lanczos*2")` trên video đang hiển thị; kết quả thay video hiển thị. Ở Reels, `gr.State` của nút Kéo dài vẫn giữ video chưa upscale. Bị vô hiệu cho đến khi có video.
- `run()` của v1 nhận thêm cách gửi job (`submit_task` hoặc `submit_media_postprocessing`); phần callback tiến độ giữ nguyên.

## Ước tính thời gian

Hàm thuần `estimate_seconds(model, windows, resolution, calib)`, với `windows` là danh sách số frame mỗi cửa sổ:

```
LOAD + Σ_cửa sổ  steps(model) × SEC_PER_STEP[model] × (frames × w × h) / (REF_FRAMES[model] × REF_PIXELS[model]) × calib.get(model, 1)
```

Mốc lấy từ số đo ngày 2026-09-30 (spec v1):

| Model | Bước | Giây/bước | Mốc |
|---|---|---|---|
| `i2v_2_2_Enhanced_Lightning_v2` | 4 | 97 | 81 frame, 720×720 |
| `..._svi2pro` | 8 | 106 | 81 frame, 720×720 |
| `animate2_distilled` | 10 | 94 | 65 frame, 480×848 |
| `animate` + FusionX | 10 | 94 (chưa đo, mượn của Animate 2) | 65 frame, 480×848 |

`LOAD` = 120 giây (nạp model + giải mã). Hiển thị dạng "⏱ ≈ 35 phút (SVI 2 Pro · 3 cửa sổ × 8 bước)", cập nhật khi đổi thời lượng, chất lượng, ảnh, video mẫu, số ảnh hoặc model.

Hiệu chỉnh: sau mỗi job thành công, app đo thời gian thật và ghi `calib[model] = thật / ước tính_không_calib` vào `simple_app_timings.json` (gốc repo, thêm vào `.gitignore`). Hiện thêm "Thực tế: 27 phút" dưới video. File thiếu hoặc hỏng thì dùng hệ số 1.

> ponytail: thời gian coi như tuyến tính theo frame × điểm ảnh (attention thực ra tăng nhanh hơn); hệ số calib bù phần lệch cho cỡ video hay dùng.

Ước tính ở 480p, ảnh dọc `480x832`, trước khi hiệu chỉnh: Reels 5 s ≈ 7 phút, Reels 15 s ≈ 35 phút, Multi 3 ảnh × 3 s ≈ 11 phút, Motion 27 s ≈ 1.8 giờ.

## Tab Reels

Khác v1:

- Ảnh đầu vào kèm thẻ chỉ đọc: `Filename · gốc 968×1216 → video 480×832`.
- Ý ngắn, nút **3 ý tưởng**, dropdown chọn ý (thay radio), nút **Viết chi tiết**, ô Prompt.
- Cài đặt nâng cao: thời lượng **2–15 giây** (mặc định 5, bước 1), 480p/720p, Smooth x2, LoRA, Pha đầu (1), **Pha cuối (0)**, Seed.
- Giữ nút **Kéo dài video (neo ảnh gốc)** như v1.

### Chọn model theo thời lượng

Hàm thuần `reel_plan(seconds)` → `(model_type, durations)`:

- `seconds ≤ 5`: `REELS_MODEL` (Lightning v2, 4 bước), một cửa sổ, settings như v1 (`video_length = frames_for(seconds)`, prompt nguyên văn).
- `seconds > 5`: `EXTEND_MODEL` (SVI 2 Pro, 8 bước) trong **một job**, chia thành `k = ceil(seconds / 5)` cửa sổ, thời lượng nguyên giây chia đều nhất có thể (7 → 4+3, 11 → 4+4+3, 15 → 5+5+5).

Settings khi `seconds > 5`, xây trên `session.get_default_settings(EXTEND_MODEL)`:

```python
{
    "model_type": EXTEND_MODEL,
    "prompt": split_timeline(prompt, durations),
    "multi_prompts_gen_type": "PW",
    "image_prompt_type": "S",
    "image_start": image,
    "video_prompt_type": "KI",
    "image_refs": [image],              # neo ảnh gốc cho mọi cửa sổ (window N dùng image_refs[min(N, len)-1])
    "resolution": pick_resolution(w, h, quality),
    "video_length": seconds * 16 + 1,   # bị Wan2GP ghi đè bằng tổng các [/duration]
    "seed": seed,
    **_lora_settings(loras, high, low, smooth),
}
```

Mỗi đoạn `[/duration=Ns]` ra đúng `N × 16` frame mới (cửa sổ đầu 81 frame, các cửa sổ sau 85 frame gồm 4 frame chồng lấn), nên tổng là `seconds × 16 (+1)` frame, đúng thời lượng.

### Tách prompt theo cửa sổ

Hàm thuần `split_timeline(prompt, durations)` → chuỗi các đoạn ngăn bởi một dòng trống, mỗi đoạn mở đầu bằng `[/duration=Ns]`.

- Dòng có mốc thời gian, chấp nhận: `(at 3 seconds: …)`, `(0-2 seconds: …)`, `(at 0–2 seconds: …)`, `(from 0 to 2 seconds: …)`, đơn vị `seconds|second|s|giây`. Mốc đơn `X` hiểu là khoảng `[X, X+1)`.
- Cửa sổ thứ k phủ `[t0, t1)`. Dòng nào giao với khoảng này thì vào cửa sổ đó, mốc được trừ `t0` và cắt trong `[0, t1-t0]`. Dòng vắt qua ranh giới vào cả hai cửa sổ. Ghi lại theo dạng gốc: mốc đơn → `(at X seconds: …)`, khoảng → `(X-Y seconds: …)`.
- Dòng không có mốc (ví dụ dòng mô tả nhân vật) được giữ ở đầu mọi cửa sổ.
- Prompt không có dòng nào có mốc: mọi cửa sổ nhận nguyên prompt.
- Bỏ mọi dòng trống bên trong một đoạn (luật của `PW`).

### Gợi ý prompt

`IDEAS_INSTRUCTIONS` và `DETAIL_INSTRUCTIONS` thành hàm của số giây: "one continuous {N} second shot", "one line per second, from 0 to {N} seconds". Nút gợi ý truyền giá trị thanh thời lượng hiện tại.

## Tab Reels Multi Image

Model: `REELS_MODEL` (Lightning v2, 4 bước). Mỗi đoạn kết thúc đúng ở ảnh kế tiếp, nên chính các ảnh giữ nhân vật.

### Giao diện

- Thẻ "Load multi Image": bộ đếm `N/6 ảnh · tổng X giây`, ô **Mô tả nhân vật chung** (ghép vào đầu mọi prompt), 6 ô ảnh dựng sẵn, ban đầu hiện 2.
- Mỗi ô: ảnh, thanh **Thời lượng đoạn** 1–5 giây (mặc định 3), Prompt, nút **Xóa**. Xóa ô i thì dồn nội dung các ô sau lên một bậc và ẩn ô cuối. Nút **+ Thêm ảnh** hiện thêm một ô (tối đa 6).
- Cài đặt nâng cao: 480p/720p (theo ảnh 1), Smooth x2, LoRA, Pha đầu, Pha cuối (0), Seed.
- Cột phải: Hủy / Tạo Video, ước tính, video ra, tiến độ, Upscale. Không có Kéo dài.

### Ý nghĩa

Ảnh i là mốc. Đoạn i bắt đầu ở ảnh i, dài bằng thời lượng ghi trên ảnh i, và kết thúc ở ảnh i+1. Đoạn của ảnh cuối không có ảnh đích, chạy tự do. N ảnh → N đoạn.

### Settings

Hàm thuần `multi_settings(defaults, *, images, durations, prompts, common, resolution, seed, loras, high, low, smooth)`:

```python
{
    "model_type": REELS_MODEL,
    "image_prompt_type": "SE" if len(images) > 1 else "S",
    "image_start": images[0],
    "image_end": images[1:],            # cửa sổ k kết thúc ở image_end[k-1]; cửa sổ N không có ảnh đích
    "multi_prompts_gen_type": "PW",
    "prompt": "\n\n".join(f"[/duration={d}s] {join(common, p)}" for d, p in zip(durations, prompts)),
    "resolution": resolution,
    "video_length": sum(durations) * 16 + 1,
    "seed": seed,
    **_lora_settings(loras, high, low, smooth),
}
```

`join(common, p)` nối hai phần bằng ". ", bỏ phần rỗng, và gộp các dòng thành một dòng (luật `PW`). Đã xác nhận bằng đọc code (`wgp.py:1453-1517`, `7559-7570`, `shared/utils/frame_scheduler.py`): API chấp nhận một ảnh đầu + nhiều ảnh cuối ở chế độ `PW`, mỗi `[/duration]` thành đúng một cửa sổ.

### Kiểm tra trước khi gửi

- Ô đang hiện mà chưa có ảnh → `gr.Error`.
- Ô có cả prompt riêng và mô tả chung đều rỗng → `gr.Error`.
- Ảnh có tỉ lệ khung lệch quá 10% so với ảnh 1 → cảnh báo trong dòng ước tính (Wan2GP tự co và cắt), vẫn cho chạy.

## Tab Motion Control

### Giao diện

Ảnh nhân vật, Video tham chiếu, thẻ chỉ đọc `27.1 giây · 30 fps · 720×1280 → 433 frame ở 16 fps`, mô tả nhân vật (không bắt buộc). Cài đặt nâng cao:

- Checkbox **Giới hạn thời lượng** (mặc định tắt). Thanh 1 → min(15, độ dài video mẫu) giây **chỉ bật khi tick**; cắt từ đầu video mẫu.
- 480p/720p, Smooth x2 (16 → 32 fps), Seed.
- **Chọn Model**: `Animate 2 distilled` (mặc định) | `Wan 2.2 Animate 14B`.

### Thời lượng

Hàm thuần `motion_frames(src_fps, src_frames, limit=None)`:

```
seconds = src_frames / src_fps, hoặc min(limit, seconds) nếu có giới hạn
frames  = round_frames(seconds × 16)
```

Video 27.1 s → 433 frame = 27.06 s. Wan2GP lấy mẫu lại video mẫu về 16 fps theo mốc thời gian (bỏ hoặc lặp frame, giữ thời lượng; `shared/utils/video_decode.py:65-85`). Animate 2 có `control_video_trim`, nên nếu dư frame thì dừng khi hết video mẫu.

### Settings

Animate 2: như v1, khác `force_fps: "16"` (thay `"control"`), `video_length = motion_frames(...)`, `temporal_upsampling = "rife*2" if smooth else ""`. Cửa sổ 65, chồng lấn 1: số cửa sổ = `1 + ceil((N - 65) / 64)`, 433 frame → 7 cửa sổ.

Wan 2.2 Animate 14B (`model_type="animate"`, file `ckpts/wan2.2_animate_14B_quanto_bf16_int8.safetensors` đã có):

```python
{
    "model_type": "animate",
    "prompt": description or MOTION_DEFAULT_PROMPT,
    "image_prompt_type": "",
    "video_prompt_type": "PVBKI",       # Animate Person in Reference Image using Motion of Whole Control Video
    "image_refs": [image],
    "video_guide": video,
    "activated_loras": ["Wan2.1_I2V_14B_FusionX_LoRA.safetensors"],   # có sẵn trong loras/wan_i2v
    "loras_multipliers": "1",
    "num_inference_steps": 10, "guidance_phases": 1, "guidance_scale": 1,   # theo profile "Image2Video FusioniX - 10 Steps"
    "sliding_window_size": 81,          # chồng lấn mặc định 5
    "resolution": ..., "video_length": ..., "force_fps": "16", "seed": seed,
    "temporal_upsampling": "rife*2" if smooth else "",
}
```

Số cửa sổ = `1 + ceil((N - 81) / 76)`; 433 frame → 6 cửa sổ. LoRA relighting không dùng (chỉ cho chế độ Replace).

## Kiểm thử

`test_simple_app.py` (không GPU, không import `wgp`) thêm assert cho:

- `reel_plan`: 2 và 5 → Lightning, một cửa sổ; 6 → SVI `[3, 3]`; 7 → `[4, 3]`; 15 → `[5, 5, 5]`.
- `split_timeline`: mốc đơn, khoảng `0-2`, dòng vắt qua ranh giới vào cả hai cửa sổ, dòng không mốc lặp ở mọi cửa sổ, prompt không mốc, không còn dòng trống trong đoạn, mỗi đoạn mở đầu `[/duration=Ns]`.
- `reels_settings` khi > 5 s (model, `PW`, `KI`, `image_refs`).
- `multi_settings`: 1 ảnh (`S`, không `image_end`), 3 ảnh (`SE`, 2 ảnh cuối, 3 đoạn), ghép mô tả chung.
- `motion_frames`: không giới hạn (813 frame @ 30 fps → 433), có giới hạn 5 s → 81.
- `motion_settings` cho cả hai model, `force_fps == "16"`.
- `estimate_seconds`: tăng theo số cửa sổ và độ phân giải; `calib` nhân đúng.

Harness không cần UI, assert thời lượng video ra (`frame / fps` đọc bằng `cv2`) lệch không quá 0.2 s:

```
python simple_app.py --check-reel  "D:\001_Personal_Proj\Comfy\PostIG\day_53_1\Ryo\ZImage_2026_06_06_101818_60_00001.png" 15
python simple_app.py --check-motion "D:\...\motionControl\day_27_1\Klein_I2I_260920_77_00001_.png" "D:\...\motionControl\day_27_1\vietnameseboy__1788933546_3982185246061537055_27770465737.mp4"
```

- `--check-reel <ảnh> [giây]` (mặc định 5): tự viết prompt N giây rồi sinh; mong đợi 15.0 s (ảnh 968×1216 → `480x832`), khoảng 35 phút.
- `--check-motion <ảnh> <video> [animate2|animate]`: toàn bộ video mẫu; mong đợi 27.06 s so với 27.10 s, khoảng 1.8 giờ, chạy sau cùng.
- Harness Multi: viết sau khi người dùng có bộ ảnh mẫu.

Chạy tay: mở app, kiểm tra bố cục ba tab, dialog Hướng dẫn, checkbox giới hạn bật/tắt thanh, thêm/xóa ảnh ở Multi, Upscale một video.

## Rủi ro

1. **Cửa sổ SVI 85 frame** vượt `frames_maximum` 81 một chút (5 s + 4 frame chồng lấn). Nếu chất lượng cửa sổ sau kém đi, đổi cách chia thành tối đa 4.75 s mỗi cửa sổ.
2. **Animate 14B + FusionX chưa chạy thử** trên máy này. FusionX là LoRA Wan 2.1 I2V; changelog cho biết Animate nhận LoRA tăng tốc của i2v. Giây/bước đang mượn của Animate 2; hệ số calib sẽ sửa sau lần chạy đầu.
3. **Multi chưa chạy thử**: luồng `SE` + `PW` + `[/duration]` mới chỉ xác nhận bằng đọc code. Chồng lấn 1 frame của Lightning có thể làm chuyển động giật ở chỗ nối; ảnh đích ở mỗi đoạn giúp nhân vật không trôi.
4. **Motion 27 s** chiếm GPU khoảng 1.8 giờ; trong lúc đó app không làm việc khác.

## Tham khảo

Trong repo:
- `docs/PROCESSING.md`: sliding window, trôi nhân vật, khuyên dùng ảnh cuối làm mốc.
- `docs/PROMPTS.md:242-279`, `shared/utils/frame_scheduler.py`, `shared/utils/prompt_parser.py`: `PW`, `[/duration]`.
- `wgp.py:1440-1520` (kiểm tra ảnh đầu/cuối), `7499` (prompt theo cửa sổ), `7559-7570` (ảnh cuối theo cửa sổ), `2908-2924` (`force_fps`).
- `models/wan/any2video.py:709-730`: neo SVI theo cửa sổ.
- `models/wan/wan_handler.py:769-778` (`PVBKI`), `998-999` (chồng lấn SVI).
- `shared/api.py:794`: `submit_media_postprocessing`. `postprocessing/spatial_upsamplers.py`: Lanczos.
- `profiles/wan_i2v/Image2Video FusioniX - 10 Steps.json`, `defaults/animate.json`.
- `wangp-agent/SKILL.md`: luật khi gọi Wan2GP.

Mockup người dùng: `C:\Users\tonyn\Downloads\Wangp (1).zip` (3 SVG).
