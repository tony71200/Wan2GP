# Simple App: Reels + Motion Control

Ngày: 2026-09-30. Nhánh: `20260930_simple_app`.

## Mục tiêu

Một app Gradio một màn hình, hai tab, chạy trên máy RTX 5070 Ti Laptop 12GB, dùng lại toàn bộ phần nạp model và sinh video của Wan2GP:

1. **Reels**: ảnh → video ngắn, có gợi ý prompt từ ảnh.
2. **Motion Control**: video mẫu có người → nhân vật trong ảnh chuyển động theo (kiểu Kling Motion Control).

Yêu cầu quan trọng xuyên suốt: **nhân vật trong ảnh không bị biến thành người khác**. Xem mục "Giữ nhân vật".

Tiêu chí xong: mỗi tab sinh ra được một video từ đầu đến cuối trên máy người dùng mà không mở UI lớn của Wan2GP.

## Ngoài phạm vi bản đầu

Wan-Move (vẽ quỹ đạo), chế độ thay nhân vật trong video, tự bật LoRA theo prompt, upscale, âm thanh, hàng đợi nhiều job, H3 Face Refiner (giai đoạn 2), xóa model (người dùng tự quyết).

## Kiến trúc

- `simple_app.py` ở gốc repo: toàn bộ UI và logic.
- `run_simple_app.bat`: kích hoạt env conda `wan2gp` rồi chạy `python simple_app.py` (theo mẫu `run_wgp.bat`).
- `test_simple_app.py`: kiểm thử các hàm thuần.
- Không sửa `wgp.py`. Không thêm thư viện (Gradio 5.29 đã có trong `requirements.txt`).

App gọi `shared.api.init(root=<thư mục repo>, cli_args=["--profile", "4"])` một lần khi khởi động. Mỗi lần bấm "Tạo video" là một `session.submit_task(settings)`; tiến độ lấy qua callback `on_status` / `on_progress` và đẩy vào `gr.Progress`. Nút Hủy gọi `job.cancel()`. Mỗi lúc chỉ chạy một job.

## Bố cục giao diện

Hai tab, mỗi tab hai cột: cột trái nhập liệu, cột phải video kết quả.

```
[ Reels ] [ Motion control ]
+-----------------------------+---------------------------+
| Ảnh đầu vào                 |                           |
| Ý ngắn: [...............]   |      Video kết quả        |
| [3 ý tưởng] [Viết chi tiết] |                           |
| ( ) ý 1  ( ) ý 2  ( ) ý 3   |                           |
| Prompt: [................]  | tiến độ                   |
| Thời lượng 2–5 s            | [Kéo dài video]           |
| (480p | 720p)  [x] Mượt x2  |                           |
| LoRA [chọn nhiều]           |                           |
| Pha đầu 1.0   Pha sau 0     |                           |
| Seed  [Tạo video] [Hủy]     |                           |
+-----------------------------+---------------------------+

Tab Motion control: [Ảnh nhân vật] [Video mẫu], mô tả nhân vật,
giới hạn thời lượng 2–10 s, (480p | 720p), seed, [Tạo video] [Hủy],
cảnh báo tải 17 GB nếu thiếu model; cột phải là video kết quả.
```

## Các hàm thuần

| Hàm | Vào | Ra |
|---|---|---|
| `pick_resolution(w, h, quality)` | kích thước ảnh, `"480p"` hoặc `"720p"` | xem bảng dưới |
| `frames_for(seconds, fps=16)` | số giây | số frame dạng 4k+1 gần nhất, kẹp trong [17, 81] (5 giây → 81) |
| `round_frames(n)` | số frame bất kỳ | làm tròn xuống dạng 4k+1, tối thiểu 5 |
| `pair_loras(names, high, low)` | tên LoRA đã chọn, cường độ pha đầu, cường độ pha sau | `(activated_loras, loras_multipliers)` |
| `parse_ideas(text)` | đầu ra LLM | danh sách tối đa 3 ý, bỏ số thứ tự và dòng trống |

`pick_resolution`: tỉ lệ `r = w / h`.

| | r < 0.9 (dọc) | 0.9 ≤ r ≤ 1.1 (vuông) | r > 1.1 (ngang) |
|---|---|---|---|
| 480p | `480x832` | `720x720` | `832x480` |
| 720p | `720x1280` | `960x960` | `1280x720` |

`pair_loras`: với mỗi tên file, không phân biệt hoa thường:
- có từ `high` hoặc `highnoise` đứng riêng (không dính chữ cái liền trước/sau), hoặc `_hn`, hoặc `_h.` → hệ số `"{high};0"`;
- có từ `low` hoặc `lownoise` đứng riêng, hoặc `_ln`, hoặc `_l.` → hệ số `"0;{low}"`;
- còn lại → `"{high};{low}"` (ví dụ `slow_motion.safetensors` không bị coi là file pha sau).

`loras_multipliers` là các hệ số nối bằng dấu cách, cùng thứ tự với `activated_loras`. Dropdown hiển thị từng file riêng; người dùng tự chọn cả hai file của một cặp.

## Tab Reels

Model: `i2v_2_2_Enhanced_Lightning_v2` (đã có trên đĩa).

Thành phần UI: ảnh đầu vào, ô "ý ngắn", nút **3 ý tưởng**, radio 3 ý, nút **Viết chi tiết**, ô prompt, thanh thời lượng 2–5 giây (mặc định 5), radio chất lượng 480p/720p (mặc định 480p), tick **Mượt x2** (mặc định bật), dropdown LoRA nhiều lựa chọn, thanh **Pha đầu** (0–2, mặc định 1), thanh **Pha sau** (0–2, mặc định 0), seed (-1 là ngẫu nhiên), nút Tạo video, nút Hủy, video kết quả, nút **Kéo dài video**.

Settings gửi đi, xây trên `session.get_default_settings(model_type)`:

```python
{
    "model_type": "i2v_2_2_Enhanced_Lightning_v2",
    "prompt": prompt,
    "image_prompt_type": "S",
    "image_start": image_path,
    "resolution": pick_resolution(w, h, quality),
    "video_length": frames_for(seconds),
    "seed": seed,
    "activated_loras": [...],
    "loras_multipliers": "...",
    "temporal_upsampling": "rife*2" if smooth else "",
}
```

Danh sách LoRA lấy từ `session.list_loras("i2v_2_2_Enhanced_Lightning_v2")` (thư mục `loras/wan`), nạp lại mỗi lần mở dropdown.

Tốc độ khung hình: model sinh ở 16 fps; "Mượt x2" dùng RIFE cho ra 32 fps mà không đổi thời lượng.

### Kéo dài video

Nút **Kéo dài video** nối thêm một đoạn vào video vừa sinh, dùng model `i2v_2_2_Enhanced_Lightning_v2_svi2pro` (cùng hai file model đang có, cộng hai LoRA SVI 2 Pro, tổng 2.45 GB, Wan2GP tự tải lần đầu; 8 bước). Ảnh đầu vào gốc làm ảnh neo cho mọi đoạn nối.

```python
{
    "model_type": "i2v_2_2_Enhanced_Lightning_v2_svi2pro",
    "prompt": prompt,
    "image_prompt_type": "V",
    "video_source": last_raw_video_path,
    "video_prompt_type": "KI",
    "image_refs": [original_image_path],
    "resolution": <như lần sinh trước>,
    "video_length": frames_for(seconds) - 4,   # trừ 4 frame chồng lấn của SVI để chỉ sinh một cửa sổ
    "seed": seed,
    "activated_loras": [...], "loras_multipliers": "...",
    "temporal_upsampling": "rife*2" if smooth else "",
}
```

App giữ đường dẫn video vừa sinh và ảnh gốc trong `gr.State`. Nút bị vô hiệu cho đến khi có video. Kết quả thay video hiện tại, nên bấm nhiều lần sẽ nối dài dần.

## Gợi ý prompt từ ảnh

Dùng Qwen3.5-4B Abliterated có vision (đã có trên đĩa) qua hàm nội bộ của `wgp.py`.

Cả hai nút đi qua một hàm `enhance(session, model_type, text, image, instructions=None, progress=None)`, hàm này gọi `wgp.process_prompt_enhancer` ở chế độ `"TI"` và bao quanh bằng cùng chuỗi nạp/nhả mà `exec_prompt_enhancer_engine` dùng (`wait_for_model_unload`, `acquire_GPU_ressources`, `ensure_prompt_enhancer_loaded`, rồi nhả sau khi xong).

- **3 ý tưởng**: `model_type=""` (không dùng chỉ dẫn của model) và chỉ dẫn riêng yêu cầu đúng 3 ý tưởng video ngắn khác nhau, mỗi ý một dòng, một câu, phù hợp với ảnh và giữ nguyên nhân vật. Kết quả qua `parse_ideas` rồi hiện thành radio. Chọn một ý thì điền vào ô "ý ngắn".
- **Viết chi tiết**: cũng `model_type=""`, với chỉ dẫn riêng `DETAIL_INSTRUCTIONS`: một cú máy liên tục, không chuyển cảnh, mỗi giây một dòng dạng `(at X seconds: ...)`, dòng nào cũng nhắc lại cùng nhân vật và bối cảnh, chỉ đổi hành động và chuyển động máy. Đầu vào là ảnh và ô "ý ngắn". Kết quả điền vào ô prompt, người dùng sửa được.

Không dùng chỉ dẫn có sẵn trong định nghĩa model Enhanced Lightning v2: nó khuyến khích đổi góc máy và chuyển cảnh theo từng giây. Thử ngày 2026-09-30 trên ảnh mẫu, prompt sinh từ chỉ dẫn đó làm nhân vật đổi tóc và đổi bối cảnh ngay giữa video; với `DETAIL_INSTRUCTIONS` thì người và bối cảnh giữ nguyên qua cả 5 giây.

Hàm cần `state` của session (`session._state`, thuộc tính riêng tư) và module `wgp` đã nạp. `python simple_app.py --check-enhancer <ảnh>` chạy riêng phần này để kiểm tra.

## Tab Motion Control

Model: `animate2_distilled` (tải 17 GB ở lần chạy đầu; app hiện cảnh báo trong tab nếu `session.get_model_availability` báo thiếu file).

Thành phần UI: ảnh nhân vật, video mẫu, ô mô tả nhân vật (mặc định lấy từ định nghĩa model), thanh giới hạn thời lượng 2–10 giây (mặc định 5), radio 480p/720p (mặc định 480p), seed, Tạo video, Hủy, video kết quả.

Settings, xây trên `session.get_default_settings("animate2_distilled")`. Ảnh nhân vật được gửi ở chế độ **Character Reference Image** để mọi cửa sổ 65 frame đều tham chiếu ảnh gốc:

```python
{
    "model_type": "animate2_distilled",
    "prompt": character_description,
    "image_prompt_type": "T",
    "video_prompt_type": "UVI",
    "image_refs": [character_image_path],
    "video_guide": driving_video_path,
    "resolution": pick_resolution(w, h, quality),   # theo ảnh nhân vật
    "video_length": round_frames(min(số frame video mẫu, giới hạn giây × fps video mẫu)),
    "force_fps": "control",
    "seed": seed,
}
```

Tốc độ khung hình theo video mẫu (`force_fps: "control"`). Video mẫu 30 fps tốn gần gấp đôi số frame so với 16 fps cho cùng thời lượng.

Khi ô mô tả để trống, app gửi `MOTION_DEFAULT_PROMPT` (cùng người, cùng bối cảnh như ảnh tham chiếu). Không dùng prompt mặc định của model: đó là prompt mẫu tả một con mèo mặc đồng phục.

## Giữ nhân vật

Không có LoRA dùng chung nào khóa được khuôn mặt cho mọi nhân vật. Bản đầu dùng bốn cơ chế, không cái nào cần model mới ngoài 2.45 GB LoRA SVI:

| # | Cơ chế | Tab | Mặc định |
|---|---|---|---|
| 1 | Ảnh nhân vật ở chế độ tham chiếu (`video_prompt_type` có `I`) | Motion Control | luôn bật |
| 2 | Lựa chọn 720p để mặt có nhiều điểm ảnh hơn | cả hai | 480p |
| 3 | LoRA tách hai pha; pha sau (quyết định chi tiết khuôn mặt) mặc định 0 | Reels | pha sau = 0 |
| 4 | Nối clip bằng SVI 2 Pro neo về ảnh gốc, thay cho lấy frame cuối làm ảnh đầu | Reels | nút Kéo dài video |

**Giai đoạn 2 (chỉ làm nếu mặt vẫn trôi sau khi thử bản đầu):** thêm tick "Sửa mặt theo ảnh gốc" cho cả hai tab, dùng H3 Face Refiner của Wan2GP (`spatial_upsampling` + `spatial_upsampler_reference_images` = ảnh gốc). Cần tải MiniMax H3 Ref2VA pruned, 21 GB. Phương án thay model cho Motion Control nếu cần: SCAIL-2 (16.6 GB, 40 bước).

Không dùng: Lynx, Stand-In, Phantom (chuyển khuôn mặt cho text-to-video Wan 2.1, không bắt đầu từ ảnh người dùng).

## Tiến độ

Wan2GP gọi callback tiến độ từ luồng job, nơi Gradio không có ngữ cảnh sự kiện. `run()` chụp ngữ cảnh bằng `contextvars.copy_context()` trong handler và chạy callback bên trong ngữ cảnh đó; nếu không, thanh tiến độ đứng yên suốt lúc sinh video.

## Xử lý lỗi

- Thiếu ảnh, thiếu video mẫu, hoặc prompt Reels rỗng: `gr.Error` trước khi gửi job.
- Job thất bại: hiện `result.errors[0].message`.
- Job bị hủy: giữ nguyên video cũ, không báo lỗi.
- Lỗi khi gợi ý prompt: nhả GPU rồi hiện lỗi; ô prompt giữ nguyên.

## Kiểm thử

- `test_simple_app.py`: assert cho `pick_resolution`, `frames_for`, `round_frames`, `pair_loras`, `parse_ideas`. Chạy bằng `python test_simple_app.py`, không cần GPU và không import `wgp`.
- Chạy tay trên máy người dùng: một video Reels (có và không có gợi ý), một lần Kéo dài video, một video Motion Control. Ở mỗi video, so khuôn mặt frame đầu và frame cuối với ảnh gốc.

## Rủi ro và phương án dự phòng

1. **Gợi ý prompt ở chế độ độc lập.** API không có hàm công khai; app dựa vào hàm nội bộ và `session._state`. Bước đầu tiên của kế hoạch là thử riêng lời gọi này. Nếu không chạy được, chuyển app thành plugin tab (`plugins/wan2gp-simple-app/`) dùng `api_session` và `state` do Wan2GP cấp; UI và các hàm thuần giữ nguyên. **Kết quả 2026-09-30:** chạy được ở chế độ độc lập (`--check-enhancer`, và qua endpoint của app); không cần phương án plugin.
2. **Đổi model trên 12GB.** Qwen và model video không nằm cùng lúc trên GPU; mỗi lần gợi ý rồi sinh video sẽ tốn thời gian nạp lại. Chấp nhận ở bản đầu. **Kết quả 2026-09-30:** đổi qua lại Qwen ↔ model video không lỗi; mỗi lần gợi ý mất khoảng 1 phút tính cả nạp model.
3. **Khóa settings của Animate 2 và SVI 2 Pro chưa chạy thử.** Các giá trị `image_prompt_type` / `video_prompt_type` / `image_refs` ở trên đọc từ `models/wan/wan_handler.py`; xác nhận bằng cách so với settings xuất từ UI Wan2GP và lần chạy tay đầu tiên. **Kết quả 2026-09-30:** SVI 2 Pro (`V` + `KI` + `image_refs`) chạy được qua `--check-extend`. Animate 2 (`T` + `UVI` + `image_refs`) chạy được qua `--check-motion` trên ảnh hai nhân vật và video mẫu 30 fps: 149 frame, 3 cửa sổ, mặt và trang phục giữ nguyên qua cả ba cửa sổ.
4. **Kéo dài video khi nguồn là 32 fps.** Nếu Wan2GP không tự đưa video nguồn về 16 fps, app giữ bản 16 fps gốc (sinh với `temporal_upsampling` rỗng) để làm `video_source` và chỉ nội suy ở bản hiển thị. **Kết quả 2026-09-30:** không xảy ra; nguồn 32 fps (161 frame) được nối đúng tốc độ, kết quả 9.78 giây ở 32 fps.
5. **720p trên 12GB.** Chậm hơn và có thể thiếu VRAM với Animate 2; mặc định vẫn là 480p. **Kết quả 2026-09-30:** chưa thử 720p.

## Kết quả đo ngày 2026-09-30 (RTX 5070 Ti Laptop 12GB, profile 4)

Ảnh mẫu vuông 1344x1344 → `720x720`, Mượt x2 bật.

| Việc | Thời gian |
|---|---|
| Reel 5 giây, 81 frame, 4 bước (`--check-reel`, gồm tự viết prompt và nạp model) | 9 phút 14 giây; riêng khử nhiễu 6 phút 26 giây (khoảng 97 giây mỗi bước) |
| Kéo dài thêm 2 giây, 29 frame, 8 bước (`--check-extend`) | 7 phút 43 giây; riêng khử nhiễu 6 phút 14 giây |
| Kéo dài 81 frame, 8 bước (trước khi trừ chồng lấn) | cửa sổ một 14 phút 11 giây, cửa sổ hai thừa 4 phút 9 giây |
| Reel 2 giây, 33 frame, qua hàng đợi của app | 3 phút 43 giây |
| Motion Control 5 giây ở 30 fps, 149 frame, `480x848`, 10 bước, 3 cửa sổ (`--check-motion`) | khử nhiễu 15 phút 35 giây + 15 phút 42 giây + 6 phút 37 giây; khoảng 94 giây mỗi bước cho cửa sổ 65 frame |

Máy thiếu RAM để ghim cả hai transformer (log: "Unable to pin data of 'transformer2'"), nên tốc độ bị giới hạn bởi việc chuyển dữ liệu RAM → VRAM.

Lệnh kiểm tra không cần UI:

```
python simple_app.py --check-enhancer <ảnh>
python simple_app.py --check-reel <ảnh>
python simple_app.py --check-extend <video> <ảnh> <prompt ...>
python simple_app.py --check-motion <ảnh> <video>
```

## Ứng viên xóa sau khi app chạy ổn (người dùng quyết định)

| File | Dung lượng |
|---|---|
| `ckpts/wan2.2_animate_relighting_lora.safetensors` | 2.7 GB |

Từ v2 ([spec 2026-10-01](2026-10-01-simple-app-v2-design.md)), `ckpts/wan2.2_animate_14B_quanto_bf16_int8.safetensors` và `loras/wan_i2v/Wan2.1_I2V_14B_FusionX_LoRA.safetensors` được Motion Control dùng, không còn là ứng viên xóa.

## Tham khảo

Trong repo:
- `docs/API.md`: `init`, `submit_task`, callback, `list_loras`, `get_model_availability`.
- `docs/LORAS.md`: cú pháp hệ số theo pha (`1;0`, `0;1`).
- `docs/PROCESSING.md`: sliding window và hiện tượng trôi nhân vật ở video dài.
- `docs/SPATIAL_UPSAMPLERS.md`, `postprocessing/h3_face_refiner/`: H3 Face Refiner.
- `models/wan/wan_handler.py`: chế độ Animate 2 (`image_ref_choices`), SVI 2 Pro (`KI`), thư mục LoRA.
- `defaults/i2v_2_2_Enhanced_Lightning_v2.json`, `defaults/i2v_2_2_Enhanced_Lightning_v2_svi2pro.json`, `defaults/animate2_distilled.json`, `defaults/scail2_14B.json`.
- `postprocessing/rife/temporal_upsampler.py`: RIFE x2/x3/x4.
- `wgp.py`: `process_prompt_enhancer`, `exec_prompt_enhancer_engine`.
- `plugins/sample/plugin.py`: mẫu plugin tab (phương án dự phòng).

Project tham chiếu (chỉ mượn ý tưởng UI):
- `D:\001_Personal_Proj\wan2-2-i2v-v3`, nguồn https://huggingface.co/spaces/observantdistressed/wan2-2-i2v-v3

Model:
- https://huggingface.co/DeepBeepMeep/Wan2.2 (Enhanced Lightning v2, Animate 2, LoRA SVI 2 Pro)
- https://huggingface.co/DeepBeepMeep/Wan2.1 (SCAIL-2, RIFE)
- https://huggingface.co/DeepBeepMeep/MiniMax-H3 (H3 Ref2VA cho Face Refiner)
- https://civitai.com/models/2053259 (tác giả Enhanced Lightning v2)

Giữ nhân vật:
- https://www.runcomfy.com/trainer/ai-toolkit/wan-2-2-i2v-character-consistency-lora
- https://civitai.com/models/2288568/wan22-stand-in-face-consistency (chưa kiểm chứng, ComfyUI)
- https://wananimate.app/character-animation-replacement
- https://poyo.ai/hub/wan-animate-2-complete-guide

## Vấn đề còn mở

Người dùng gặp một lần lỗi `TypeError: 'method' object does not support item assignment` trong `gradio/queueing.py:327` (`Queue.start_processing`), làm vòng xử lý hàng đợi dừng. Ngày 2026-09-30 chưa tái hiện được: các sự kiện của app qua `/queue/join`, một lần sinh video thật qua hàng đợi, và các sự kiện gửi sau đó đều chạy bình thường. Chưa biết nguyên nhân; cần thao tác cụ thể và log console phía trên dòng lỗi.
