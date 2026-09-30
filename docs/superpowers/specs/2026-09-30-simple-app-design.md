# Simple App: Reels + Motion Control

Ngày: 2026-09-30. Nhánh: `20260930_simple_app`.

## Mục tiêu

Một app Gradio một màn hình, hai tab, chạy trên máy RTX 5070 Ti Laptop 12GB, dùng lại toàn bộ phần nạp model và sinh video của Wan2GP:

1. **Reels**: ảnh → video ngắn, có gợi ý prompt từ ảnh.
2. **Motion Control**: video mẫu có người → nhân vật trong ảnh chuyển động theo (kiểu Kling Motion Control).

Tiêu chí xong: mỗi tab sinh ra được một video từ đầu đến cuối trên máy người dùng mà không mở UI lớn của Wan2GP.

## Ngoài phạm vi bản đầu

Wan-Move (vẽ quỹ đạo), chế độ thay nhân vật trong video, tự bật LoRA theo prompt, upscale, âm thanh, hàng đợi nhiều job, xóa model (người dùng tự quyết).

## Kiến trúc

- `simple_app.py` ở gốc repo: toàn bộ UI và logic.
- `run_simple_app.bat`: kích hoạt env conda `wan2gp` rồi chạy `python simple_app.py` (theo mẫu `run_wgp.bat`).
- `test_simple_app.py`: kiểm thử các hàm thuần.
- Không sửa `wgp.py`. Không thêm thư viện (Gradio 5.29 đã có trong `requirements.txt`).

App gọi `shared.api.init(root=<thư mục repo>, cli_args=["--profile", "4"])` một lần khi khởi động. Mỗi lần Generate là một `session.submit_task(settings)`; tiến độ lấy qua callback `on_status` / `on_progress` và đẩy vào `gr.Progress`. Nút Hủy gọi `job.cancel()`. Mỗi lúc chỉ chạy một job.

## Các hàm thuần

| Hàm | Vào | Ra |
|---|---|---|
| `pick_resolution(w, h)` | kích thước ảnh | `"480x832"` nếu w/h < 0.9, `"832x480"` nếu w/h > 1.1, còn lại `"720x720"` |
| `frames_for(seconds, fps=16)` | số giây | số frame dạng 4k+1, kẹp trong [17, 81] |
| `pair_loras(names, strength)` | danh sách tên LoRA đã chọn | `(activated_loras, loras_multipliers)` |
| `parse_ideas(text)` | đầu ra LLM | danh sách tối đa 3 ý, bỏ số thứ tự và dòng trống |
| `last_frame(video_path)` | đường dẫn video | ảnh frame cuối (cv2) |

Quy tắc `pair_loras`: tên chứa `high` (hoặc `_HN`, `_H.`) không phân biệt hoa thường nhận hệ số `s;0`; tên chứa `low` (hoặc `_LN`, `_L.`) nhận `0;s`; tên không khớp nhận `s` cho cả hai pha. Dropdown hiển thị từng file riêng; người dùng tự chọn cả hai file của một cặp.

## Tab Reels

Model: `i2v_2_2_Enhanced_Lightning_v2` (đã có trên đĩa).

Thành phần UI: ảnh đầu vào, ô "ý ngắn", nút **3 ý tưởng**, radio 3 ý, nút **Viết chi tiết**, ô prompt, thanh thời lượng 2–5 giây (mặc định 5), tick **Mượt x2**, dropdown LoRA nhiều lựa chọn, thanh cường độ LoRA (0–2, mặc định 1), seed (-1 là ngẫu nhiên), nút Generate, nút Hủy, video kết quả, nút **Dùng frame cuối làm ảnh đầu**.

Settings gửi đi, xây trên `session.get_default_settings(model_type)`:

```python
{
    "model_type": "i2v_2_2_Enhanced_Lightning_v2",
    "prompt": prompt,
    "image_prompt_type": "S",
    "image_start": image_path,
    "resolution": pick_resolution(w, h),
    "video_length": frames_for(seconds),
    "seed": seed,
    "activated_loras": [...],
    "loras_multipliers": "...",
    "temporal_upsampling": "rife*2" if smooth else "",
}
```

Danh sách LoRA lấy từ `session.list_loras("i2v_2_2_Enhanced_Lightning_v2")` (thư mục `loras/wan`), nạp lại mỗi lần mở dropdown.

## Gợi ý prompt từ ảnh

Dùng Qwen3.5-4B Abliterated có vision (đã có trên đĩa) qua hàm nội bộ của `wgp.py`.

- **3 ý tưởng**: gọi `process_prompt_enhancer` với `model_def=None`, chế độ `"TI"`, và chỉ dẫn riêng yêu cầu đúng 3 ý tưởng video ngắn khác nhau, mỗi ý một dòng, một câu, phù hợp với ảnh. Kết quả qua `parse_ideas` rồi hiện thành radio. Chọn một ý thì điền vào ô "ý ngắn". Bao quanh bằng cùng chuỗi nạp/nhả mà `exec_prompt_enhancer_engine` dùng (`acquire_GPU_ressources`, `ensure_prompt_enhancer_loaded`, nhả sau khi xong).
- **Viết chi tiết**: gọi `exec_prompt_enhancer_engine` với chế độ `"TI"`, ảnh đầu vào và ô "ý ngắn"; dùng chỉ dẫn theo từng giây có sẵn trong định nghĩa model. Kết quả điền vào ô prompt, người dùng sửa được.

Cả hai cần `state` của session (`session._state`, thuộc tính riêng tư) và module `wgp` đã nạp.

## Tab Motion Control

Model: `animate2_distilled` (tải 17 GB ở lần chạy đầu; app hiện cảnh báo trước khi gửi job nếu `session.get_model_availability` báo thiếu file).

Thành phần UI: ảnh nhân vật, video mẫu, ô mô tả nhân vật (mặc định lấy từ định nghĩa model), thanh giới hạn thời lượng 2–10 giây (mặc định 5), seed, Generate, Hủy, video kết quả.

Settings, xây trên `session.get_default_settings("animate2_distilled")`:

```python
{
    "model_type": "animate2_distilled",
    "prompt": character_description,
    "image_prompt_type": "S",
    "image_start": character_image_path,
    "video_prompt_type": "UV",
    "video_guide": driving_video_path,
    "resolution": pick_resolution(w, h),   # theo ảnh nhân vật
    "video_length": min(số frame video mẫu, giới hạn giây × fps video mẫu),  # làm tròn xuống dạng 4k+1
    "force_fps": "control",
    "seed": seed,
}
```

Video dài hơn 65 frame dựa vào sliding window mặc định của model (65 frame mỗi cửa sổ).

## Xử lý lỗi

- Thiếu ảnh, thiếu video mẫu, hoặc prompt Reels rỗng: `gr.Error` trước khi gửi job.
- Job thất bại: hiện `result.errors[0].message`.
- Job bị hủy: giữ nguyên video cũ, không báo lỗi.
- Lỗi khi gợi ý prompt: nhả GPU rồi hiện lỗi; ô prompt giữ nguyên.

## Kiểm thử

- `test_simple_app.py`: assert cho `pick_resolution`, `frames_for`, `pair_loras`, `parse_ideas`. Chạy bằng `python test_simple_app.py`, không cần GPU và không import `wgp`.
- Chạy tay trên máy người dùng: một video Reels (có và không có gợi ý), một video Motion Control.

## Rủi ro và phương án dự phòng

1. **Gợi ý prompt ở chế độ độc lập.** API không có hàm công khai; app dựa vào hàm nội bộ và `session._state`. Bước đầu tiên của kế hoạch là thử riêng lời gọi này. Nếu không chạy được, chuyển app thành plugin tab (`plugins/wan2gp-simple-app/`) dùng `api_session` và `state` do Wan2GP cấp; UI và các hàm thuần giữ nguyên.
2. **Đổi model trên 12GB.** Qwen và model video không nằm cùng lúc trên GPU; mỗi lần gợi ý rồi Generate sẽ tốn thời gian nạp lại. Chấp nhận ở bản đầu.
3. **Thiết lập Animate 2 chưa chạy thử.** Các khóa settings lấy từ mặc định của model; xác nhận bằng lần chạy tay đầu tiên.

## Ứng viên xóa sau khi app chạy ổn (người dùng quyết định)

| File | Dung lượng |
|---|---|
| `ckpts/wan2.2_animate_14B_quanto_bf16_int8.safetensors` | 16.7 GB |
| `ckpts/wan2.2_animate_relighting_lora.safetensors` | 2.7 GB |
| `loras/wan_i2v/Wan2.1_I2V_14B_FusionX_LoRA.safetensors` | 354 MB |
