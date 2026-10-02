# Simple App v3: giao diện tối, giữ da & tóc, harness Multi, âm thanh và trang phục ở Motion Control

Ngày: 2026-10-03. Nhánh: `20261003_simple_app_v3` (tách từ `20261001_simple_app_v2`). Nối tiếp [spec v2](2026-10-01-simple-app-v2-design.md); phần nào không nhắc ở đây giữ như v2. Thuật ngữ theo [GLOSSARY.md](../../../GLOSSARY.md).

## Mục tiêu

1. Giao diện theo các quy ước của taste-skill, impeccable.style và DESIGN.md của Linear/Runway: nền tối trung tính, một màu nhấn.
2. Reels giữ da và tóc của Character ở cảnh cận: LoRA chi tiết ở Pha cuối, và "Viết chi tiết" không cho máy đẩy sát mặt.
3. Harness cho Reels Multi Image đọc Story.
4. Motion Control: video ra mang âm thanh của Driving Video; tùy chọn cho trang phục đổi theo Driving Video (đoạn cởi áo).
5. Không để lại Intermediate Video sau khi job xong.

Tiêu chí xong: bốn lần chạy harness ở mục Kiểm thử chạy qua, kết quả ghi vào `logs/simple_app_harness.log` và bảng "Kết quả đo".

## Giới hạn

Như v2: một file `simple_app.py`, không sửa `wgp.py` và code Wan2GP, không thêm thư viện. Model mới duy nhất: LoRA `loras/wan/detailz-wan.safetensors` (307 MB, người dùng đã đồng ý tải 2026-10-03).

## 1. Giao diện

Nguồn: taste-skill (`skills/taste-skill/SKILL.md`), impeccable (`/slop`, `reference/craft-floor.md`, `operate.md`), awesome-design-md (`linear.app`, `runwayml`). Điểm chung được áp dụng: một màu nhấn chỉ cho hành động chính, trạng thái và focus; nền trung tính; viền 1px thay bóng; bo góc nhỏ đồng nhất; một họ font sans; không emoji làm icon; không monospace làm "trang phục"; tương phản AA.

Bảng màu (đã duyệt qua mockup ngày 2026-10-03):

| Token | Màu | Dùng cho |
|---|---|---|
| bg | `#0B0B0C` | nền trang |
| surface | `#141416` | khối, dialog |
| panel | `#1C1C1F` | ô nhập, nút phụ |
| border | `#27272A` | viền khối và ô nhập |
| border_strong | `#3F3F46` | viền nút phụ, checkbox, dialog |
| text | `#F4F4F5` | chữ chính |
| text2 | `#A1A1AA` | nhãn, chữ phụ |
| accent | `#F97316` | nút Tạo Video, tab đang chọn, slider, focus |
| accent_hover | `#FB923C` | hover nút chính |
| on_accent | `#0B0B0C` | chữ trên nút cam (tương phản ~7:1) |
| success / warn / error | `#4ADE80` / `#FBBF24` / `#F87171` | thông báo |

- Theme: `gr.themes.Base(primary_hue="orange", neutral_hue="zinc", radius_size=radius_sm)` với font Inter Tight (chữ) và JetBrains Mono (chỉ cho dòng số: ước tính, thông tin ảnh/video, thời gian thực tế). Các biến theme đặt cho cả chế độ sáng và `_dark`; trang luôn tối (`js` thêm class `dark`).
- Bố cục: rộng tối đa 1280px, căn giữa. Header là tiêu đề "Simple App" 24px/600 cùng dòng phụ tên ba tab, nút "Hướng dẫn" (nút phụ) bên phải, đường kẻ dưới. Giữ hai cột như v2, cột trái rộng hơn (5:4).
- Mỗi tab một nút chính (cam) là "Tạo Video"; Hủy, Kéo dài, Upscale, 3 ý tưởng, Viết chi tiết, Thêm ảnh, Xóa là nút phụ.
- Không emoji: dòng ước tính là "Ước tính 35 phút · SVI 2 Pro · 3 cửa sổ × 8 bước"; cảnh báo bắt đầu bằng "Lưu ý:" (màu warn).
- Không đổ bóng (`*_shadow` = `none`), viền 1px.
- Mockup: `mockup_v3.py` trong scratchpad của phiên (không commit).

## 2. Giữ da & tóc

### LoRA Detailz

`detailz-wan.safetensors` từ https://huggingface.co/DAKARA555/DetailEnhancerforWANVideos (bản sao của https://civitai.com/models/1385506, Apache 2.0). Train trên 704 ảnh người độ phân giải cao, tăng chi tiết da và mặt mà không rời ảnh gốc. Là LoRA Wan 2.1 14B; cộng đồng dùng cho expert low-noise của Wan 2.2.

- Tick **Giữ da & tóc** (tab Reels và Reels Multi Image, mặc định tắt) thêm `detailz-wan.safetensors` vào `activated_loras` với hệ số `0;0.6`: chỉ chạy ở Pha cuối, độc lập với thanh Pha cuối của các LoRA khác.
- Nếu người dùng đã tự chọn file này trong ô LoRA thì không thêm lần hai.
- Áp dụng cho cả Kéo dài video.

### Không đẩy sát mặt

`detail_instructions` thêm: camera giữ cùng khoảng cách và khung hình như ảnh, không zoom, không push-in, không cận mặt; người có thể tự bước lại gần. Nguyên nhân: ở harness 15 s của v2, prompt có "slow zoom towards his face" và mặt/tóc đổi ở giây 5–10.

### Hướng dẫn

Dialog thêm: "Ảnh cận mặt: bật *Giữ da & tóc*, nên chọn 720p."

## 3. Harness Reels Multi Image

`python simple_app.py --check-multi <thư mục> [--detail]` đọc `<thư mục>/story.txt`.

Định dạng Story: các khối cách nhau bởi dòng trống; dòng đầu khối là `<tên ảnh> : <thời gian> :`, các dòng sau là nội dung. `<thời gian>` là khoảng `0–3s` / `0-3s` (thời lượng = cuối − đầu) hoặc thời lượng `3s` / `3`. Hàm thuần `parse_story(text) -> list[tuple[str, int, str]]` trả (tên ảnh, số giây, prompt gộp một dòng), theo thứ tự trong file; thời lượng ngoài 1–5 giây hoặc dòng đầu sai định dạng → `ValueError` nêu dòng đầu khối bị lỗi.

Ảnh trong thư mục mà Story không nhắc tới thì bỏ qua. Prompt giữ nguyên văn, Mô tả nhân vật chung để trống. Story hiện có (`D:\001_Personal_Proj\Comfy\PostIG\day_24_1\Ryo\story.txt`) cho 3 Segment 3 + 4 + 4 = 11 giây; ảnh 832×1216 → `480x832`. Mong đợi video 11 s ± 0.2.

## 4. Motion Control

### Âm thanh

Tick **Giữ âm thanh video mẫu** (mặc định bật) đặt `postprocess_audio = "control"`. Wan2GP lấy âm thanh từ `video_guide`, đệm rồi cắt bằng `-shortest` theo độ dài video ra (`wgp.py:7166-7168`, `shared/utils/audio_video.py:554-603`). Bỏ tick → `""`. Driving Video không có âm thanh thì video ra không có âm thanh, không lỗi.

### Trang phục theo video mẫu

Animate 2 chỉ nhận một ảnh tham chiếu và dùng nó cho mọi Window (`models/wan/any2video.py:860-869`), nên ở chế độ hiện tại (`UVI`) trang phục luôn khóa theo ảnh: trong Driving Video `day_27_1`, cả hai người cởi áo ở giây ~10.3–13.5 và ở trần từ giây ~14, nhưng video ra chỉ có động tác.

Tick **Trang phục theo video mẫu** (mặc định tắt) chuyển Animate 2 sang chế độ mặc định của model: `image_prompt_type = "S"`, `image_start = ảnh`, `video_prompt_type = "UV"`, không `image_refs`. Window đầu bắt đầu từ ảnh, các Window sau nối từ frame cuối của Window trước, nên trạng thái cởi áo có thể đi tiếp. Đánh đổi: không còn Anchor cố định, mặt có thể trôi qua nhiều Window. Với Wan 2.2 Animate 14B, tùy chọn bị vô hiệu (chế độ `PVBKI` luôn tham chiếu ảnh); app báo "Lưu ý:" và giữ nguyên.

## 5. Intermediate Video

Khi khởi động, `simple_app` đặt `sys.modules["wgp"].server_config["keep_intermediate_sliding_windows"] = 0` (chỉ trong bộ nhớ; `wgp_config.json` và UI chính của Wan2GP không đổi). Wan2GP khi đó thay file của Window trước bằng file mới (`wgp.py:8443-8444`, `shared/utils/media_recording.py:58-72`); job bị hủy giữa chừng còn lại file của Window cuối đã xong.

Sau job thành công nhiều Window, giao diện hiện `gr.Info("Đã xóa N video trung gian, chỉ giữ video cuối.")` với N = số Window − 1, và dòng thời gian thực tế ghi thêm "· đã xóa N video trung gian". Harness ghi N vào log.

## 6. Log harness

Hàm `log_harness(entry: dict)` ghi thêm vào `logs/simple_app_harness.log` (thư mục `logs/` trong `.gitignore`) một khối:

```
=== 2026-10-03 06:10:00 --check-reel ...
model: i2v_2_2_Enhanced_Lightning_v2_svi2pro  resolution: 480x832  seed: 1234  detail_lora: True
expected: 10.00s  got: 10.06s (322 frames @ 32 fps)  ok: True
took: 24m 51s  intermediate_removed: 1  audio: -
video: D:\...\outputs\....mp4
sheet: D:\...\logs\2026-10-03-0610_reel.jpg
```

Ảnh ghép (`contact_sheet(video, times, image=None) -> path`) gồm ảnh gốc (nếu có) và frame ở các mốc, lưu cạnh log. Kèm các frame đầy đủ độ phân giải ở cùng mốc để soi da và tóc.

## Kiểm thử

`test_simple_app.py` thêm assert cho: `parse_story` (khoảng có gạch ngang dài và ngắn, thời lượng đơn, gộp dòng, thứ tự, lỗi thời lượng 0 hoặc 6, lỗi dòng đầu), `detail_lora` trong `_lora_settings` (thêm `0;0.6`, không trùng), `motion_settings` với `audio` và `outfit`, `detail_instructions` có câu cấm zoom, tạo bảng màu theme không lỗi.

Harness (GPU, mỗi lần một job):

| # | Lệnh | Kiểm |
|---|---|---|
| 1 | `--check-reel <day_77_1 ảnh> 10 docs/superpowers/harness/reel_day77_10s.txt --seed 20261003` | 10 s ± 0.2; ảnh ghép giây 0/3/6/9 |
| 2 | như (1) thêm `--detail` | như (1); so da/tóc với (1) |
| 3 | `--check-multi D:\001_Personal_Proj\Comfy\PostIG\day_24_1\Ryo` | 11 s ± 0.2; frame ở đầu mỗi Segment |
| 4 | `--check-motion <Klein.png> <vietnameseboy.mp4> animate2 16 --outfit` | 16 s ± 0.2; có luồng âm thanh; frame giây 8/11/13/15 (áo) và 0/15 (mặt) |

`--check-reel` mới nhận file prompt (không qua Qwen) và `--seed`; thiếu file prompt thì giữ cách cũ (Qwen viết prompt). Prompt harness 10 s nằm ở `docs/superpowers/harness/reel_day77_10s.txt`: mỗi giây một dòng, giây 0–4 tạo dáng khoe cơ bắp, giây 5–9 bước lại gần, cúi nhẹ và vẫy tay; dòng nào cũng nhắc lại da, tóc, trang phục và khoảng cách máy cố định.

Chạy tay: mở app, xem ba tab theo bảng màu, dialog Hướng dẫn, các checkbox mới.

## Rủi ro

1. Detailz là LoRA Wan 2.1: có thể không tác dụng hoặc làm lệch màu trên Wan 2.2 low-noise. Harness (1)/(2) cùng seed trả lời câu này; phương án sau: Instareal_low (614 MB, cần hỏi lại trước khi tải).
2. Trang phục theo video mẫu có thể vẫn không cởi áo (Window 3 bắt đầu từ frame còn mặc áo) hoặc làm mặt trôi; harness (4) kiểm. Phương án sau: ảnh nhân vật thứ hai từ giây X, cắt Driving Video thành hai job.
3. Story dùng tên ảnh có ký tự `%`; đường dẫn ghép bằng `Path`, không qua shell.
