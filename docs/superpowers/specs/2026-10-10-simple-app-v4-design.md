# Simple App v4: đoạn dài tới 15 s, prompt theo giây và enhancer cho Reels Multi Image, log prompt

Ngày: 2026-10-10. Nhánh: `20261010_Simple_3_1` (tách từ `20261003_simple_app_v3`). Nối tiếp [spec v3](2026-10-03-simple-app-v3-design.md); phần nào không nhắc ở đây giữ như v3. Thuật ngữ theo [GLOSSARY.md](../../../GLOSSARY.md).

## Mục tiêu

1. Reels Multi Image: mỗi Segment dài 1–15 s thay vì 1–5 s; prompt của Segment viết theo giây đúng định dạng Lightning v2 được huấn luyện, để thứ tự động tác trong Segment được giữ (lỗi video 2026-10-09: quấn khăn trước khi quay lưng).
2. Reels Multi Image có "Viết chi tiết" như tab Reels: enhancer biết ảnh đầu, ảnh cuối và Segment trước, viết các dòng theo giây; mối nối giữa các Segment mượt hơn nhờ prompt liên tục.
3. Cả hai tab: chỉ dẫn enhancer theo cấu trúc của [Veed Wan 2.2 prompting guide](https://www.veed.io/learn/wan-2-2-prompting-guide); sau khi tạo xong hiện đúng prompt đã gửi cho Wan2GP.
4. Log `logs/simple_app.log` ghi input/output của enhancer và prompt cuối của mọi job, từ UI lẫn harness.
5. Harness `--check-multi` đọc Story có cặp dòng VI/EN, thử được tiếng Việt, có cờ chạy enhancer trước.

Tiêu chí xong: unit test chạy qua; `--check-enhancer` ra dòng `(at N seconds:`; `--check-multi` với Story `story4_test20261010` (23 s, 4 Segment) ra đúng thời lượng, contact sheet cho thấy khăn được quấn sau khi quay lưng; kết quả ghi vào bảng "Kết quả đo".

## Giới hạn

Như v3: một file `simple_app.py`, không sửa `wgp.py` và code Wan2GP, không thêm thư viện, không tải model mới. Enhancer vẫn là Qwen3.5-4B có thị giác đang có trong `ckpts/Qwen3_5_4B_Abliterated` (`enhancer_enabled = 3`).

## Phát hiện quyết định thiết kế

Đọc code ngày 2026-10-10:

- **Ảnh kết thúc được tiêu thụ theo thứ tự cửa sổ** (`wgp.py:7559`): cửa sổ i lấy `image_end[i-1]`, hết danh sách thì cửa sổ không có ảnh kết thúc. Danh sách không được chứa ô trống (`clean_image_list`, `wgp.py:371`). Vì vậy một Segment giữa hai Keyframe bắt buộc là **một cửa sổ**; không chia Segment 10 s thành hai cửa sổ 5 s được.
- **Overlap giữa các cửa sổ của Lightning i2v bị khóa 1 frame** (`models/wan/wan_handler.py:997`). Cửa sổ sau chỉ nhận frame cuối, không nhận đà chuyển động. Mối nối chỉ mượt được bằng prompt: giây cuối Segment trước và giây đầu Segment sau tả cùng một động tác, máy quay đứng yên ở ranh giới, không thả hiệu ứng mới ngay đầu Segment.
- **Một cửa sổ tối đa 257 frame = 16 s** (`wgp.py:12486`); Wan2GP mặc định 129 frame = 8 s. Model huấn luyện ở 81 frame, trên 8 s hay lặp động tác hoặc trôi mặt, thời gian tăng nhanh hơn tuyến tính.
- **Định dạng prompt được huấn luyện của Enhanced Lightning v2** là mỗi giây một dòng `(at N seconds: …)` (`defaults/i2v_2_2_Enhanced_Lightning_v2.json`). Mốc kiểu `(2s-4s)` trong một cửa sổ model không hiểu. Tab Multi hiện ép prompt thành một câu (`_one_line`), mất mốc giây.
- **Enhancer không đưa ảnh vào LLM**: caption từng ảnh rồi ghép caption của ảnh đầu tiên với text (`shared/prompt_enhancer/prompt_enhance_utils.py:374`). Muốn LLM biết ảnh cuối phải tự caption ảnh đó và chèn vào text.
- **Text encoder umT5 giới hạn 512 token mỗi cửa sổ** (`models/wan/configs/shared_config.py:11`), vượt thì cắt.
- **Video 2026-10-09 16h03** (`outputs/…seed879420944…`): Keyframe 2 (`R1_4`) đã là tư thế ngồi quấn khăn hướng camera; Segment 1 yêu cầu 6 động tác trong 5 s và phải kết thúc khớp Keyframe 2, model nén lại nên quấn khăn ở giây 2–3. Mối nối 5 → 6 s giật vì Segment 2 mở đầu bằng "steam swirls, camera orbits".
- **Ngôn ngữ**: umT5-XXL mã hóa được tiếng Việt nhưng tài liệu Wan 2.2 chỉ ví dụ tiếng Anh và tiếng Trung, prompt extension chính thức chỉ nhắm `zh`/`en`. Tiếng Việt chạy được, bám prompt yếu hơn, chưa có số đo; harness v4 đo thử.
- **`qwen3vl_4b_fp8_scaled.safetensors`** là file text encoder đóng gói cho ComfyUI (Qwen3-VL-4B fp8, khoảng 4.5 GB). VRAM 12 GB đủ nhưng loader enhancer của Wan2GP chỉ nhận Llama 3.2, JoyCaption và Qwen3.5-4B (`shared/prompt_enhancer/loader.py:44`), và nó không hơn Qwen3.5-4B đang có. Không thêm.

## 1. Reels Multi Image

### Segment dài tới 15 s

- Slider "Thời lượng đoạn (giây)" của mỗi card: 1–15, mặc định 3.
- Dưới slider một dòng trạng thái (class `eta`): "Kết thúc ở ảnh N+1" với card chưa phải cuối, "Đoạn cuối, chạy tự do" với card cuối. Từ 9 s trở lên dòng này đổi sang class `note` (vàng): "Hơn 8 s: có thể lặp động tác, thời gian tăng nhanh". Không chặn.
- `multi_settings` giữ nguyên cách một Segment = một cửa sổ, chỉ bỏ ràng buộc 5 s. `video_length = tổng giây × 16 + 1`.

### Prompt của cửa sổ

Cửa sổ i là một đoạn văn: dòng đầu `[/duration=Ds] <Mô tả nhân vật chung>` (nếu có), các dòng sau là prompt của Segment i **giữ nguyên văn**, nối bằng xuống dòng đơn; các cửa sổ cách nhau đúng một dòng trống (Wan2GP tách cửa sổ theo dòng trống, `prompt_parser.validate_sliding_window_prompt_boundaries`). Dòng trống bên trong prompt của Segment bị bỏ. `_one_line` không còn dùng cho Segment.

Hàm thuần `multi_prompt(common, prompts, durations) -> str` tách khỏi `multi_settings` để test.

### Ước tính thời gian

`estimate_seconds` tính công việc của mỗi cửa sổ f frame là `f ** 1.5 / 81 ** 0.5` thay vì `f` (gần tuyến tính quanh 81 frame, tăng dần trên đó; attention thực tăng bình phương). Với f ≤ 81 kết quả gần như cũ. Calibration sau lần chạy đầu tự chỉnh như v2.

### Giao diện

- Mỗi card thêm nút **Viết chi tiết** (secondary, sm) cạnh nút Xóa. Placeholder ô Prompt i: "ý ngắn, hoặc các dòng (at N seconds: …)".
- Dưới hàng card, cạnh "Thêm ảnh": nút **Viết chi tiết tất cả**.
- Cột phải, dưới dòng thời gian thật: ô **Prompt đã gửi** (Textbox chỉ đọc, 8 dòng, font mono), điền sau khi tạo xong bằng đúng `settings["prompt"]`. Tab Reels có ô giống vậy, hiện chuỗi sau `split_timeline` (hoặc prompt gốc khi ≤ 5 s); Kéo dài video cũng điền.
- Không đổi: tối đa 6 ảnh, độ phân giải theo ảnh 1, LoRA, Giữ da & tóc, seed, Mượt x2, Upscale.

## 2. Enhancer

### `write_segment` (mức module, UI và harness cùng gọi)

```
write_segment(session, image, end_image, seconds, common, hint, previous, progress=None) -> str
```

1. **Caption ảnh cuối** (khi `end_image` có): `enhance(session, "", "Describe this image.", end_image, POSE_INSTRUCTIONS)`. `POSE_INSTRUCTIONS`: một câu tiếng Anh tả tư thế, vị trí trong khung, trang phục hoặc đồ đang cầm, khung máy; không bình luận. Không bật Think.
2. **Text cho LLM** (tiếng Anh, ảnh chính là `image`):
   - `Character and setting: <common>` nếu có.
   - `Goal: <hint>`; nếu `hint` trống thì "Continue the scene naturally".
   - `The clip lasts <seconds> seconds.`
   - `It must end exactly in this pose: <caption>` khi có ảnh cuối; ngược lại "The clip ends freely".
   - `The previous clip ended with: <dòng cuối của previous>` khi có `previous`.
3. **Chỉ dẫn `segment_instructions(seconds, has_end, has_prev)`**: kế thừa `detail_instructions` (một cú máy liên tục, cùng người, cùng chỗ, máy giữ khoảng cách, không zoom sát mặt) và thêm:
   - Mỗi dòng đúng một chuyển động chính của nhân vật và một chuyển động máy cụ thể (pan, tilt, dolly, static); tả cách chuyển động, không tả lại ảnh; cường độ tăng dần.
   - Máy quay đứng yên ở dòng đầu và dòng cuối của đoạn.
   - Dòng đầu tiếp nối đúng động tác mà đoạn trước kết thúc (khi `has_prev`).
   - Dòng cuối đưa nhân vật về đúng tư thế ảnh cuối (khi `has_end`); các dòng trước phải khả thi về thời gian để tới được tư thế đó.
   - Bước thời gian: `step = 1` nếu `seconds ≤ 8`, ngược lại `2`; dòng ở `0, step, 2·step, …` nhỏ hơn `seconds`, mỗi dòng bắt đầu `(at N seconds:` và tối đa 25 từ.
   - Viết tiếng Anh dù Goal bằng tiếng khác. Chỉ in các dòng, không có text khác.
4. `enhance(..., think=True)`; lọc `<think>…</think>` nếu còn sót; trả về text, đồng thời ghi log `enhance` (mục 4).

Nút **Viết chi tiết** của card i gọi `write_segment` với `image = ảnh i`, `end_image = ảnh i+1` (None ở card cuối), `hint = Prompt i` hiện tại, `previous = Prompt i-1` hiện tại; kết quả ghi đè Prompt i. **Viết chi tiết tất cả** gọi tuần tự từ card 1 đến n, mỗi card dùng prompt vừa viết của card trước; trả về cả n ô Prompt. Yêu cầu không có job đang chạy, như "3 ý tưởng". Enhancer nạp và giải phóng theo từng lần gọi như hiện nay (khoảng 20 giây mỗi lần, ponytail: chưa gộp).

### Think

`enhance()` thêm tham số `think=False` và `purpose=""` (để ghi log); chế độ gửi cho Wan2GP là `wgp.build_prompt_enhancer_value("TI", think)` (thêm chữ `K`). "Viết chi tiết" ở tab Reels và `write_segment` bật; "3 ý tưởng" và caption không bật.

### Tab Reels

`detail_instructions` nhận cùng bộ quy tắc Veed: mỗi dòng một chuyển động chính và một chuyển động máy cụ thể, tả cách chuyển động, cường độ tăng dần, tối đa 25 từ mỗi dòng, viết tiếng Anh dù ý ngắn bằng tiếng khác. Vẫn mỗi giây một dòng vì `split_timeline` chia về từng cửa sổ ≤ 5 s. Giữ các quy tắc v3 (không zoom sát mặt, cùng người, cùng chỗ).

### Không làm

Negative prompt của Veed (Lightning chạy guidance 1, negative vô tác dụng); model `qwen3vl_4b_fp8_scaled`; chỉnh overlap cửa sổ (bị khóa).

## 3. Prompt đã gửi

`generate()` trong `build_ui` trả thêm `settings["prompt"]`; `make_reel`, `extend_reel`, `make_multi` điền vào ô "Prompt đã gửi" của tab mình. Khi hủy thì giữ nguyên ô.

## 4. Log

`log_harness(entry, path)` đổi tên `log_block(entry, path=APP_LOG)`; `HARNESS_LOG` giữ nguyên cho harness, thêm `APP_LOG = logs/simple_app.log`. Cùng định dạng khối: dòng `=== <thời gian> <loại>` rồi `key: value`; giá trị nhiều dòng ghi nguyên, các dòng tiếp theo thụt hai khoảng trắng.

| Loại | Khi nào | Khóa |
|---|---|---|
| `enhance` | mọi lần gọi `enhance()` | tab, purpose (ideas / detail / caption / segment N), image, end_image, think, input, output |
| `generate` | job xong (UI và harness) | tab hoặc check, model, resolution, seed, windows, prompt (nguyên văn), video, took, calib |
| `error` | `enhance()` hoặc job ném lỗi | tab, prompt, error |

Harness ghi `generate` vào cả hai file (khối harness giữ các khóa cũ: expected, got, ok, sheet). `logs/` không commit (như v3; file log v3 trên đĩa đã mất, bảng "Kết quả đo" của v3 là bản ghi còn lại).

## 5. Harness Reels Multi Image

```
python simple_app.py --check-multi <thư mục> [--enhance] [--lang en|vi] [--detail [x]] [--quality 480p|720p] [--seed N]
```

### Định dạng Story (`story.txt`)

Các khối cách nhau bởi dòng trống.

- Khối đầu **không có** dòng tiêu đề ảnh là Mô tả nhân vật chung (tùy chọn).
- Khối Segment: dòng đầu `<tên ảnh> : <thời gian> :` như v3, `<thời gian>` là `0–10s` hoặc `10s`; thời lượng 1–15 s nguyên.
- Dòng nội dung có thể bắt đầu bằng `(VI)` hoặc `(EN)`; dòng không có tiền tố thuộc cả hai ngôn ngữ. Dấu đầu dòng `-`, `*`, `•`, `1.` bị bỏ.

`parse_story(text, lang="en") -> (common, [(tên ảnh, giây, prompt)])`: `prompt` là các dòng thuộc `lang` nối bằng xuống dòng đơn, không ép thành một câu. Sai định dạng dòng đầu hoặc thời lượng ngoài 1–15 → `ValueError` như v3.

### Chạy

1. Đọc Story theo `--lang`.
2. Có `--enhance`: với từng Segment gọi `write_segment(image_i, image_i+1, giây, common, prompt_i, previous)` tuần tự, `previous` là prompt vừa viết; prompt viết ra vào log `enhance` và in ra console. Không có cờ: prompt Story gửi nguyên văn (so sánh với cách cũ, và với `--lang vi` là đo Wan hiểu tiếng Việt trực tiếp).
3. `multi_settings` như UI, `check("multi", …)`: thời lượng tổng ± 0.2 s; contact sheet ở đầu mỗi Segment và `tổng − 0.2` s; khối harness ghi thêm `segments`, `lang`, `enhance`.

### Story `D:\001_Personal_Proj\Comfy\result\story4_test20261010\story.txt`

Viết từ `prompt.txt` của người dùng (giữ cặp VI/EN), đoạn cuối "to End" lấy 3 s; ảnh 768×1360 → `480x832`; tổng 23 s, 4 cửa sổ (10 + 5 + 5 + 3).

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

Thời gian dự kiến ở 480p: cửa sổ 10 s khoảng 15–25 phút, ba cửa sổ còn lại khoảng 15 phút, enhancer 4 Segment × 3 lần gọi khoảng 5 phút. Chạy nền, log đọc bằng `tr '\r' '\n'`.

## 6. Hướng dẫn

Mục "Reels Multi Image" trong dialog: đoạn 1–15 s, trên 8 s có cảnh báo; prompt viết theo giây `(at N seconds: …)`; nút Viết chi tiết dùng ảnh đầu, ảnh cuối và đoạn trước; mối nối mượt khi hai đoạn kề tả cùng động tác và máy quay đứng yên ở ranh giới; ô Prompt đã gửi. Thêm một dòng về ngôn ngữ: viết tiếng Anh, ý ngắn tiếng Việt thì để enhancer dịch; prompt tiếng Việt gửi thẳng chạy được nhưng bám yếu hơn.

## Kiểm thử

Unit (`test_simple_app.py`, python thuần, chạy bằng `"$USERPROFILE/anaconda3/envs/wan2gp/python.exe" -m pytest test_simple_app.py`):

1. `multi_prompt`: giữ nguyên dòng `(at N seconds:`, xuống dòng đơn trong cửa sổ, một dòng trống giữa các cửa sổ, bỏ dòng trống bên trong Segment, Mô tả chung ở dòng `[/duration=`.
2. `parse_story`: khối mô tả chung, Segment 10 s, lọc `(VI)`/`(EN)` theo `lang`, bỏ dấu đầu dòng, lỗi khi 16 s.
3. `segment_instructions`: bước 1 s ở 8 s, 2 s ở 9 s; có câu về ảnh cuối chỉ khi `has_end`, về đoạn trước chỉ khi `has_prev`.
4. `estimate_seconds`: cửa sổ 161 frame tốn hơn hai cửa sổ 81 frame; cửa sổ ≤ 81 gần như cũ.
5. `log_block`: ghi đúng file, giá trị nhiều dòng thụt lề.
6. `multi_settings`: `video_length` với Segment 10 s, không còn giới hạn 5.

GPU (chạy nền, mỗi lần một process):

7. `--check-enhancer <ảnh Reels> 5`: output có dòng `(at 0 seconds:` … `(at 4 seconds:`, không có thẻ think; log `enhance` có hai khối.
8. `--check-multi story4_test20261010 --enhance --seed 20261010`: 23 s ± 0.2; contact sheet 0, 10, 15, 20, 22.8 s; soi thêm frame 2–6 s: quấn khăn sau khi quay lưng; mối nối 10, 15, 20 s.
9. `--check-multi story4_test20261010 --lang vi --seed 20261010` (không enhance): đo Wan hiểu tiếng Việt trực tiếp, so với (8).
10. Mở app: tab Multi slider 15 s và cảnh báo; Viết chi tiết một card rồi Viết chi tiết tất cả; tạo video ngắn (2 ảnh, 2 + 2 s) và xem ô Prompt đã gửi ở tab Multi và tab Reels.

## Rủi ro

1. Cửa sổ trên 8 s lặp động tác hoặc trôi mặt: cảnh báo trên UI; nếu lần (8) xấu ở Segment 10 s thì người dùng chia Keyframe dày hơn, app không tự chia được (ảnh kết thúc theo thứ tự cửa sổ).
2. 512 token: Mô tả chung dài cộng 8 dòng × 25 từ có thể vượt; quy tắc 25 từ và bước 2 s là chốt chặn duy nhất, chưa đếm token.
3. Mối nối vẫn giật vì overlap 1 frame là bản chất model; prompt liên tục chỉ giảm.
4. Think của Qwen3.5 lâu hơn và có thể để sót thẻ; lọc bằng regex.
5. Enhancer gọi 3 lần mỗi Segment, nạp lại model mỗi lần; 4 Segment mất khoảng 5 phút. Gộp nạp một lần nếu thấy chậm.
6. Ước tính `f ** 1.5` chưa đo; calibration chỉnh theo model, không theo độ dài cửa sổ.
7. Tiếng Việt gửi thẳng có thể bị Wan bỏ qua gần hết; lần (9) chỉ để đo.

## Kết quả đo (RTX 5070 Ti Laptop 12 GB)

Chi tiết từng lần chạy trong `logs/simple_app_harness.log` và `logs/simple_app.log` (không commit); ảnh ghép frame trong `logs/`.

| # | Việc | Kết quả |
|---|---|---|
| 7 | `--check-enhancer` | chưa chạy |
| 8 | `--check-multi story4 --enhance` | chưa chạy |
| 9 | `--check-multi story4 --lang vi` | chưa chạy |
