# PatchCore cho B/C/D/G và số liệu nhánh R đã sửa — 25/11/2026

> **File này thay thế `Ket_qua_nhanh_R_BCDG_20261124.md`.** Số liệu nhánh R trong file 24/11 sai vì lỗi căn chỉnh (mục 1). Phần "đoạn đề xuất cho báo cáo" trong file cũ cũng không dùng được nữa.

## 1. Lỗi căn chỉnh đã tìm ra và đã sửa

- **Nguyên nhân:**
  - Bước căn ảnh SIFT–RANSAC lấy điểm đặc trưng trên **toàn ảnh**.
  - Nền đen có vân và **đứng yên** giữa các lần chụp; sau CLAHE, nền này cho rất nhiều điểm SIFT.
  - Khi bo chỉ xê dịch khoảng 100 px, RANSAC chọn khớp theo nền và trả về gần như "không dịch chuyển", trong khi bo thực tế đã lệch.
- **Cách kiểm tra độc lập với SIFT:** sau khi căn, vùng bo màu xanh của ảnh chụp phải chồng khít vùng bo của ảnh mẫu, IoU ≥ 0,95.

| Thiết kế | Số ảnh | Cách cũ: căn sai | Cách mới: không căn được |
|---|---|---|---|
| B | 176 | 62 | 0 |
| C | 115 | 112 | 0 |
| D | 90 | 89 | 7 (báo chụp lại) |
| G | 40 | 27 | 0 |
| A (nhánh B cũ) | 54 | 1 | 1 |

- **Cách mới (`board_align.py`):**
  - Chỉ lấy điểm SIFT trong vùng bo.
  - Bo D có hàng lỗ giống hệt nhau nên phép lọc tỉ lệ (ratio test) loại gần hết cặp điểm; khi đó dùng khớp hai chiều.
  - Phương án cuối: căn thô theo viền bo, rồi tinh chỉnh bằng ECC.
  - Mọi kết quả đều phải qua bước kiểm tra chồng khít ở trên.
- **Thiết kế A gần như không bị ảnh hưởng** (1/54 ảnh), nên kết quả nhánh B giai đoạn 2 vẫn dùng được.
- **Phạm vi ảnh hưởng ở thiết kế B/C/D/G:**
  - Số liệu nhánh R ngày 24/11.
  - Phần "vùng R chỉ xem" trong `PCB_AOI_UI`: đã sửa, đóng gói lại thành `PCB_AOI_Robot_20261125.zip`, 66/66 ca kiểm thử đạt.

## 2. Nhánh R (so ảnh mẫu), số liệu đã sửa

Cách làm giữ như ngày 24/11. Ngưỡng hiệu chuẩn lại trên B/C với quy tắc ≤ 5% ảnh tốt bị cảnh báo, ra **11,0**, rồi khóa và chạy D_01, D_02, G_01 với mẫu lấy từ D_00, G_00.

- **Trên B/C:**
  - Bước kiểm tra chất lượng ảnh của nhánh R từ chối gần hết ảnh của các bo C khác bo mẫu: 46/46 ảnh lỗi và 23/30 ảnh tốt bị báo chụp lại.
  - Nghĩa là nhánh R không chấp nhận được biến thiên giữa các bo C.
- **Trên D/G** (50 ảnh tốt, 30 ảnh lỗi, 130 nhãn):

| Chỉ số | Kết quả |
|---|---|
| Nhãn khoanh trúng ở ngưỡng 11,0 | 42/130 (32%) |
| Nhãn nằm trong 8 vùng mạnh nhất | 99/130 |
| Ảnh tốt bị cảnh báo | 7/50 (5 REVIEW, 2 chụp xác nhận) |
| Nếu cho nhánh R vào kết luận | 18/30 ảnh lỗi bị chuyển sang chụp xác nhận / chụp lại thay vì FAIL |

→ **Kết luận vẫn như cũ:** nhánh R so một ảnh mẫu không phù hợp để kết luận khi kiểm tra bo khác bo mẫu.

## 3. PatchCore theo vị trí

- **Đặc trưng:** WideResNet-50 ImageNet, lớp 2 + 3, bước 8 px, 256 kênh.
- **So sánh:** mỗi ô so với bộ nhớ ảnh tốt **cùng vị trí ±8 px** sau khi căn. PatchCore gốc (so với mọi vị trí) sẽ coi lỗ bị lấp thiếc giống một pad ở chỗ khác.
- **Chuẩn hóa điểm:** chia cho phân vị 99,9% của điểm ảnh tốt, đo bằng cách giữ từng bo ra ngoài bộ nhớ.

### 3.1 Phát triển trên B/C (giữ một bo ra, D/G không đụng tới)

- **Cách chia:** mỗi lượt giữ một bo ra. Bộ nhớ gồm ảnh tốt c01–c08 của các bo còn lại; kiểm tra toàn bộ ảnh tốt và ảnh lỗi của bo bị giữ ra.
- **Số lượng:** 8 bo, 120 ảnh tốt, 171 ảnh lỗi, 1.589 nhãn.
- **Quy tắc chọn trước khi chạy:**
  - Ngưỡng = điểm nhỏ nhất sao cho ≤ 5% ảnh tốt bị cảnh báo.
  - Giữa hai biến thể chuẩn hóa ("global" và "theo từng ô"), chọn biến thể khoanh trúng nhiều nhãn hơn ở ngưỡng riêng của nó.
- **Tiêu chí chấm chặt:** một nhãn tính là trúng khi điểm bản đồ **bên trong hộp nhãn** đạt ngưỡng.

| Biến thể | Ngưỡng | Ảnh tốt bị cảnh báo | Ảnh lỗi vượt ngưỡng | Nhãn trúng (chấm chặt) | Vùng vượt ngưỡng ngoài nhãn |
|---|---|---|---|---|---|
| **global (chọn)** | **1,30** | 6/120 | **171/171** | **1.275/1.589 (80%)**: B 913/969, C 362/620 | 31 trên ảnh lỗi (26 ở B_03), 7 trên ảnh tốt |
| theo từng ô | 1,94 | 6/120 | 171/171 | 1.096/1.589 | 29 / 6 |

Phiên bản chạy trước khi sửa căn chỉnh chỉ có 1–7/171 ảnh lỗi vượt ngưỡng. Phần lớn khác biệt là do căn sai, không phải do PatchCore.

### 3.2 Kiểm thử một lần trên D/G (ngưỡng 1,30 và biến thể global đã khóa)

Cùng giao thức với nhánh R: bộ nhớ chỉ lấy ảnh tốt của **bo mẫu D_00 / G_00**, kiểm tra D_01, D_02, G_01.

| Chỉ số (50 ảnh tốt, 30 ảnh lỗi, 130 nhãn) | Chỉ YOLO | Nhánh R (đã sửa) | **PatchCore** |
|---|---|---|---|
| Nhãn khoanh trúng ở ngưỡng đã khóa | 126/130 | 42/130 | **111/130 (85%)** |
| Ảnh tốt bị cảnh báo | 0/50 | 7/50 | 5/50 (cả 5 ở G_01; D: 0/40) |
| Ảnh lỗi vượt ngưỡng (trên số ảnh căn được) | – | – | 27/27 |
| Ảnh không căn được | – | 3 | 3 (D_01) |
| YOLO ∪ nhánh so ảnh | – | 126/130 | **128/130** |

- **Nhãn trúng theo bo:** D_01 26/40, D_02 40/45, G_01 45/45.
- **Thử thêm, bộ nhớ 2 bo D** (D_00 + một bo D khác, kiểm tra bo còn lại): 0/40 ảnh tốt bị cảnh báo, 56/85 nhãn trúng. Cảnh báo nhầm ở G là do bộ nhớ chỉ có 10 ảnh của một bo.
- **Kết luận cả bo trong app (YOLO + PatchCore):** 30/30 ảnh lỗi FAIL. 3 ảnh PatchCore không căn được vẫn giữ FAIL của YOLO. 45/50 ảnh tốt ra "không thấy lỗi", 5 ảnh ra REVIEW.

## 4. App mới `D:\FPTU\KLTN\PCB_PatchCore_AOI`

- Bản sao của `PCB_AOI_UI`; thiết kế A, robot và YOLO giữ nguyên.
- Thiết kế B/C/D/G dùng **YOLO + PatchCore**:
  - PatchCore chỉ có quyền đẩy sang **REVIEW**, không tự kết luận FAIL.
  - Không căn được bo: giữ FAIL của YOLO nếu có, nếu không thì báo chụp lại.
- 69/69 ca kiểm thử đạt (57 cũ + 12 mới), đã chạy thử trên giao diện thật.
- Tốc độ: PatchCore khoảng 0,8–1,2 s mỗi bo; lần đầu chọn một thiết kế khoảng 2,6 s.
- Bộ nhớ khoảng 2 GB nên chưa đóng zip. Chi tiết trong `HUONG_DAN_PATCHCORE.md`.

## 5. Đoạn đề xuất cho báo cáo

**Mục 6.3 (thay đoạn mở rộng nhánh R):**

> Để mở rộng nhánh so ảnh ra ngoài bo A, nhóm dựng ảnh mẫu cho B, C, D, G từ ảnh tốt của một bo mỗi thiết kế. Khi kiểm tra, nhóm phát hiện bước căn chỉnh SIFT–RANSAC trên toàn ảnh thường khớp theo nền đen đứng yên: theo tiêu chí chồng khít vùng bo (IoU ≥ 0,95), 290/421 ảnh B/C/D/G bị căn sai. Nhóm sửa bằng cách chỉ lấy điểm đặc trưng trong vùng bo, kèm khớp hai chiều, căn theo viền bo với ECC, và kiểm tra chồng khít sau khi căn. Sau khi sửa, 414/421 ảnh căn đúng; 7 ảnh D còn lại được báo chụp lại. Với căn chỉnh đúng, nhánh so ảnh mẫu ở ngưỡng giữ ≤ 5% cảnh báo nhầm trên B/C chỉ khoanh được 42/130 nhãn D/G. Bước kiểm tra chất lượng ảnh của nhánh này cũng từ chối phần lớn ảnh của bo khác bo mẫu. Nhóm thay nhánh so ảnh mẫu bằng PatchCore theo vị trí: đặc trưng WideResNet-50, mỗi ô chỉ so với bộ nhớ ảnh tốt cùng vị trí ±8 px. Ngưỡng 1,30 được chọn trên B/C bằng cách giữ từng bo ra ngoài bộ nhớ (6/120 ảnh tốt bị cảnh báo; 1.275/1.589 nhãn có điểm trong hộp vượt ngưỡng). Khi khóa ngưỡng và kiểm tra trên D/G với bộ nhớ chỉ từ bo mẫu D_00, G_00, PatchCore khoanh được 111/130 nhãn, cảnh báo nhầm 5/50 ảnh tốt (đều ở G_01, khi bộ nhớ G chỉ có một bo), và hợp với YOLO tìm được 128/130 nhãn so với 126/130 của riêng YOLO.

**Bảng A2 — thay dòng B/C/D/G:**

| Bo | Nguồn | Thành phần | Trạng thái |
|---|---|---|---|
| B, C, D, G | Ảnh tốt c01–c08 của các bo cùng thiết kế; ảnh mẫu từ B_04, C_00, D_00, G_00 | Bộ nhớ đặc trưng PatchCore, mặt nạ vùng bo, ngưỡng 1,30 | Dùng trong bản PCB_PatchCore_AOI: chỉ đẩy sang REVIEW |

**Mục 7.2 Hạn chế (thêm):**

> PatchCore cần ảnh tốt của chính thiết kế, tốt nhất từ hai bo trở lên. Khi bộ nhớ chỉ có một bo G, 5/10 ảnh tốt G_01 bị đưa sang REVIEW. Ngưỡng 1,30 lấy từ 120 ảnh tốt của 8 bo B/C, chưa phải hiệu chuẩn sản xuất. PatchCore không phân loại lỗi nên chỉ dùng để yêu cầu xem lại. Bo A không có bo tốt nên vẫn dùng nhánh so ảnh mẫu cũ.

**Mục 7.3 Hướng phát triển (thêm):**

> Bổ sung ảnh tốt của nhiều bo cho mỗi thiết kế. Giảm dung lượng bộ nhớ bằng coreset. Thử các backbone mới hơn (ví dụ DINOv2), và hiệu chuẩn lại ngưỡng trên một bộ kiểm thử mới.

## 6. Tệp nguồn

| Nội dung | Vị trí |
|---|---|
| Kiểm tra căn chỉnh | `PCB_PatchCore_AOI\results\patchcore_20261124\alignment_audit.csv`, `alignment_audit_summary.json` |
| PatchCore B/C giữ một bo ra | `bc_lobo_summary.json` (global), `bc_lobo_summary_pos.json`; dữ liệu thô `bc_lobo_raw*.json` |
| PatchCore D/G | `test_dg_summary.json`, `test_dg_raw.json` (cùng thư mục) |
| Nhánh R đã sửa | `PCB_DualPath_AOI\results\branch_r_bcdg_20261125_boardalign\` (`calib_bc_summary.json`, `test_dg_summary.json`) |
| Bản chạy cũ (căn sai), chỉ để đối chiếu | `results\patchcore_20261124\_old_alignment\`, `PCB_DualPath_AOI\results\branch_r_bcdg_20261124\` |
