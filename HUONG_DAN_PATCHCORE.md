# PCB_PatchCore_AOI — app kiểm tra với PatchCore cho thiết kế B, C, D, G

Folder này là bản sao của `PCB_AOI_UI` (24/11/2026). Mọi phần cũ giữ nguyên: YOLO 3 lớp, nhánh B của thiết kế A, robot, giao diện. Điểm khác duy nhất: với thiết kế **B, C, D, G**, nhánh so ảnh được thay bằng **PatchCore theo vị trí**. `PCB_AOI_UI` không bị sửa bởi folder này.

## Chạy

- **Giao diện:** `run_ui.bat`, hoặc `run_ui_robot.bat` nếu chạy cùng robot. Ở mục **LOẠI BO**:
  - **Thiết kế A (nhánh A + B):** như cũ.
  - **Thiết kế B / C / D / G (nhánh A + PatchCore):** YOLO kết luận lớp lỗi. PatchCore khoanh vùng khác với ảnh tốt (khung **tím**, ký hiệu `P`).
  - **Bo loại khác (chỉ nhánh A):** như cũ.
- **Kiểm thử:** `run_tests.bat` (69 ca).
- **Dựng lại bộ nhớ PatchCore:** `.venv\Scripts\python.exe scripts\build_patchcore_memory.py <thư mục Data_BCD>`

## Quy tắc kết luận với B/C/D/G

| Tình huống | Kết luận |
|---|---|
| YOLO có lỗi đủ ngưỡng | FAIL. PatchCore không thay đổi. |
| Không có lỗi YOLO, có vùng PatchCore điểm ≥ 1,30 nằm ngoài hộp YOLO | REVIEW: người vận hành chọn khay |
| Không có gì vượt ngưỡng | KHÔNG THẤY LỖI (chưa hiệu chuẩn) |
| Không căn được bo với ảnh mẫu | Giữ FAIL của YOLO nếu có, nếu không thì CHỤP LẠI |

- PatchCore **không bao giờ tự kết luận FAIL**, vì nó không biết loại lỗi. Nó chỉ buộc người vận hành xem lại.
- Vùng có điểm từ 1,0 đến dưới 1,30 được ghi là "vùng lạ yếu", không ảnh hưởng kết luận.
- Ngưỡng 1,30 được hiệu chuẩn trên B/C, theo kiểu giữ từng bo ra ngoài bộ nhớ, sao cho tối đa 5% ảnh tốt bị cảnh báo. Ngưỡng này chưa được hiệu chuẩn trên dây chuyền thật.

## Cách PatchCore hoạt động ở đây

1. **Căn chỉnh:** ảnh được căn về ảnh mẫu của thiết kế bằng SIFT–RANSAC, **chỉ trong vùng bo**. Sau khi căn, vùng bo của ảnh phải chồng khít ảnh mẫu, IoU ≥ 0,95; bo D có hàng lỗ giống nhau nên có thêm phương án khớp hai chiều và ECC (`src/core/board_align.py`). Cách căn cũ trên cả ảnh hay khớp nhầm theo nền đen: trên tổng 421 ảnh B/C/D/G có 290 ảnh căn sai; cách mới còn 7 ảnh D không căn được và sẽ báo chụp lại.
2. **Trích đặc trưng:** WideResNet-50 (ImageNet), lớp 2 + lớp 3, bước 8 px, 256 kênh (`src/core/patchcore_branch.py`).
3. **So với bộ nhớ theo vị trí:** mỗi ô chỉ so với các ô **cùng vị trí ±1 ô (±8 px)** trong bộ nhớ ảnh tốt. Lý do: lỗ bị lấp thiếc trông giống pad ở chỗ khác, nên PatchCore gốc (so mọi vị trí) sẽ bỏ sót.
4. **Chuẩn hóa điểm:** chia cho phân vị 99,9% của điểm ảnh tốt, đo bằng cách giữ từng bo ra ngoài bộ nhớ. Điểm 1,0 là mép trên của mức "bình thường".
5. **Bộ nhớ** (`assets/patchcore/<thiết kế>/memory.pt`): ảnh tốt c01–c08 của mọi bo tốt.

| Thiết kế | Bộ nhớ | Dung lượng |
|---|---|---|
| B | 40 ảnh | 895 MB |
| C | 24 ảnh | 349 MB |
| D | 24 ảnh | 593 MB |
| G | 16 ảnh | 189 MB |

   Mỗi lúc GPU chỉ giữ bộ nhớ của một thiết kế.
6. **Thời gian trên RTX 3070:** PatchCore mất khoảng 0,8–1,2 s mỗi lần kiểm tra. Lần đầu chọn một thiết kế mất khoảng 2–2,6 s vì phải nạp bộ nhớ.

## Thư mục

```
src/core/board_align.py          căn chỉnh theo vùng bo + kiểm tra chồng khít
src/core/patchcore_branch.py     backbone, khoảng cách theo vị trí, bản đồ điểm, vùng
src/core/patchcore_inspector.py  PatchCore cho app (căn chỉnh -> điểm -> vùng trong tọa độ ảnh)
src/core/real_ab_pipeline.py     YOLO + nhánh B (A) / PatchCore (B, C, D, G), combine_patchcore()
assets/patchcore/<D>/            memory.pt, settings.json (ngưỡng 1,30)
assets/branch_r_profiles/<D>/    ảnh mẫu + mặt nạ vùng bo (khung tham chiếu)
assets/weights/wide_resnet50_2_imagenet.pth
experiments/                     các thí nghiệm (B/C giữ một bo ra, D/G, kiểm tra căn chỉnh)
results/patchcore_20261124/      số liệu thô và tổng hợp
tests/test_patchcore_branch.py   12 ca kiểm thử mới
tests_old/                       ca kiểm thử của bản "vùng R chỉ xem" (không còn dùng ở folder này)
```

## Giới hạn

- Cần ảnh tốt của **chính thiết kế đó**, tốt nhất từ ≥ 2 bo. Bộ nhớ G chỉ có 1 bo thì 5/10 ảnh tốt G_01 bị REVIEW. Thiết kế A không có bo tốt nên vẫn dùng nhánh B cũ.
- Bộ nhớ trong app gồm cả D_01, D_02, G_01, vì đây là bản demo. Số liệu D/G cho báo cáo lấy từ `experiments/test_dg.py`, bộ nhớ chỉ có D_00/G_00.
- Ngưỡng 1,30 lấy từ 120 ảnh tốt của 8 bo B/C, chưa phải hiệu chuẩn sản xuất.
- Bộ nhớ khoảng 2 GB, nên chưa đóng gói zip được như gói robot. Có thể chạy script để dựng lại khi có thư mục Data_BCD.
