# PCB AOI - ban lam UI/UX

Ban sao sach cua PCB_DualPath_AOI, chi gom nhung gi can de chay UI tren bo mach that.

## Chay
- `run_ui.bat` : UI voi model `real_3class.pt` (YOLO11n, nhanh A anh nguyen kho + nhanh B hieu chuan).
  Model 3 lop (23/11/2026): missing_hole = mach tit -> FAIL, open_circuit = mach ho -> FAIL,
  spur = thua thiec/dong -> REWORK. Train tren thiet ke A, B, C; D va G de rieng lam tap test.
  Model cu 6 lop van chay duoc: `run_ui.bat --weights real_all_boards.pt`.
- `run_ui.bat --camera-quality` : chi nhanh B hieu chuan.
- `run_tests.bat` : chay test.
- Anh that de thu khong can camera: thu muc `samples/`, nap bang nut "Open Image File...".

## Moi truong Python
Hai file .bat dung, theo thu tu uu tien: `.venv` trong thu muc nay, python ghi trong
`venv_python.txt` (tao boi --link-venv, tro ve venv cua repo goc), roi `python` cua he thong.
Tao venv rieng cho ban nay:
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Khong co trong ban nay
- Che do mau PKU cu (chay UI khong co tham so trong src/ui/tkinter_app.py), anh mau PKU,
  templates PKU, cac model cu.
- Du lieu train, script train/do, ket qua thi nghiem, ban build/exe, server FastAPI, ung dung WPF.
Cac thu do van nam o repo goc.

## Phan loai bang robot
Chay 2 cua so: `run_ui_robot.bat` va `run_robot_sim.bat` (gia lap) hoac `run_robot.bat` (Dobot that).
- UI ghi lenh vao `robot_io/outbox/<id>.json`, robot lam xong ghi `robot_io/inbox/<id>.json`.
  UI khoa nut KIEM TRA cho toi khi robot bao xong (toi da 120 s).
- Quy tac (src/decision/sorting.py):
  FAIL/REWORK -> khay LOI tu dong; KHONG THAY LOI -> khay DAT tu dong;
  REVIEW -> nguoi van hanh bam Loi / Dat / Khong gap; CHUP LAI, CHUP XAC NHAN, loi -> khong gap.
- Nhanh B chi day bo vao REVIEW khi vung la co diem >= 12 (B_REVIEW_SCORE trong
  src/core/real_ab_pipeline.py). Vung la yeu van ve tren anh nhung khong chan khay DAT.
- Loai bo: "Thiet ke A" dung A + B. "Bo loai khac" chi dung nhanh A (B khong co anh mau cho thiet ke khac).
- Robot that: `pip install pydobot`, sua cong COM va toa do trong `configs/robot_positions.json`
  (gia tri mau, phai day lai tren tram). Vi tri home phai nam ngoai khung hinh camera.
  Lenh cu hon 30 s bi bo qua (expired), robot loi thi dung han va bao ve UI.

## Kiem tra nhanh A tren thiet ke khac
`python scripts/check_a_other_designs.py --images <thu muc anh>` -> `results/a_other_designs/`
(anh *_A.jpg + a_only.csv). Box nao khong nam tren loi that la bao gia.
