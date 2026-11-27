# PCB AOI - ban lam UI/UX

Ban sao sach cua PCB_DualPath_AOI, chi gom nhung gi can de chay UI tren bo mach that.

## Chay
- `run_ui.bat` : UI voi model `real_3class.pt` (YOLO11n, nhanh A anh nguyen kho + nhanh B hieu chuan).
  Model 3 lop (23/11/2026): missing_hole = mach tit -> FAIL, open_circuit = mach ho -> FAIL,
   spur = thua thiec/dong -> FAIL (nguong 0.35). Train tren thiet ke A, B, C; D va G de rieng lam tap test.
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
- Quy tac (src/decision/sorting.py): YOLO class 0/1/2 -> C11/C12/C13 tu dong;
  KHONG THAY LOI -> A11 tu dong. REVIEW (bao gom PatchCore) bat buoc nguoi van hanh
  chon A11/C11/C12/C13 hoac Khong gap. CHUP LAI, CHUP XAC NHAN, loi va YOLO khong
  co lop 0/1/2 -> khong gap.
- Nhanh B chi day bo vao REVIEW khi vung la co diem >= 12 (B_REVIEW_SCORE trong
  src/core/real_ab_pipeline.py). Vung la yeu van ve tren anh nhung khong chan khay DAT.
- Loai bo: "Thiet ke A" dung A + B. "Bo loai khac" chi dung nhanh A (B khong co anh mau cho thiet ke khac).
- Robot that dung giao thuc serial truc tiep cua Dobot Magician Lite (`pyserial`, da co trong
  `requirements.txt`), khong can DobotLab, DobotLink hoac `pydobot`. Chay
  `run_robot.bat --check-config` truoc; sua cong COM va xac nhan cac toa do B11/A11/C11 trong
  `configs/robot_positions.json`. UI Robot co quet COM, worker status, Home, dung phan mem va day
  Home/B11/A11/C11/C12; luu diem luon dat `station_calibrated` ve `false`. Vi tri home phai nam
  ngoai khung hinh camera. A12/A13/C13 duoc suy ra tu vector C11 -> C12.
- Sau khi day va kiem tra Home/B11/A11/C11/C12 o toc do an toan, nhan `XAC NHAN HIEU CHUAN` trong UI.
  Phai ngat worker day truoc khi xac nhan, sau do ket noi lai robot that de chuyen sang trang thai san sang.
- Chi mot worker duoc phep dieu khien tram. Lenh cu hon 30 s bi bo qua; robot loi thi dung han va
  bao ve UI. Sau khi dong worker, co the gui dung mem bang `run_robot_stop.bat --port COM15`; uu tien
  E-stop vat ly khi canh tay khong an toan hoac khong phan hoi.
- Automation trong UI chi bat khi camera da ket noi va worker robot dang san sang. Robot that can
  `station_calibrated: true`; robot gia lap duoc phep. Vong lap chi chup bo tiep theo sau ACK `done`;
  REVIEW cho nguoi van hanh chon khay, con RECAPTURE/CONFIRM/loi thi dung an toan. Dung
  `run_ui_robot_all.bat` de mo UI, tu ket noi camera dau tien tim thay va tu bat mo phong; dung
  `run_ui_robot_all.bat --real` sau khi da hieu chuan tram that.
- Truoc AOI, Automation kiem tra su hien dien PCB tren toan khung hinh camera tai B11, khong dung ROI
  hay B12. Neu B11 trong, robot khong gap, he thong cho nap bo trong 5 giay va tu dung sau tong 10 giay.

## Kiem tra nhanh A tren thiet ke khac
`python scripts/check_a_other_designs.py --images <thu muc anh>` -> `results/a_other_designs/`
(anh *_A.jpg + a_only.csv). Box nao khong nam tren loi that la bao gia.
