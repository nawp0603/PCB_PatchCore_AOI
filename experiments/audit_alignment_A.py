"""Board-overlap audit of the deployed design-A alignment (CalibratedCameraInspection._align, template_A)."""
import json
from pathlib import Path
import cv2, numpy as np
from pclib import ROOT, read
from src.core.calibrated_camera import CalibratedCameraInspection
from src.core.board_align import board_region, BoardAligner

cam = CalibratedCameraInspection(ROOT / "results")
ref = board_region(cam.template)
new = BoardAligner(cam.template)
dirs = [Path(r"D:\FPTU\SU26\Data_AnhThat\Finalfinaldataset"), Path(r"D:\FPTU\SU26\Data_AnhThat\Additional_Dataset_For_A020405"),
        Path(r"D:\FPTU\SU26\Data_AnhThat\Data_BCD\MachA_da_tao_loi")]
paths = sorted({p for d in dirs for p in d.rglob("A_0*.png")})
rows = []
for p in paths:
    img = read(p); h, w = cam.gray.shape
    row = dict(image=p.name)
    for tag, H in (("old", cam._align(img)), ("new", new.align(img))):
        if H is None:
            row[tag] = None; continue
        b = board_region(cv2.warpPerspective(img, H, (w, h)))
        row[tag] = round(float((b & ref).sum() / max((b | ref).sum(), 1)), 3)
    rows.append(row)
bad = lambda k: sum(r[k] is None or r[k] < 0.95 for r in rows)
print(len(rows), "images | old bad", bad("old"), "| new bad", bad("new"))
print([r for r in rows if r["old"] is None or r["old"] < 0.95][:12])
