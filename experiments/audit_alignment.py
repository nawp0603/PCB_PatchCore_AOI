"""Audit: old alignment (whole image, as deployed) vs board-only alignment, on every B/C/D/G capture.
Correctness check independent of SIFT: IoU of the green board region of the warped capture with the
reference board region. Writes results/patchcore_20261124/alignment_audit.csv"""
import csv, json, time
from pclib import *
from src.core.board_align import board_region

OUT = ROOT / "results" / "patchcore_20261124"; OUT.mkdir(parents=True, exist_ok=True)
rows, t0 = [], time.time()
for design in "BCDG":
    d = Design(design, None, board_only=True)
    ref = board_region(d.cam.template)
    for p in sorted(DATA.glob(f"{design}_*_v0_c*.png")) + sorted(DEFECT.glob(f"{design}_*_v*_c*.png")):
        img = read(p)
        row = dict(design=design, image=p.stem)
        for tag, H in (("old", d.cam._align(img)), ("new", d.aligner.align(img))):
            if H is None:
                row[f"{tag}_iou"] = None; continue
            w = cv2.warpPerspective(img, H, (d.w, d.h))
            b = board_region(w)
            if tag == "new":
                row["new_method"] = d.aligner.last.get("method")
            row[f"{tag}_iou"] = round(float((b & ref).sum() / max((b | ref).sum(), 1)), 4)
            row[f"{tag}_shift"] = round(float(np.hypot(H[0, 2], H[1, 2])), 1)
        rows.append(row)
    print(design, len(rows), f"{time.time()-t0:.0f}s", flush=True)
with open(OUT / "alignment_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k not in ("design", "image"), k)))
    w.writeheader(); w.writerows(rows)
bad = lambda tag, r: r.get(f"{tag}_iou") is None or r[f"{tag}_iou"] < 0.95
summ = {dz: dict(n=sum(r["design"] == dz for r in rows),
                 old_bad=sum(bad("old", r) for r in rows if r["design"] == dz),
                 new_bad=sum(bad("new", r) for r in rows if r["design"] == dz),
                 old_bad_v0=sum(bad("old", r) for r in rows if r["design"] == dz and "_v0_" in r["image"]),
                 old_bad_defect=sum(bad("old", r) for r in rows if r["design"] == dz and "_v0_" not in r["image"]))
        for dz in "BCDG"}
(OUT / "alignment_audit_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
print(json.dumps(summ, indent=1))
print("worst new:", sorted(((r.get("new_iou") or 0), r["image"]) for r in rows)[:6])
