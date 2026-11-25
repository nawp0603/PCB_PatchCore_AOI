"""PatchCore scores for the R-vs-PatchCore comparison, under the SAME reference as Branch R:
memory = GOOD (v0) captures of the template board only (B_04, C_00, D_00, G_00).
  dev  (B/C): B_00..B_03 (memory B_04), C_01, C_02 (memory C_00)  -- same images as R calibration
  test (D/G): D_01, D_02 (memory D_00), G_01 (memory G_00)          -- same images as R test
Per capture: image score (max of the score map), and for every threshold t on GRID the connected
components of score >= t (area >= 60 px, frame coords): how many, how many hit a label, and which labels
are hit (criterion pclib.hits = same as Branch R: IoU >= 0.1 or centre containment). D/G also get YOLO
predictions and the app regions (components >= 1.0, top 8) for the system-level decision.
Output: PCB_DualPath_AOI/results/so_sanh_R_vs_PatchCore_20261125/patchcore_dump.json
"""
import json, time
from pclib import *
import src.core.real_ab_pipeline as pipe

OUT = Path(r"D:\FPTU\KLTN\PCB_DualPath_AOI\results\so_sanh_R_vs_PatchCore_20261125"); OUT.mkdir(parents=True, exist_ok=True)
GRID = [round(x, 2) for x in np.arange(0.60, 3.001, 0.05)]
PLAN = [("dev", "B", "B_04", ["B_00", "B_01", "B_02", "B_03"]), ("dev", "C", "C_00", ["C_01", "C_02"]),
        ("test", "D", "D_00", ["D_01", "D_02"]), ("test", "G", "G_00", ["G_01"])]
CLASS = {0: "missing_hole", 1: "open_circuit", 2: "spur"}


def gt_with_class(stem, w=2592, h=1944):
    p = LABELS / f"{stem}.txt"; out = []
    if p.exists():
        for line in p.read_text().split("\n"):
            if line.strip():
                k, x, y, bw, bh = line.split(); x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
                out.append((CLASS[int(k)], [x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2]))
    return out


bb = Backbone(WEIGHTS)
yolo = pipe.YOLODetector(ROOT / "assets/weights/pcb_camera/real_3class.pt", conf_threshold=0.20, imgsz=2592)
rows, t0 = [], time.time()
for split, design, ref, boards in PLAN:
    d = Design(design, bb)
    mem = [(r[1].cpu(), r[2].cpu()) for p in sorted(DATA.glob(f"{ref}_v0_c*.png"), key=cap_no) if (r := d.features(p))]
    norm = fit_with_norm(d, {ref: mem})
    for b in boards:
        for kind, p in [("GOOD", p) for p in sorted(DATA.glob(f"{b}_v0_c*.png"), key=cap_no)] + \
                       [("DEFECT", p) for p in sorted(DEFECT.glob(f"{b}_v*_c*.png"))]:
            img = read(p)
            gt = gt_with_class(p.stem) if kind == "DEFECT" else []
            row = dict(split=split, design=design, board=b, kind=kind, image=p.stem, norm=norm,
                       labels=[c for c, _ in gt])
            if split == "test":
                row["yolo"] = [dict(bbox=[float(v) for v in it["bbox"]], class_name=it["class_name"],
                                    class_id=int(it["class_id"]), conf=float(it["conf"]))
                               for it in yolo.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), imgsz=2592, conf=0.20,
                                                      iou=0.7, fallback_on_oom=False)]
            a = d.align(img)
            if a is None:
                row.update(aligned=False); rows.append(row); continue
            H, warped, valid = a
            feat, cells = d.pc.features(warped, valid)
            m = d.pc.score_map(feat, cells)
            row.update(aligned=True, image_score=float(m.max()), grid=[])
            for t in GRID:
                comps = d.pc.regions(m, top_k=10000, seed=t)
                boxes = [project_box(c.box, H) for c in comps]
                hit_lab = [any(hits(g, bx) for bx in boxes) for _, g in gt]
                row["grid"].append([t, len(boxes), sum(any(hits(g, bx) for _, g in gt) for bx in boxes), hit_lab])
                if t == 1.0:
                    inv = np.linalg.inv(H)
                    row["app_regions"] = [dict(center=cv2.perspectiveTransform(np.float32([c.center])[None], inv)[0][0].tolist(),
                                               box=bx, score=float(c.score), area_template=float(c.area))
                                          for c, bx in sorted(zip(comps, boxes), key=lambda z: -z[0].score)[:8]]
            rows.append(row)
        print(split, b, len(rows), f"{time.time()-t0:.0f}s", flush=True)
    del d, mem
    torch.cuda.empty_cache()
(OUT / "patchcore_dump.json").write_text(json.dumps(dict(grid=GRID, rows=rows)), encoding="utf-8")
print("DONE", len(rows), flush=True)
