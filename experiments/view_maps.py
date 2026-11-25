"""Heatmaps for one B fold (held-out board, 'pos' variant) -> PCB_DualPath_AOI/results/patchcore_view (viewable)."""
import sys
from pclib import *

OUTV = Path(r"D:\FPTU\KLTN\PCB_DualPath_AOI\results\patchcore_view"); OUTV.mkdir(exist_ok=True)
design, held, names = sys.argv[1], sys.argv[2], sys.argv[3:]
boards = {"B": ["B_00", "B_01", "B_02", "B_03", "B_04"], "C": ["C_00", "C_01", "C_02"]}[design]
d = Design(design, Backbone(WEIGHTS))
mem = {b: [r[1:] for c in MEMORY_CAPS if (r := d.features(DATA / f"{b}_v0_c{c:02d}.png"))] for b in boards if b != held}
fit_positional(d, mem)
for n in names:
    p = (DATA if "_v0_" in n else DEFECT) / f"{n}.png"
    img = read(p); H, warped, valid = d.align(img)
    feat, cells = d.pc.features(warped, valid)
    m = d.pc.score_map(feat, cells)
    heat = cv2.applyColorMap(np.clip(m / 1.3 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(warped, 0.55, heat, 0.45, 0)
    for r in d.pc.regions(m)[:5]:
        cv2.rectangle(over, r.box[:2], r.box[2:], (255, 255, 255), 3)
        cv2.putText(over, f"{r.score:.2f}", (r.box[0], max(r.box[1] - 8, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.imencode(".jpg", cv2.resize(over, None, fx=.5, fy=.5))[1].tofile(str(OUTV / f"{design}_{n}.jpg"))
    print(n, "max", round(float(m.max()), 3), [round(r.score, 2) for r in d.pc.regions(m)[:5]])
