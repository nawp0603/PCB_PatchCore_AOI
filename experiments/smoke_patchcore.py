"""Quick check on design B: memory B_04 + B_01..B_03 (c01-c08), score B_00 GOOD c01 and B_00 defect v1_c01."""
import time
from pclib import *

bb = Backbone(WEIGHTS)
d = Design("B", bb)
t = time.time()
mem = {}
for b in ["B_01", "B_02", "B_03", "B_04"]:
    mem[b] = []
    for c in MEMORY_CAPS:
        r = d.features(DATA / f"{b}_v0_c{c:02d}.png")
        if r: mem[b].append(r[1:])
print("features", sum(map(len, mem.values())), f"{time.time()-t:.1f}s", "shape", next(iter(mem.values()))[0][0].shape)
t = time.time(); norm = fit_with_norm(d, mem); print("norm", round(norm, 3), f"{time.time()-t:.1f}s")
for p in [DATA / "B_00_v0_c01.png", DATA / "B_00_v0_c07.png", DEFECT / "B_00_v1_c01.png", DEFECT / "B_00_v3_c01.png"]:
    t = time.time(); H, feat, cells = d.features(p); s = score_image(d, H, feat, cells)
    gt = gt_boxes(p.stem)
    hit = sum(any(hits(g, r["box"]) for r in s["regions"]) for g in gt)
    print(p.stem, "max", round(s["max_score"], 2), "regions", [round(r["score"], 1) for r in s["regions"]],
          f"hit {hit}/{len(gt)}", f"{time.time()-t:.2f}s")
print("gpu MB", torch.cuda.max_memory_allocated() // 2**20)
