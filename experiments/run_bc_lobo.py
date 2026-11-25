"""Leave-one-board-out PatchCore on the training-side designs B and C. D/G are not read here.

For each held-out board: memory = GOOD captures c01..c08 of every OTHER board of the design; norm from
leave-one-board-out scores inside that memory; test = all GOOD (v0) and all defect captures of the board.

Pre-registered decision threshold (fixed before running):
  P_REVIEW_SCORE = smallest value on the grid 0.50, 0.51, ..., 3.00 such that at most 5% of the held-out
  GOOD captures of B and C have a region with peak >= that value. One global value.
Comparison with Branch R is also reported on R's calibration subset (B_00..B_03, C_01, C_02).
"""
import json, sys, time
import numpy as np
from pclib import *

# variant: "global" (one norm per memory) or "pos" (per-cell normalization, fit_positional). Both run on B/C
# only; the variant with more B/C box hits at its own <=5%-false-alarm threshold is kept (tie -> global).
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "global"
FIT = fit_positional if VARIANT == "pos" else fit_with_norm
TAG = "" if VARIANT == "global" else "_" + VARIANT
OUT = ROOT / "results" / "patchcore_20261124"; OUT.mkdir(parents=True, exist_ok=True)
BOARDS = {"B": ["B_00", "B_01", "B_02", "B_03", "B_04"], "C": ["C_00", "C_01", "C_02"]}
bb = Backbone(WEIGHTS)
rows, t0 = [], time.time()
for design, boards in BOARDS.items():
    d = Design(design, bb)
    cache = {b: [(r[1].cpu(), r[2].cpu()) for c in MEMORY_CAPS if (r := d.features(DATA / f"{b}_v0_c{c:02d}.png"))]
             for b in boards}
    print(design, "memory captures", {b: len(v) for b, v in cache.items()}, f"{time.time()-t0:.0f}s", flush=True)
    for held in boards:
        norm = FIT(d, {b: cache[b] for b in boards if b != held})
        tests = [("GOOD", p) for p in sorted(DATA.glob(f"{held}_v0_c*.png"), key=cap_no)] + \
                [("DEFECT", p) for p in sorted(DEFECT.glob(f"{held}_v*_c*.png"))]
        for kind, p in tests:
            r = d.features(p)
            if r is None:
                rows.append(dict(design=design, board=held, kind=kind, image=p.stem, aligned=False)); continue
            gt = gt_boxes(p.stem) if kind == "DEFECT" else []
            s = score_image(d, *r, gt=gt)
            rows.append(dict(design=design, board=held, kind=kind, image=p.stem, aligned=True, norm=norm, gt=gt, **s))
        print(design, held, "norm", round(norm, 3), len(rows), f"{time.time()-t0:.0f}s", flush=True)
    del cache, d
    torch.cuda.empty_cache()
(OUT / f"bc_lobo_raw{TAG}.json").write_text(json.dumps(rows), encoding="utf-8")

ok = [r for r in rows if r["aligned"]]
good = [r for r in ok if r["kind"] == "GOOD"]
peak = lambda r: max([g["score"] for g in r["regions"]], default=0.0)
gpk = np.array([peak(r) for r in good])
grid = np.round(np.arange(0.50, 3.0001, 0.01), 2)
T = float(next(s for s in grid if (gpk >= s).mean() <= 0.05))


def stats(sub, t):
    g = [r for r in sub if r["kind"] == "GOOD"]; d = [r for r in sub if r["kind"] == "DEFECT"]
    box = lambda r, thr: sum(any(hits(gt, x["box"]) for x in r["regions"] if x["score"] >= thr) for gt in r["gt"])
    return dict(good=len(g), good_flagged=sum(peak(r) >= t for r in g),
                defect_images=len(d), defect_images_flagged=sum(peak(r) >= t for r in d),
                boxes=sum(len(r["gt"]) for r in d), box_hits_top8=sum(box(r, 0) for r in d),
                box_hits_at_T=sum(box(r, t) for r in d),
                # strict: label detected if the score map inside the label box reaches t
                label_peak_hits_at_T=sum(sum(pk >= t for pk in r["gt_peak"]) for r in d),
                # components (score >= 1.0) with peak >= t lying outside every label
                false_comps_on_defect_images=sum(sum(c[0] >= t and c[1] for c in r["comps"]) for r in d),
                false_comps_on_good_images=sum(sum(c[0] >= t for c in r["comps"]) for r in g))


r_subset = [r for r in ok if r["board"] in ("B_00", "B_01", "B_02", "B_03", "C_01", "C_02")]
summary = dict(rule="smallest grid peak score with <=5% held-out B/C GOOD captures flagged", P_REVIEW_SCORE=T,
               not_aligned=len(rows) - len(ok), all=stats(ok, T), branch_r_subset=stats(r_subset, T),
               per_design={dz: stats([r for r in ok if r["design"] == dz], T) for dz in BOARDS},
               per_board={b: stats([r for r in ok if r["board"] == b], T) for bs in BOARDS.values() for b in bs},
               good_peak_pct=dict(zip(("50", "90", "95", "100"), np.percentile(gpk, [50, 90, 95, 100]).round(3).tolist())),
               false_alarm_curve=[(float(s), float((gpk >= s).mean())) for s in grid[::5]])
(OUT / f"bc_lobo_summary{TAG}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "false_alarm_curve"}, indent=1))
