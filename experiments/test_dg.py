"""One-shot test of PatchCore on D/G with everything locked on B/C (same protocol as the Branch R test).

Variant (pre-registered): 'global' vs 'pos' -> the one with more B/C box hits at its own <=5% threshold
(tie -> global). Threshold = that variant's P_REVIEW_SCORE. Nothing here is tuned on D/G.
Primary: memory = GOOD captures of the template board only (D_00, G_00) -> test D_01, D_02, G_01.
Secondary (labelled as such): two-board memory for D (D_00 + D_02 -> test D_01; D_00 + D_01 -> test D_02).
YOLO = deployed real_3class.pt (imgsz 2592, conf 0.20, NMS 0.7); decision = deployed combine(), with
PatchCore regions in place of Branch B regions.
"""
import json
from pclib import *
import src.core.real_ab_pipeline as pipe

RES = ROOT / "results" / "patchcore_20261124"
sg = json.loads((RES / "bc_lobo_summary.json").read_text(encoding="utf-8"))
sp = json.loads((RES / "bc_lobo_summary_pos.json").read_text(encoding="utf-8"))
# selection uses the strict label metric (score inside the label box >= own threshold), fixed before D/G
hit = lambda s: s["all"]["label_peak_hits_at_T"]
VARIANT, CAL = ("pos", sp) if hit(sp) > hit(sg) else ("global", sg)
FIT = fit_positional if VARIANT == "pos" else fit_with_norm
T = float(CAL["P_REVIEW_SCORE"])
pipe.B_REVIEW_SCORE = T
print("locked:", VARIANT, "T", T, "| B/C hits global", hit(sg), "pos", hit(sp), flush=True)

bb = Backbone(WEIGHTS)
yolo = pipe.YOLODetector(ROOT / "assets/weights/pcb_camera/real_3class.pt", conf_threshold=0.20, imgsz=2592)


def predict(img):
    return [{**it, "source": "yolo"} for it in yolo.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), imgsz=2592, conf=0.20,
                                                            iou=0.7, fallback_on_oom=False)]


def mem_of(d, board):
    return [r[1:] for p in sorted(DATA.glob(f"{board}_v0_c*.png"), key=cap_no) if (r := d.features(p))]


def run(protocol, design, memory_boards, test_board):
    d = Design(design, bb)
    norm = FIT(d, {b: mem_of(d, b) for b in memory_boards})
    out = []
    for kind, p in [("GOOD", p) for p in sorted(DATA.glob(f"{test_board}_v0_c*.png"), key=cap_no)] + \
                   [("DEFECT", p) for p in sorted(DEFECT.glob(f"{test_board}_v*_c*.png"))]:
        img = read(p)
        a_items = predict(img)
        gt = gt_boxes(p.stem) if kind == "DEFECT" else []
        r = d.features(p)
        row = dict(protocol=protocol, design=design, board=test_board, kind=kind, image=p.stem, norm=norm,
                   aligned=r is not None, gt=len(gt), a_boxes=len(a_items),
                   concl_a=pipe.combine(a_items, "INSPECT", [])["conclusion"])
        a_hit = [any(iou(g, it["bbox"]) >= 0.3 for it in a_items) for g in gt]
        if r is None:          # deployed behaviour: an unaligned capture asks for a new capture
            row.update(concl_ap="RECAPTURE", peak=None, p_hit_top8=0, p_hit_T=0, ap_hit_T=sum(a_hit), ap_hit_top8=sum(a_hit))
        else:
            s = score_image(d, *r, gt=gt)
            regs = [dict(center=((x["box"][0] + x["box"][2]) / 2, (x["box"][1] + x["box"][3]) / 2), box=x["box"],
                         score=x["score"], area_template=0.0) for x in s["regions"]]
            p8 = [any(hits(g, x["box"]) for x in regs) for g in gt]
            pT = [pk >= T for pk in s["gt_peak"]]          # strict: score inside the label box reaches T
            row["false_comps"] = sum(c[0] >= T and c[1] for c in s["comps"])
            row.update(concl_ap=pipe.combine(a_items, "INSPECT", regs)["conclusion"],
                       peak=max([x["score"] for x in regs], default=0.0),
                       p_hit_top8=sum(p8), p_hit_T=sum(pT), ap_hit_T=sum(a or b for a, b in zip(a_hit, pT)),
                       ap_hit_top8=sum(a or b for a, b in zip(a_hit, p8)))
        row["a_hit"] = sum(a_hit)
        out.append(row)
    print(protocol, test_board, "norm", round(norm, 3), len(out), flush=True)
    return out


rows = []
rows += run("primary", "D", ["D_00"], "D_01") + run("primary", "D", ["D_00"], "D_02") + run("primary", "G", ["G_00"], "G_01")
rows += run("secondary_2board", "D", ["D_00", "D_02"], "D_01") + run("secondary_2board", "D", ["D_00", "D_01"], "D_02")
(RES / "test_dg_raw.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def agg(sub):
    g = [r for r in sub if r["kind"] == "GOOD"]; d = [r for r in sub if r["kind"] == "DEFECT"]
    cnt = lambda rs, k: {v: sum(r[k] == v for r in rs) for v in sorted({r[k] for r in rs})}
    return dict(good=len(g), defect_images=len(d), boxes=sum(r["gt"] for r in d),
                not_aligned=sum(not r["aligned"] for r in sub),
                good_concl_yolo=cnt(g, "concl_a"), good_concl_yolo_patchcore=cnt(g, "concl_ap"),
                defect_concl_yolo=cnt(d, "concl_a"), defect_concl_yolo_patchcore=cnt(d, "concl_ap"),
                good_peak_over_T=sum((r["peak"] or 0) >= T for r in g),
                defect_peak_over_T=sum((r["peak"] or 0) >= T for r in d),
                false_comps_on_good=sum(r.get("false_comps", 0) for r in g),
                false_comps_on_defect=sum(r.get("false_comps", 0) for r in d),
                box_hits=dict(yolo=sum(r["a_hit"] for r in d), patchcore_top8=sum(r["p_hit_top8"] for r in d),
                              patchcore_label_peak_at_T=sum(r["p_hit_T"] for r in d),
                              yolo_or_patchcore_label_peak_at_T=sum(r["ap_hit_T"] for r in d),
                              yolo_or_patchcore_top8=sum(r["ap_hit_top8"] for r in d)))


summary = dict(variant=VARIANT, P_REVIEW_SCORE=T,
               primary=agg([r for r in rows if r["protocol"] == "primary"]),
               primary_per_board={b: agg([r for r in rows if r["protocol"] == "primary" and r["board"] == b])
                                  for b in ("D_01", "D_02", "G_01")},
               secondary_2board_D=agg([r for r in rows if r["protocol"] == "secondary_2board"]))
(RES / "test_dg_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=1, ensure_ascii=False))
