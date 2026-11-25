"""Build the PatchCore memory used by the app for designs B, C, D, G.

memory  = GOOD (v0) captures c01..c08 of every board of the design that has GOOD captures, aligned to the
          design reference (assets/branch_r_profiles/<D>) with board-only SIFT (board_align.py)
norm    = leave-one-board-out (fit_with_norm, variant 'global' selected on B/C)
review  = P_REVIEW_SCORE 1.30, calibrated on B/C leave-one-board-out (results/patchcore_20261124)
The app memory includes every GOOD board, also D_01, D_02, G_01: the D/G numbers in the report come from
experiments/test_dg.py (memory D_00 / G_00 only), not from this memory.
usage: build_patchcore_memory.py [data dir] [designs...]
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "experiments")); sys.path.insert(0, str(ROOT))
import pclib
from pclib import Design, Backbone, WEIGHTS, MEMORY_CAPS, fit_with_norm, cap_no
import torch

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else pclib.DATA
DESIGNS = sys.argv[2:] or ["B", "C", "D", "G"]
BOARDS = {"B": ["B_00", "B_01", "B_02", "B_03", "B_04"], "C": ["C_00", "C_01", "C_02"],
          "D": ["D_00", "D_01", "D_02"], "G": ["G_00", "G_01"]}
P_REVIEW_SCORE = json.loads((ROOT / "results/patchcore_20261124/bc_lobo_summary.json").read_text())["P_REVIEW_SCORE"]
bb = Backbone(WEIGHTS)
for design in DESIGNS:
    d = Design(design, bb)
    memory, used, skipped = {}, [], []
    for b in BOARDS[design]:
        memory[b] = []
        for c in MEMORY_CAPS:
            p = DATA / f"{b}_v0_c{c:02d}.png"
            r = d.features(p) if p.exists() else None
            if r is None:
                skipped.append(p.stem); continue
            memory[b].append((r[1].cpu(), r[2].cpu())); used.append(p.stem)
    memory = {b: v for b, v in memory.items() if v}
    norm = fit_with_norm(d, memory)
    out = ROOT / "assets" / "patchcore" / design; out.mkdir(parents=True, exist_ok=True)
    state = d.pc.state()
    torch.save({k: v for k, v in state.items()}, out / "memory.pt")
    (out / "settings.json").write_text(json.dumps(dict(
        design=design, P_REVIEW_SCORE=P_REVIEW_SCORE, variant="global", norm=norm, dim=state["memory"].shape[1],
        memory_captures=used, skipped_not_aligned=skipped, reference_profile=f"assets/branch_r_profiles/{design}",
        backbone="WideResNet-50-2 ImageNet, layer2+layer3"), indent=2), encoding="utf-8")
    print(design, "memory", len(used), "skipped", skipped, "norm", round(norm, 3),
          "MB", round((out / "memory.pt").stat().st_size / 2**20), flush=True)
    del d, memory
    torch.cuda.empty_cache()
