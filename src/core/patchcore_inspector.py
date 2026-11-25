"""PatchCore branch of the app for designs B, C, D, G (patchcore_branch.py + board_align.py).

inspect(frame) -> {"action": "INSPECT" | "RECAPTURE", "regions": [...], "max_score": float | None}
regions are in captured-frame coordinates with keys center, box, score, area_template (same shape as the
Branch B regions, so the deployed combine() can use them). Only regions with peak >= SHOW_FROM are kept.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.core.board_align import BoardAligner
from src.core.patchcore_branch import Backbone, PositionalPatchCore

SHOW_FROM = 1.0     # normalized score 1.0 = upper edge of what GOOD captures produce


def _read(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


def patchcore_designs(root: Path) -> dict:
    """Designs with both a reference profile and a PatchCore memory: {design: (profile_dir, memory_dir)}."""
    out = {}
    memory_root = root / "assets" / "patchcore"
    if memory_root.is_dir():
        for m in sorted(memory_root.iterdir()):
            profile = root / "assets" / "branch_r_profiles" / m.name
            if (m / "memory.pt").is_file() and (m / "settings.json").is_file() and (profile / "template_A.png").is_file():
                out[m.name] = (profile, m)
    return out


class PatchCoreInspector:
    def __init__(self, backbone: Backbone, profile_dir: Path, memory_dir: Path):
        template = _read(Path(profile_dir) / "template_A.png")
        mask = _read(Path(profile_dir) / "template_mask.png", cv2.IMREAD_GRAYSCALE) > 0
        self.h, self.w = mask.shape
        self.aligner = BoardAligner(template)
        self.pc = PositionalPatchCore(backbone, template, mask)
        state = torch.load(Path(memory_dir) / "memory.pt", map_location="cpu", weights_only=True)
        self.pc.load_state(state)
        settings = json.loads((Path(memory_dir) / "settings.json").read_text(encoding="utf-8"))
        self.review_score = float(settings["P_REVIEW_SCORE"])

    def release(self) -> None:
        """Free the GPU memory of this design (another design was selected)."""
        self.pc.memory = self.pc.mem_sq = self.pc.tol = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def inspect(self, frame: np.ndarray) -> dict:
        H = self.aligner.align(frame)
        if H is None:
            return dict(action="RECAPTURE", regions=[], max_score=None)
        warped = cv2.warpPerspective(frame, H, (self.w, self.h))
        valid = cv2.warpPerspective(np.ones(frame.shape[:2], np.uint8), H, (self.w, self.h), flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        feat, cells = self.pc.features(warped, valid)
        score = self.pc.score_map(feat, cells)
        inverse = np.linalg.inv(H)
        regions = []
        # components of score >= SHOW_FROM (tighter than the 0.5 candidate seed); the decision uses the peak
        for r in self.pc.regions(score, seed=SHOW_FROM):
            l, t, rr, b = r.box
            pts = np.float32([r.center, (l, t), (rr, t), (rr, b), (l, b)])[None]
            center, *corners = cv2.perspectiveTransform(pts, inverse)[0]
            lo, hi = np.min(corners, axis=0), np.max(corners, axis=0)
            regions.append(dict(center=center.tolist(), box=np.concatenate([lo, hi]).tolist(),
                                score=float(r.score), area_template=float(r.area)))
        return dict(action="INSPECT", regions=regions, max_score=float(score.max()))
