"""Position-aware PatchCore for bare-PCB designs with a registered reference image (B, C, D, G).

Plain PatchCore compares every patch with the most similar normal patch anywhere on the board. On a PCB a
tin-filled hole looks like a tinned pad elsewhere and a cut trace looks like a normal trace end, so here a
patch is only compared with normal patches at the same board position (+- RADIUS feature cells), after
the image is aligned to the design reference with the existing SIFT-RANSAC step.

  features : WideResNet-50 (ImageNet) layer2 + layer3, 3x3 local average (PatchCore), stride 8 px,
             channels averaged down to DIM
  memory   : features of aligned GOOD captures of one or more boards of the design
  score    : per cell, min L2 distance to memory cells within +-RADIUS; bilinear upsampling,
             Gaussian sigma 4 px; divided by `norm` (99.9th percentile of leave-one-out normal scores),
             so 1.0 = edge of what normal captures produce
  regions  : connected components of score >= 1 inside the board mask, ranked by peak score
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

DIM = 256
RADIUS = 1          # +-1 cell = +-8 px after SIFT alignment (median reprojection error <= 2.5 px)
STRIDE = 8
MIN_AREA = 60
TOP_K = 8
SEED = 0.5          # candidate regions: normalized score >= 0.5
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Backbone(torch.nn.Module):
    def __init__(self, weights: Path | None = None, device: str | None = None):
        super().__init__()
        net = torchvision.models.wide_resnet50_2()
        if weights is not None:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=False)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(self.device).eval()
        if self.device.type == "cuda":
            self.half()

    @torch.no_grad()
    def forward(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 HxW -> features [DIM, ceil(H/8), ceil(W/8)] (float16 on GPU)."""
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).permute(2, 0, 1)[None].float() / 255
        x = ((rgb - MEAN) / STD).to(self.device, next(self.parameters()).dtype)
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([f2, f3], 1)[0]                                  # [1536, h, w]
        c, h, w = feat.shape
        feat = F.adaptive_avg_pool1d(feat.reshape(c, -1).T[None].float(), DIM)[0].T.reshape(DIM, h, w)
        return feat.to(f2.dtype)


def _min_dist(query: torch.Tensor, memory: torch.Tensor, mem_sq: torch.Tensor, radius: int) -> torch.Tensor:
    """query [D,h,w], memory [N,D,h,w], mem_sq [N,h,w] (inf where invalid) -> [h,w] min L2 distance.
    One memory capture and one shift at a time keeps the temporary tensors at [D,h,w]."""
    d, h, w = query.shape
    q_sq = query.float().pow(2).sum(0)
    best = torch.full((h, w), float("inf"), device=query.device)
    for k in range(memory.shape[0]):
        mem = F.pad(memory[k], (radius,) * 4)
        sq = F.pad(mem_sq[k], (radius,) * 4, value=float("inf"))
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                dot = (query * mem[:, dy:dy + h, dx:dx + w]).sum(0, dtype=torch.float32)
                best = torch.minimum(best, q_sq + sq[dy:dy + h, dx:dx + w] - 2 * dot)
    return best.clamp_min(0).sqrt()


@dataclass(frozen=True)
class PRegion:
    center: tuple[float, float]
    box: tuple[int, int, int, int]
    score: float
    area: int


class PositionalPatchCore:
    """One design. Works in the reference (template) frame, cropped to the board bounding box."""

    def __init__(self, backbone: Backbone, template_bgr: np.ndarray, board_mask: np.ndarray):
        self.backbone = backbone
        ys, xs = np.nonzero(board_mask)
        pad = 16
        self.y0, self.x0 = max(int(ys.min()) - pad, 0), max(int(xs.min()) - pad, 0)
        self.y1 = min(int(ys.max()) + pad, board_mask.shape[0])
        self.x1 = min(int(xs.max()) + pad, board_mask.shape[1])
        self.shape = board_mask.shape
        self.mask = board_mask[self.y0:self.y1, self.x0:self.x1]
        self.template = template_bgr
        self.memory = None
        self.mem_sq = None
        self.tol = None          # optional per-cell normal score level [h,w] (positional normalization)
        self.norm = 1.0

    def _crop(self, warped: np.ndarray) -> np.ndarray:
        return warped[self.y0:self.y1, self.x0:self.x1]

    def _cells(self, valid: np.ndarray, h: int, w: int) -> torch.Tensor:
        """Valid-pixel mask (full template frame) -> bool [h,w] of feature cells fully inside it."""
        crop = (self._crop(valid.astype(np.uint8)) * self.mask).astype(np.float32)
        cell = cv2.resize(cv2.erode(crop, np.ones((STRIDE, STRIDE), np.uint8)), (w, h), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(cell > 0.99).to(self.backbone.device)

    def features(self, warped: np.ndarray, valid: np.ndarray):
        feat = self.backbone(self._crop(warped))
        return feat, self._cells(valid, *feat.shape[-2:])

    def fit(self, samples: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """samples: (features, valid cells) of aligned GOOD captures (CPU or GPU tensors)."""
        dev = self.backbone.device
        self.memory = self.mem_sq = None
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        self.memory = torch.stack([f.to(dev) for f, _ in samples])
        sq = self.memory.float().pow(2).sum(1)
        cells = torch.stack([v.to(dev) for _, v in samples])
        self.mem_sq = torch.where(cells, sq, torch.full_like(sq, float("inf")))

    def cell_scores(self, feat: torch.Tensor) -> torch.Tensor:
        return _min_dist(feat, self.memory, self.mem_sq, RADIUS)

    def score_map(self, feat: torch.Tensor, cells: torch.Tensor) -> np.ndarray:
        """-> float32 map in the full template frame, normalized by self.norm, 0 outside the board."""
        dev = self.backbone.device
        feat, cells = feat.to(dev), cells.to(dev)
        raw = self.cell_scores(feat)
        if self.tol is not None:
            raw = raw / self.tol
        cell = torch.where(cells, raw, torch.zeros_like(cells, dtype=torch.float32))
        h, w = self.mask.shape
        up = F.interpolate(cell[None, None], scale_factor=STRIDE, mode="bilinear", align_corners=False)[0, 0]
        up = up[:h, :w].cpu().numpy()
        up = cv2.GaussianBlur(up, (0, 0), 4) * self.mask
        full = np.zeros(self.shape, np.float32)
        full[self.y0:self.y1, self.x0:self.x1] = up / self.norm
        return full

    @staticmethod
    def regions(score: np.ndarray, top_k: int = TOP_K, seed: float = SEED) -> list[PRegion]:
        """Candidate regions (score >= seed), ranked by peak. The decision threshold is applied to the peak."""
        n, lab, stats, cents = cv2.connectedComponentsWithStats((score >= seed).astype(np.uint8), 8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < MIN_AREA:
                continue
            peak = float(score[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].max())
            out.append(PRegion(tuple(map(float, cents[i])), (x, y, x + w, y + h), peak, area))
        return sorted(out, key=lambda r: -r.score)[:top_k]

    def state(self) -> dict:
        return dict(memory=self.memory.cpu(), mem_sq=self.mem_sq.cpu(), norm=self.norm,
                    tol=None if self.tol is None else self.tol.cpu(), crop=(self.y0, self.y1, self.x0, self.x1))

    def load_state(self, state: dict) -> None:
        dev = self.backbone.device
        self.memory = state["memory"].to(dev)
        self.mem_sq = state["mem_sq"].to(dev)
        self.tol = None if state.get("tol") is None else state["tol"].to(dev)
        self.norm = float(state["norm"])
