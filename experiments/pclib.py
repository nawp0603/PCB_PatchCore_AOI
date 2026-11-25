"""Shared helpers for the PatchCore experiments (alignment = deployed SIFT-RANSAC of the reference profile)."""
import sys, warnings
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.core.calibrated_camera import CalibratedCameraInspection
from src.core.patchcore_branch import Backbone, PositionalPatchCore
from src.core.board_align import BoardAligner

DATA = Path(r"D:\FPTU\SU26\Data_AnhThat\Data_BCD")
DEFECT = DATA / "AnhDefect"
LABELS = Path(r"D:\FPTU\KLTN\PCB_DualPath_AOI\results\bcdg_labels_final_20261122\labels")
PROFILES = ROOT / "assets" / "branch_r_profiles"
WEIGHTS = ROOT / "assets" / "weights" / "wide_resnet50_2_imagenet.pth"
MEMORY_CAPS = range(1, 9)          # c01..c08 of each memory board (L-stop and free-rotation captures)
warnings.filterwarnings("ignore", category=UserWarning)


def read(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def cap_no(p):
    return int(Path(p).stem.split("_c")[-1])


class Design:
    """Reference frame + SIFT alignment of one design, and a PatchCore bound to it."""

    def __init__(self, design: str, backbone: Backbone, board_only: bool = True):
        self.cam = CalibratedCameraInspection(PROFILES / design, profile_dir=PROFILES / design)
        self.pc = PositionalPatchCore(backbone, self.cam.template, self.cam.mask)
        self.h, self.w = self.cam.gray.shape
        self.aligner = BoardAligner(self.cam.template) if board_only else None

    def align(self, img):
        H = self.aligner.align(img) if self.aligner is not None else self.cam._align(img)
        if H is None:
            return None
        warped = cv2.warpPerspective(img, H, (self.w, self.h))
        valid = cv2.warpPerspective(np.ones(img.shape[:2], np.uint8), H, (self.w, self.h), flags=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((9, 9), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        return H, warped, valid

    def features(self, path):
        a = self.align(read(path))
        if a is None:
            return None
        H, warped, valid = a
        feat, cells = self.pc.features(warped, valid)
        return H, feat, cells


def project_box(box, H):
    """Template-frame box -> axis-aligned box in the captured frame."""
    l, t, r, b = box
    pts = np.float32([[l, t], [r, t], [r, b], [l, b]])[None]
    q = cv2.perspectiveTransform(pts, np.linalg.inv(H))[0]
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def gt_boxes(stem, w=2592, h=1944):
    p = LABELS / f"{stem}.txt"
    out = []
    if p.exists():
        for line in p.read_text().split("\n"):
            if line.strip():
                k, x, y, bw, bh = line.split()
                x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
                out.append([x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2])
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def hits(gt, box):
    """Same criterion as the Branch R evaluation: IoU >= 0.1 or either box contains the other's centre."""
    cg = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2); cb = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    inside = lambda p, r: r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]
    return iou(gt, box) >= 0.1 or inside(cg, box) or inside(cb, gt)


def fit_with_norm(design: Design, memory: dict, sample_per_board=3):
    """memory: {board: [(feat, cells), ...]}. Norm = 99.9th pct of leave-one-board-out normal scores
    (one memory board -> leave-one-capture-out). Then fits design.pc on all memory with that norm."""
    vals = []
    boards = list(memory)
    if len(boards) > 1:
        for b in boards:
            design.pc.fit([s for o in boards if o != b for s in memory[o]])
            design.pc.norm = 1.0
            step = max(1, len(memory[b]) // sample_per_board)
            for feat, cells in memory[b][::step][:sample_per_board]:
                m = design.pc.score_map(feat, cells)
                vals.append(m[m > 0])
    else:
        items = memory[boards[0]]
        for i in range(0, len(items), max(1, len(items) // sample_per_board)):
            design.pc.fit([s for j, s in enumerate(items) if j != i])
            design.pc.norm = 1.0
            m = design.pc.score_map(*items[i])
            vals.append(m[m > 0])
    norm = float(np.percentile(np.concatenate(vals), 99.9))
    design.pc.fit([s for b in boards for s in memory[b]])
    design.pc.norm = norm
    return norm


def fit_positional(design: Design, memory: dict, sample_per_board=3, q=0.95):
    """Variant 'pos': per-cell normalization. Every memory capture is scored leave-one-board-out
    (one board -> leave-one-capture-out); tol = 95th percentile of those normal cell scores per cell,
    widened by a 3x3 max filter (alignment slack) and floored at its median. Then the global norm is the
    99.9th percentile of the tol-normalized leave-out maps of a few captures per board."""
    pc = design.pc
    boards = list(memory)
    groups = [(b, [s for o in boards if o != b for s in memory[o]], memory[b]) for b in boards] if len(boards) > 1 \
        else [(i, [s for j, s in enumerate(memory[boards[0]]) if j != i], [memory[boards[0]][i]])
              for i in range(len(memory[boards[0]]))]
    pc.tol, pc.norm = None, 1.0
    stack = []
    dev = pc.backbone.device
    for _, mem, held in groups:
        pc.fit(mem)
        for feat, cells in held:
            s = pc.cell_scores(feat.to(dev))
            stack.append(torch.where(cells.to(dev), s, torch.full_like(s, float("nan"))))
    S = torch.stack(stack)
    tol = torch.nanquantile(S.flatten(1).T, q, dim=1).reshape(S.shape[1:]) if S.shape[0] > 1 else S[0]
    floor = torch.nanmedian(tol)
    tol = torch.nan_to_num(tol, nan=float(floor))
    tol = torch.nn.functional.max_pool2d(tol[None, None], 3, 1, 1)[0, 0].clamp_min(floor)
    pc.tol = tol
    vals = []
    for _, mem, held in groups:
        pc.fit(mem)
        step = max(1, len(held) // sample_per_board)
        for feat, cells in held[::step][:sample_per_board]:
            m = pc.score_map(feat, cells)
            vals.append(m[m > 0])
    pc.norm = float(np.percentile(np.concatenate(vals), 99.9))
    pc.fit([s for b in boards for s in memory[b]])
    return pc.norm


def score_image(design: Design, H, feat, cells, gt=(), thr=None):
    """Regions (seed 0.5, top 8, frame coords) + strict per-label metrics:
    gt_peak[i] = max score inside label i (label box projected into the reference frame);
    tight = components of score >= thr (decision threshold), with how many lie outside every label."""
    m = design.pc.score_map(feat, cells)
    regs = design.pc.regions(m)
    out = dict(max_score=float(m.max()), regions=[dict(box=project_box(r.box, H), score=r.score) for r in regs])
    gt_tpl = []
    for g in gt:
        pts = cv2.perspectiveTransform(np.float32([[g[0], g[1]], [g[2], g[1]], [g[2], g[3]], [g[0], g[3]]])[None], H)[0]
        x0, y0 = np.floor(pts.min(0)).astype(int); x1, y1 = np.ceil(pts.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, m.shape[1]), min(y1, m.shape[0])
        gt_tpl.append((x0, y0, x1, y1))
    out["gt_peak"] = [float(m[y0:y1, x0:x1].max()) if x1 > x0 and y1 > y0 else 0.0 for x0, y0, x1, y1 in gt_tpl]
    inside = lambda r: any(not (r.box[2] < x0 or r.box[0] > x1 or r.box[3] < y0 or r.box[1] > y1)
                           for x0, y0, x1, y1 in gt_tpl)
    # components of score >= 1.0 (edge of normal): (peak, lies outside every label, area px)
    out["comps"] = [(round(r.score, 3), not inside(r), int(r.area))
                    for r in design.pc.regions(m, top_k=10000, seed=1.0)]
    return out


torch.backends.cudnn.benchmark = True
