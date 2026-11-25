"""Alignment quality after SIFT-RANSAC: NCC of gradient magnitude (warped vs reference) inside the board mask,
and the best extra integer shift (+-40 px) that would improve it."""
import sys
from pclib import *

def grad(g):
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

def ncc(a, b, m):
    a, b = a[m], b[m]; a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

design = sys.argv[1]
d = Design(design, None.__class__ and Backbone(WEIGHTS))
ref = grad(d.cam.gray); m = cv2.erode(d.cam.mask.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
for name in sys.argv[2:]:
    p = (DATA if "_v0_" in name else DEFECT) / f"{name}.png"
    H, warped, valid = d.align(read(p))
    g = grad(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    base = ncc(g, ref, m & valid)
    best = max(((ncc(np.roll(g, (dy, dx), (0, 1)), ref, m & valid), dy, dx)
                for dy in range(-40, 41, 4) for dx in range(-40, 41, 4)))
    print(f"{name}: ncc {base:.3f} | best shift {best[1]:+d},{best[2]:+d} -> {best[0]:.3f}")
