"""Red = warped capture, green = reference, after SIFT-RANSAC alignment -> patchcore_view/align_*.jpg"""
import sys
from pclib import *
OUTV = Path(r"D:\FPTU\KLTN\PCB_DualPath_AOI\results\patchcore_view")
d = Design(sys.argv[1], Backbone(WEIGHTS))
for name in sys.argv[2:]:
    p = (DATA if "_v0_" in name else DEFECT) / f"{name}.png"
    H, warped, valid = d.align(read(p))
    w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY); t = d.cam.gray
    img = np.dstack([np.zeros_like(t), t, w])
    print(name, "H=", np.round(H, 3).tolist())
    cv2.imencode(".jpg", cv2.resize(img, None, fx=.5, fy=.5))[1].tofile(str(OUTV / f"align_{name}.jpg"))
