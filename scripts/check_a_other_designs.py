"""Run Branch A alone on boards of other designs and save overlays for a visual check.

    python scripts/check_a_other_designs.py --images <folder of .png/.jpg> [--out <folder>]

Default model real_3class.pt was trained on designs A, B, C (not D, G). Every box drawn on a spot that is NOT a real defect
is a false alarm the GOOD board of a new design could also get. A false box at or above the
FAIL/REWORK threshold sends a good board to the FAIL bin automatically.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.core.real_ab_pipeline import RealABInspection  # noqa: E402

WEIGHTS = ROOT / "assets" / "weights" / "pcb_camera" / "real_3class.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "a_other_designs")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()

    paths = sorted(p for p in args.images.rglob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if not paths:
        parser.error(f"No images in {args.images}")
    args.out.mkdir(parents=True, exist_ok=True)
    inspection = RealABInspection(args.weights, ROOT / "results")
    inspection.warmup()
    counts = Counter()
    with open(args.out / "a_only.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "conclusion", "boxes", "max_conf", "classes"])
        for path in paths:
            frame = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"skip (cannot read): {path}")
                continue
            res = inspection.inspect(frame, use_b=False)
            items = res["a_items"]
            counts[res["conclusion"]] += 1
            writer.writerow([path.name, res["conclusion"], len(items),
                             f"{max((i['conf'] for i in items), default=0):.2f}",
                             " ".join(f"{i['class_name']}:{i['conf']:.2f}" for i in items)])
            cv2.imwrite(str(args.out / f"{path.stem}_A.jpg"), res["annotated_bgr"],
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"{path.name}: {res['conclusion']} ({len(items)} box)")
    print("\nTong:", dict(counts))
    print(f"Mo anh *_A.jpg trong {args.out}: box nao khong nam tren loi that la bao gia.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
