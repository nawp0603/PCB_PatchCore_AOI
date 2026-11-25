import json, sys, collections
import numpy as np
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
pk = lambda r: max([g["score"] for g in r.get("regions", [])], default=0)
for kind in ("GOOD", "DEFECT"):
    for d in sorted({r["design"] for r in rows}):
        by = collections.defaultdict(list)
        for r in rows:
            if r["kind"] == kind and r["design"] == d and r.get("aligned", True):
                by[int(r["image"].split("_c")[-1])].append(pk(r))
        print(kind, d, {c: round(float(np.mean(v)), 2) for c, v in sorted(by.items())})
for r in [r for r in rows if r["kind"] == "GOOD" and pk(r) >= 1.05][:8]:
    print(r["image"], [(list(map(int, x["box"])), round(x["score"], 2)) for x in r["regions"][:2]])
