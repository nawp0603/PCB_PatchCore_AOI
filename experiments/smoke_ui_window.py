"""Open the real Tk window of PCB_PatchCore_AOI, inspect a few captures with PatchCore designs, screenshot.
Screenshots go to PCB_DualPath_AOI/results/patchcore_view (viewable folder)."""
import sys
from pathlib import Path
import tkinter as tk
import cv2, numpy as np
from PIL import ImageGrab

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.ui.tkinter_app import TkAOIApp

out = Path(r"D:\FPTU\KLTN\PCB_DualPath_AOI\results\patchcore_view"); out.mkdir(exist_ok=True)
DATA = Path(r"D:\FPTU\SU26\Data_AnhThat\Data_BCD")
root = tk.Tk(); root.geometry("1600x900+0+0")
app = TkAOIApp(root, weights_path=ROOT / "assets/weights/pcb_camera/real_3class.pt")
cases = [("D", "AnhDefect/D_01_v2_c01.png"), ("D", "D_01_v0_c12.png"), ("G", "G_01_v0_c09.png"),
         ("B", "AnhDefect/B_00_v1_c01.png"), ("C", "AnhDefect/C_01_v1_c01.png")]
state = {"i": 0, "ticks": 0}


def shot(tag):
    x, y = root.winfo_rootx(), root.winfo_rooty()
    ImageGrab.grab((x, y, x + root.winfo_width(), y + root.winfo_height())).save(out / f"app_{tag}.png")
    print(tag, "|", app.lbl_verdict.cget("text"), "|", app.lbl_verdict_sub.cget("text"), "|",
          app.lbl_defects.cget("text"), "|", app.lbl_metrics.cget("text"), flush=True)


def step():
    state["ticks"] += 1
    if state["ticks"] > 3000:
        print("timeout", flush=True); root.destroy(); return
    if app.real_ab is None or app._busy:
        root.after(100, step); return
    if state.get("waiting"):
        shot(state.pop("waiting")); state["i"] += 1
    if state["i"] >= len(cases):
        root.destroy(); return
    if "radios" not in state:
        state["radios"] = [w.cget("text") for w in app._controls if isinstance(w, tk.Radiobutton)]
        print("radios:", state["radios"], flush=True)
    design, name = cases[state["i"]]
    app.board_mode.set(design); app._switch_board()
    app._run_inspect(cv2.imdecode(np.fromfile(str(DATA / name), np.uint8), cv2.IMREAD_COLOR), source=Path(name).name)
    state["waiting"] = f"{design}_{Path(name).stem}"
    root.after(1500, step)


root.after(500, step)
root.mainloop()
