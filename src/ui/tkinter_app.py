"""Standalone in-process Tkinter Desktop Demo App.

Uses `tkinter`/`ttk` with a Dark Industrial theme and calls
`DualPathAOIEngine` (src/core/aoi_engine.py) and `GigECameraManager`
(src/service/gige_camera.py) DIRECTLY in-process - no FastAPI/Uvicorn,
no port 8765, no external server dependency.

Run:
    python src/ui/tkinter_app.py            # normal demo
    python src/ui/tkinter_app.py --self-test # headless CI self-test
"""
from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from configs.resource_paths import (  # noqa: E402
    PROJECT_ROOT as _RP,
    LOCAL_TEMPLATES, default_weights,
)
from src.core.aoi_engine import DualPathAOIEngine  # noqa: E402
from src.core.calibrated_camera import CalibratedCameraInspection  # noqa: E402
from src.core.real_ab_pipeline import RealABInspection  # noqa: E402
from src.core.patchcore_inspector import patchcore_designs  # noqa: E402
from src.decision.sorting import sort_decision  # noqa: E402
from src.service.robot_bridge import RobotBridge  # noqa: E402
from src.service.gige_camera import GigECameraManager  # noqa: E402

# ---------------------------------------------------------------------------
# Theme constants (Dark Industrial SCADA).
# ---------------------------------------------------------------------------
BG = "#0B1120"
CARD = "#111827"
NEUTRAL = "#1F2937"
INK = "#E5E7EB"
MUTED = "#9CA3AF"
ACCENT = "#38BDF8"
SUCCESS = "#047857"
ERROR = "#DC2626"
BOX_A = "#22C55E"
BOX_B = "#F59E0B"
BOX_R = "#A855F7"   # PatchCore regions (designs B/C/D/G)
TITLE_FONT = ("Segoe UI Semibold", 11)
LABEL_FONT = ("Segoe UI", 10)
MONO_FONT = ("Consolas", 10)
INSPECT_TEXT = "KIỂM TRA  (Space)"
VERDICTS = {
    "FAIL": ("✕ FAIL", ERROR),
    "REWORK": ("⚠ REWORK", "#C2410C"),
    "REVIEW": ("! REVIEW", "#B45309"),
    "CONFIRM": ("↻ CHỤP XÁC NHẬN", "#1D4ED8"),
    "RECAPTURE": ("⟲ CHỤP LẠI", "#6D28D9"),
    "NO REGIONS": ("✓ KHÔNG THẤY LỖI (chưa hiệu chuẩn)", SUCCESS),
    "UNAVAILABLE": ("— KHÔNG KIỂM TRA ĐƯỢC", "#4B5563"),
    "IDLE": ("CHƯA KIỂM TRA", NEUTRAL),
}
DEMO_ROOT = _RP / "data" / "demo_and_test_samples"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")
    return img


CLASS_LABELS = {
    "missing_hole": "Mạch tịt (lấp lỗ)", "open_circuit": "Mạch hở",
    "short": "Chập mạch", "spur": "Thừa thiếc/đồng",
    "mouse_bite": "Khuyết mép", "spurious_copper": "Đồng thừa",
}


def _branch_letter(item):
    return {"branchb": "B", "patchcore": "P"}.get(item.get("source"), "A")


def _item_label(item):
    if item.get("source") == "patchcore":
        return "Vùng lạ yếu (PatchCore)" if item.get("p_weak") else "Vùng lạ (PatchCore)"
    if item.get("source") == "branchb":
        return "Vùng lạ yếu (nhánh B)" if item.get("b_weak") else "Vùng lạ (nhánh B)"
    name = item.get("class_name") or "?"
    local = CLASS_LABELS.get(name.lower().replace(" ", "_"))
    return f"{local} ({name})" if local else name


REASON_TEXT = {
    "CONFIRM": "Giữ nguyên bo mạch và nhấn Space để chụp lần hai.",
    "RECAPTURE": "Kiểm tra vị trí bo mạch và chụp lại.",
    "NO REGIONS": "Không phát hiện vùng nghi vấn.",
}


def _reason_text(res, verdict):
    """One Vietnamese line for the verdict banner; core reasons stay unchanged."""
    if verdict == "NO REGIONS" and res.get("patchcore_design"):
        return "Nhánh A không thấy lỗi; PatchCore không có vùng vượt ngưỡng ngoài hộp A."
    if verdict == "NO REGIONS" and res.get("branch_b") is False:
        return "Nhánh A không thấy lỗi (bo loại khác, không dùng nhánh B)."
    if verdict in REASON_TEXT:
        return REASON_TEXT[verdict]
    details = [d for d in res.get("details") or [] if d.get("conf") is not None]
    if res.get("conclusion") in ("FAIL", "REWORK") and details:
        top = max(details, key=lambda d: d["conf"])
        lead = "Lỗi nghiêm trọng" if res["conclusion"] == "FAIL" else "Lỗi cần sửa"
        more = f" · +{len(details) - 1} lỗi khác" if len(details) > 1 else ""
        return f"{lead}: {_item_label(top)} · tin cậy {top['conf']:.2f}{more}"
    reason = next(iter(res.get("reasons") or []), "")
    return reason[:1].upper() + reason[1:]


BIN_TEXT = {"FAIL": "LỖI", "PASS": "ĐẠT"}
# The robot must start a command within RobotBridge.ttl_s (30 s); this leaves ~90 s to move.
ROBOT_TIMEOUT_S = 120.0


def _use_branch_b(app) -> bool:
    """Board of design A uses A + B; another design has no Branch B template."""
    mode = getattr(app, "board_mode", None)
    return mode is None or mode.get() == "A"


def _patchcore_designs() -> dict:
    """Designs with a PatchCore memory (B, C, D, G): {design: (profile dir, memory dir)}."""
    return patchcore_designs(PROJECT_ROOT)


def _patchcore_design(app):
    """Selected design that uses YOLO + PatchCore, or None."""
    mode = getattr(app, "board_mode", None)
    value = mode.get() if mode is not None else None
    return value if value not in (None, "A", "OTHER") else None


def _board_text(value) -> str:
    if value == "A":
        return "Thiết kế A (nhánh A + B)"
    if value == "OTHER":
        return "Bo loại khác (chỉ nhánh A)"
    return f"Thiết kế {value} (nhánh A + PatchCore)"


def _preview_photo(frame, width, height, items):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    size = max(1, round(w * scale)), max(1, round(h * scale))
    preview = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    image = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    for index, item in enumerate(items, 1):
        box = item.get("bbox")
        if box is None:
            continue
        x, y, right, bottom = (round(v * scale) for v in box)
        branch = _branch_letter(item)
        color = {"B": BOX_B, "P": BOX_R}.get(branch, BOX_A)
        draw.rectangle((x, y, right, bottom), outline=color, width=3)
        # The index links even tightly grouped boxes to their full table labels.
        text = f"{branch} {index}"
        tw = round(draw.textlength(text)) + 8
        left = max(0, min(x, size[0] - tw))
        top = max(0, min(y - 18, size[1] - 18))
        draw.rectangle((left, top, left + tw, top + 18), fill=(0, 0, 0, 190))
        draw.text((left + 4, top + 2), text, fill="white")
    return ImageTk.PhotoImage(image)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class TkAOIApp:
    def __init__(self, root: tk.Tk, self_test: bool = False,
                 weights_path: Path | None = None, robot_dir: Path | None = None):
        self.root = root
        self.self_test = self_test
        self.weights_path = weights_path
        self.camera_quality_mode = "--camera-quality" in sys.argv and not self_test
        self.calibrated_camera = None
        self.real_ab: RealABInspection | None = None
        self._real_ab_context = 0
        self._real_ab_source = None
        self._pending_camera_regions = None
        self._last_inspection_frame: np.ndarray | None = None
        # Robot sorting: one command in flight, or one REVIEW waiting for the operator.
        self.robot = RobotBridge(robot_dir) if robot_dir is not None else None
        self._robot_job: dict[str, Any] | None = None
        self._robot_review: tuple[str, str] | None = None
        self.root.title("PCB AOI · Trạm kiểm tra")
        self.root.geometry("1540x880")
        self.root.configure(bg=BG)
        self.root.minsize(1366, 768)

        self.engine: DualPathAOIEngine | None = None
        self.engine_ready = False
        self.engine_error: str | None = None
        self.engine_trace: str | None = None
        self.engine_info: dict[str, Any] = {}
        self.cam_mgr: GigECameraManager | None = None
        self.cam_connected = False
        self._live_timer_id: str | None = None
        self._grabbing_cam = False

        self._busy = False
        self._stats = {"total": 0, "fail": 0, "review": 0, "pass": 0}
        self._current_frame_bgr: np.ndarray | None = None
        self._file_frame: np.ndarray | None = None
        self._file_name = ""

        self._build_ui()
        self._wire_shortcuts()

        # Bootstrap engine + warmup on a background thread (window shows < 0.5s).
        init = self._init_calibrated_camera if self.camera_quality_mode else self._init_engine
        if self.weights_path is not None:
            init = self._init_real_ab
        threading.Thread(target=init, daemon=True).start()
        if not self.self_test:
            # Auto-scan LAN/USB3 cameras on startup without blocking UI.
            self.root.after(400, self._cam_scan_silent)

    # ------------------------------------------------------------------ #
    # UI build
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self._controls = []
        self._control_states = {}
        self.root.option_add("*Font", "{Segoe UI} 10")
        for widget in ("Button", "Radiobutton", "Checkbutton", "Entry"):
            self.root.option_add(f"*{widget}.highlightThickness", 2)
            self.root.option_add(f"*{widget}.highlightColor", ACCENT)
            self.root.option_add(f"*{widget}.highlightBackground", CARD)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=CARD, foreground=INK, bordercolor=NEUTRAL,
                        darkcolor=CARD, lightcolor=CARD, focuscolor=ACCENT)
        style.configure("TCombobox", fieldbackground=NEUTRAL, arrowcolor=INK,
                        padding=4, borderwidth=2)
        style.map("TCombobox", fieldbackground=[("readonly", NEUTRAL)],
                  foreground=[("readonly", INK)], bordercolor=[("focus", ACCENT)])
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=0)
        style.configure("TNotebook.Tab", background=CARD, foreground=MUTED,
                        padding=(16, 8), font=TITLE_FONT, focuscolor=ACCENT)
        style.map("TNotebook.Tab", background=[("selected", NEUTRAL)],
                  foreground=[("selected", INK)])
        style.configure("Treeview", fieldbackground=CARD, background=CARD,
                        foreground=INK, font=MONO_FONT, rowheight=32, borderwidth=0)
        style.configure("Treeview.Heading", background=NEUTRAL, foreground=INK,
                        padding=(0, 8), font=LABEL_FONT, borderwidth=0)
        style.map("Treeview", background=[("selected", ACCENT)],
                  foreground=[("selected", BG)])

        header = tk.Frame(self.root, bg=CARD, height=48)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="PCB AOI · Trạm kiểm tra", bg=CARD, fg=INK,
                 font=("Segoe UI Semibold", 16)).pack(side="left", padx=16)
        self.lbl_ai = self._badge(header, "● Khởi động")
        model = (self.weights_path.name if self.weights_path else
                 "Camera hiệu chuẩn" if self.camera_quality_mode else default_weights().name)
        self.lbl_model = self._badge(header, f"● Model: {model}")
        self.lbl_cam_badge = self._badge(header, "● Camera: Chưa kết nối")
        if self.robot is not None:
            self.lbl_robot = self._badge(header, "● Robot: Sẵn sàng")

        footer = tk.Frame(self.root, height=24, bg=CARD)
        footer.pack(side="bottom", fill="x")
        footer.pack_propagate(False)
        self.lbl_metrics = tk.Label(footer, text="", bg=CARD, fg=MUTED,
                                    font=("Segoe UI", 9))
        self.lbl_metrics.pack(side="right", padx=16)
        self.status = tk.Label(footer, text="Đang khởi động…", anchor="w",
                               bg=CARD, fg=MUTED, font=("Segoe UI", 9))
        self.status.pack(side="left", fill="x", expand=True, padx=16)
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=16)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        self._build_left(body)
        self._build_center(body)
        self._build_right(body)

    def _badge(self, parent, text):
        label = tk.Label(parent, text=text, bg=NEUTRAL, fg=INK, padx=8, pady=4)
        label.pack(side="right", padx=(0, 8))
        return label

    def _panel(self, parent):
        return tk.Frame(parent, bg=CARD, highlightbackground=NEUTRAL,
                        highlightthickness=1, padx=16, pady=16)

    def _heading(self, parent, text):
        label = tk.Label(parent, text=text, anchor="w", font=TITLE_FONT,
                         bg=CARD, fg=INK, bd=0)
        label.pack(fill="x", pady=(0, 16))
        return label

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=BG, width=300)
        left.grid(row=0, column=0, sticky="ns")
        left.pack_propagate(False)
        source = self._panel(left)
        source.pack(fill="both", expand=True)
        self._heading(source, "NGUỒN ẢNH")
        self.cmb_mode = tk.StringVar(value="Camera GigE")
        for mode in ("Camera GigE", "Ảnh từ file"):
            radio = tk.Radiobutton(source, text=mode, variable=self.cmb_mode,
                                   value=mode, command=self._switch_mode, bg=CARD,
                                   fg=INK, selectcolor=BG, activebackground=NEUTRAL,
                                   activeforeground=INK, anchor="w", takefocus=True)
            if self.camera_quality_mode and mode == "Ảnh từ file":
                radio.configure(state="disabled")
            radio.pack(fill="x", ipady=4)
            self._controls.append(radio)
        self.p_gige = tk.Frame(source, bg=CARD)
        self.p_file = tk.Frame(source, bg=CARD)
        self._build_camera_controls(self.p_gige)
        self._btn(self.p_file, "Mở ảnh…", self._open_file).pack(fill="x", pady=(8, 0))
        self.p_gige.pack(fill="x", pady=(8, 0))
        if self.weights_path is not None:
            board = tk.Frame(source, bg=CARD)
            board.pack(side="bottom", fill="x")
            self._heading(board, "LOẠI BO")
            self.board_mode = tk.StringVar(value="A")
            for value in ("A", *_patchcore_designs(), "OTHER"):
                radio = tk.Radiobutton(board, text=_board_text(value), variable=self.board_mode,
                                       value=value, command=self._switch_board, bg=CARD,
                                       fg=INK, selectcolor=BG, activebackground=NEUTRAL,
                                       activeforeground=INK, anchor="w", takefocus=True)
                radio.pack(fill="x", ipady=4)
                self._controls.append(radio)

        actions = tk.Frame(left, bg=BG)
        actions.pack(fill="x", pady=(8, 0))
        self.btn_inspect = self._btn(actions, INSPECT_TEXT, self._inspect_dispatch, accent=True)
        self.btn_inspect.pack(fill="x", ipady=8)
        if self.camera_quality_mode:
            self.btn_save_inspection = self._btn(actions, "Lưu ảnh kết quả", self._save_inspection_png)
            self.btn_save_inspection.configure(state="disabled")
            self.btn_save_inspection.pack(fill="x", pady=(8, 0))

    def _build_camera_controls(self, parent):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=(8, 0))
        self.cmb_cam = ttk.Combobox(row, state="readonly", width=10, takefocus=True,
                                    values=["Chọn camera"])
        self.cmb_cam.current(0)
        self.cmb_cam.pack(side="left", expand=True, fill="both", padx=(0, 8))
        self._controls.append(self.cmb_cam)
        self._btn(row, "Quét", self._cam_scan).pack(side="right")
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=8)
        row.columnconfigure((0, 1), weight=1, uniform="connect")
        self._btn(row, "Kết nối", self._cam_connect).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._btn(row, "Ngắt", self._cam_disconnect).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.settings_button = self._btn(parent, "▸ Cài đặt camera", self._toggle_settings)
        self.settings_button.pack(fill="x")
        self.settings = tk.Frame(parent, bg=CARD)
        self._build_camera_settings(self.settings)
        self.lbl_cam_status = tk.Label(parent, text="", bg=CARD, fg=MUTED, anchor="w",
                                       justify="left", wraplength=256)
        self.lbl_cam_status.pack(fill="x", pady=(8, 0))

    def _build_camera_settings(self, parent):
        for i, (caption, field, value) in enumerate((
                ("Exp (µs)", "ent_exp", "15000"), ("Gain (dB)", "ent_gain", "2.0"))):
            parent.columnconfigure(i, weight=1, uniform="param")
            tk.Label(parent, text=caption, bg=CARD, fg=MUTED, anchor="w").grid(
                row=0, column=i, sticky="ew")
            entry = tk.Entry(parent, width=8, bg=BG, fg=INK, relief="flat",
                             insertbackground=INK, font=MONO_FONT)
            entry.insert(0, value)
            entry.grid(row=1, column=i, sticky="ew", ipady=4, padx=(0, 8))
            setattr(self, field, entry)
            self._controls.append(entry)
        self._btn(parent, "Áp dụng", self._cam_apply_params).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=8)
        self.chk_sw_trig = tk.IntVar(value=1)
        self.chk_live = tk.IntVar(value=1)
        for i, (caption, variable, command) in enumerate((
                ("SW Trigger", self.chk_sw_trig, None),
                ("Live view", self.chk_live, self._on_live_toggle))):
            check = tk.Checkbutton(parent, text=caption, variable=variable, command=command,
                                   bg=CARD, fg=INK, selectcolor=BG, activebackground=NEUTRAL,
                                   activeforeground=INK, anchor="w", takefocus=True)
            check.grid(row=3, column=i, sticky="ew")
            self._controls.append(check)

    def _toggle_settings(self):
        if self.settings.winfo_manager():
            self.settings.pack_forget()
            self.settings_button.configure(text="▸ Cài đặt camera")
        else:
            self.settings.pack(fill="x", pady=(8, 0), before=self.lbl_cam_status)
            self.settings_button.configure(text="▾ Cài đặt camera")

    def _build_center(self, parent):
        center = tk.Frame(parent, bg=CARD, highlightbackground=NEUTRAL, highlightthickness=1)
        center.grid(row=0, column=1, sticky="nsew", padx=8)
        legend = tk.Frame(center, bg=CARD)
        legend.pack(side="bottom", fill="x", padx=16, pady=8)
        for color, desc in ((BOX_A, "A · Lỗi đã phân lớp"), (BOX_B, "B · Vùng lạ"),
                            (BOX_R, "P · Vùng lạ PatchCore")):
            tk.Label(legend, text="■", bg=CARD, fg=color).pack(side="left")
            tk.Label(legend, text=desc, bg=CARD, fg=INK).pack(side="left", padx=(4, 16))
        self.nb = ttk.Notebook(center, takefocus=True)
        self.nb.pack(fill="both", expand=True)
        self.img_annotated = self._viewport("Kết quả", "Chờ kết quả kiểm tra")
        self.img_raw = self._viewport("Ảnh gốc / Live", "Chờ ảnh đầu vào")
        self.nb.enable_traversal()
        self._focus_border(self.nb, center)

    def _viewport(self, title, placeholder):
        label = tk.Label(self.nb, text=placeholder, bg=BG, fg=MUTED, width=1, height=1)
        self.nb.add(label, text=title)
        def resize(_event):
            frame = getattr(label, "_frame_bgr", None)
            if frame is not None:
                self._set_image(label, frame)
        label.bind("<Configure>", resize)
        label.bind("<Map>", resize)
        return label

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG, width=380)
        right.grid(row=0, column=2, sticky="ns")
        right.pack_propagate(False)
        verdict = self._panel(right)
        verdict.pack(fill="x")
        self._heading(verdict, "KẾT LUẬN")
        self.banner = tk.Frame(verdict, bg=NEUTRAL, padx=8, pady=16)
        self.banner.pack(fill="x")
        self.lbl_verdict = tk.Label(self.banner, text=VERDICTS["IDLE"][0],
                                    bg=NEUTRAL, fg="white", wraplength=320,
                                    font=("Segoe UI", 28, "bold"))
        self.lbl_verdict.pack(fill="x")
        self.lbl_verdict_sub = tk.Label(self.banner, text="", bg=NEUTRAL, fg="white",
                                        wraplength=320)
        self.lbl_verdict_sub.pack(fill="x")
        if self.robot is not None:
            # Shown only while a REVIEW board waits for the operator's bin choice.
            self.op_frame = tk.Frame(verdict, bg=CARD)
            tk.Label(self.op_frame, text="Người vận hành chọn khay cho robot:",
                     bg=CARD, fg=INK, anchor="w").pack(fill="x", pady=(8, 4))
            row = tk.Frame(self.op_frame, bg=CARD)
            row.pack(fill="x")
            for i, (text, bin_name, color) in enumerate((
                    ("✕ Lỗi", "FAIL", ERROR), ("✓ Đạt", "PASS", SUCCESS),
                    ("Không gắp", None, NEUTRAL))):
                row.columnconfigure(i, weight=1, uniform="op")
                button = self._btn(row, text, lambda b=bin_name: self._operator_choice(b))
                button.configure(bg=color, fg="white")
                button.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 4, 0), ipady=4)

        evidence = self._panel(right)
        evidence.configure(padx=8)
        evidence.pack(fill="both", expand=True, pady=8)
        self.lbl_defects = self._heading(evidence, "LỖI PHÁT HIỆN (0)")
        cols = (("number", "#", 26), ("class", "Loại lỗi", 198),
                ("conf", "Tin cậy", 66), ("source", "Nhánh", 52))
        table = tk.Frame(evidence, bg=CARD)
        table.pack(fill="both", expand=True)
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(table, columns=[col[0] for col in cols],
                                 show="headings", height=1, takefocus=True)
        for key, caption, width in cols:
            self.tree.heading(key, text=caption, anchor="w" if key == "class" else "center")
            self.tree.column(key, width=width, minwidth=width if key != "class" else 80,
                             stretch=key == "class", anchor="w" if key == "class" else "center")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        def update_scroll(first, last):
            scrollbar.set(first, last)
            if float(first) > 0 or float(last) < 1:
                scrollbar.grid()
            else:
                scrollbar.grid_remove()
        self.tree.configure(yscrollcommand=update_scroll)
        self._focus_border(self.tree, evidence)
        self._build_stats(right)

    def _build_stats(self, parent):
        panel = self._panel(parent)
        panel.pack(fill="x")
        self._heading(panel, "THỐNG KÊ CA")
        grid = tk.Frame(panel, bg=CARD)
        grid.pack(fill="x", pady=(0, 8))
        self.stat_values = {}
        for i, (key, caption) in enumerate((("total", "Tổng"), ("fail", "FAIL"),
                                            ("review", "REVIEW"), ("pass", "Không thấy lỗi"))):
            grid.columnconfigure(i, weight=1)
            tk.Label(grid, text=caption, bg=CARD, fg=MUTED).grid(row=0, column=i)
            value = tk.Label(grid, text="0", bg=CARD, fg=INK, font=MONO_FONT)
            value.grid(row=1, column=i, pady=(4, 0))
            self.stat_values[key] = value
        self._btn(panel, "Đặt lại", self._reset_stats).pack(anchor="e")

    def _focus_border(self, widget, frame):
        frame.configure(highlightthickness=2)
        widget.bind("<FocusIn>", lambda _event: frame.configure(highlightbackground=ACCENT))
        widget.bind("<FocusOut>", lambda _event: frame.configure(highlightbackground=NEUTRAL))

    def _wire_shortcuts(self):
        # Consume the accelerator before a focused button can fire its own command.
        def inspect(_event):
            self._inspect_dispatch()
            return "break"
        def activate(event):
            event.widget.invoke()
            return "break"

        shortcut = "AOIInspectShortcut"
        self.root.bind_class(shortcut, "<space>", inspect)
        self.root.bind("<space>", inspect)
        for widget in (*self._controls, self.nb, self.tree):
            widget.bindtags((shortcut, *widget.bindtags()))
            if isinstance(widget, (tk.Button, tk.Radiobutton, tk.Checkbutton)):
                widget.bind("<Return>", activate)

    def _btn(self, parent, text, command, accent=False):
        button = tk.Button(parent, text=text, command=command,
                           bg=ACCENT if accent else NEUTRAL, fg=BG if accent else INK,
                           activebackground=INK if accent else MUTED, activeforeground=BG,
                           disabledforeground=MUTED, highlightbackground=CARD,
                           highlightcolor=ACCENT, highlightthickness=2, takefocus=True,
                           relief="flat", overrelief="flat", bd=0, padx=12, pady=4,
                           font=TITLE_FONT if accent else LABEL_FONT, cursor="hand2")
        self._controls.append(button)
        return button

    def _set_busy(self, busy):
        self._busy = busy
        if busy and not self._control_states:
            self._control_states = {control: str(control.cget("state"))
                                    for control in self._controls}
            for control in self._controls:
                control.configure(state="disabled")
        elif not busy:
            for control, state in self._control_states.items():
                control.configure(state=state)
            self._control_states.clear()
            if self.camera_quality_mode:
                self.btn_save_inspection.configure(
                    state="normal" if self._last_inspection_frame is not None else "disabled")
        self.btn_inspect.configure(text="Đang kiểm tra…" if busy else INSPECT_TEXT,
                                   bg=NEUTRAL if busy else ACCENT)
        if busy:
            self.lbl_ai.configure(text="● Đang kiểm tra")
        elif self.engine_ready:
            self.lbl_ai.configure(text="● Sẵn sàng")

    def _set_status(self, text):
        self.status.configure(text=text)

    def _set_ai_led(self, ready, text):
        self.engine_ready = ready
        self.lbl_ai.configure(text="● Sẵn sàng" if ready else text)

    def _init_calibrated_camera(self) -> None:
        try:
            self.calibrated_camera = CalibratedCameraInspection(PROJECT_ROOT / "results")
        except Exception as exc:
            error = str(exc)
            self.root.after(0, lambda: self._on_engine_failed(error))
            return
        self.root.after(0, lambda: self._set_ai_led(True, "● Sẵn sàng"))
        self.root.after(0, lambda: self._set_status("Kết nối camera và đặt PCB loại A."))

    def _init_engine(self) -> None:
        try:
            self.engine = DualPathAOIEngine(weights_path=default_weights(), device=None)
            self.engine.warmup()
            self.engine_error = None
            self.engine_info = {
                "device": getattr(self.engine.yolo, "device", "?"),
                "half": getattr(self.engine.yolo, "half", None),
                "imgsz": getattr(self.engine.yolo, "imgsz", None),
                "adapt_note": getattr(self.engine.yolo, "adapt_note", ""),
                "warmup_ms": getattr(self.engine, "_warmup_ms", None),
                "weights": str(self.engine.weights_path),
            }
            self.root.after(0, self._on_engine_ready)
        except Exception as exc:  # pragma: no cover - defensive
            import traceback
            self.engine_error = f"{type(exc).__name__}: {exc}"
            self.engine_trace = traceback.format_exc(limit=8)
            self.root.after(0, lambda: self._on_engine_failed(self.engine_error))

    def _init_real_ab(self) -> None:
        try:
            inspection = RealABInspection(
                self.weights_path, PROJECT_ROOT / "results", patchcore_dirs=_patchcore_designs(),
                backbone_weights=PROJECT_ROOT / "assets" / "weights" / "wide_resnet50_2_imagenet.pth")
            inspection.warmup()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.engine_error = error
            self.root.after(0, lambda: self._on_engine_failed(error))
            return
        self.real_ab = inspection
        self.root.after(0, lambda: self._set_ai_led(True, "● Sẵn sàng"))
        self.root.after(0, lambda: self._set_status("Sẵn sàng kiểm tra PCB loại A."))

    def _on_engine_ready(self) -> None:
        self._set_ai_led(True, "● Sẵn sàng")
        self._set_status("Sẵn sàng kiểm tra.")

    def _on_engine_failed(self, err: str) -> None:
        self.engine_error = err
        if self.engine_trace:
            print(self.engine_trace, file=sys.stderr)
        self._set_ai_led(False, "● Khởi tạo thất bại")
        self.btn_inspect.config(state="disabled")
        self._inspect_error(err)

    def _report_tk_exception(self, exc_type, exc_value, exc_tb) -> None:
        """Tk callback errors: log full traceback + show dialog (never silent)."""
        import traceback
        print("[AOI] Unhandled exception in Tk callback:", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
        log_hint = os.environ.get("AOI_LOG_FILE", "")
        try:
            self._set_status(f"Lỗi: {exc_type.__name__}: {exc_value}")
        except Exception:
            pass
        if self.self_test:
            return
        try:
            messagebox.showerror(
                "Lỗi không mong muốn",
                f"{exc_type.__name__}: {exc_value}"
                + (f"\n\nChi tiết lưu tại:\n{log_hint}" if log_hint else ""))
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Mode switching.
    # ------------------------------------------------------------------ #
    def _reset_pending(self) -> None:
        self._pending_camera_regions = None
        if self.weights_path is not None:
            self._real_ab_context += 1

    def _switch_mode(self) -> None:
        self._reset_pending()
        self.p_gige.pack_forget()
        self.p_file.pack_forget()
        if self.cmb_mode.get() == "Camera GigE":
            self.p_gige.pack(fill="x", pady=(8, 0))
            self._set_status("Nguồn ảnh · Camera GigE")
        else:
            self.p_file.pack(fill="x", pady=(8, 0))
            self._set_status("Nguồn ảnh · Ảnh từ file")
            if self._file_frame is not None:
                self._show_raw(self._file_frame)

    def _read_cam_params(self) -> tuple[float, float, bool]:
        try:
            exp_us = float(self.ent_exp.get().strip() or 15000.0)
        except ValueError:
            exp_us = 15000.0
        try:
            gain_db = float(self.ent_gain.get().strip() or 2.0)
        except ValueError:
            gain_db = 2.0
        sw_trig = bool(self.chk_sw_trig.get())
        return exp_us, gain_db, sw_trig

    def _cam_scan_silent(self) -> None:
        def work():
            try:
                mgr = GigECameraManager()
                info = mgr.enumerate_devices()
                devs = info.get("devices", [])
                self.root.after(0, lambda: self._cam_scan_done(devs, mgr, silent=True))
            except Exception:
                pass
        threading.Thread(target=work, daemon=True).start()

    def _cam_scan(self) -> None:
        self._set_cam_status("Đang quét camera GigE/USB3…")
        def work():
            try:
                mgr = GigECameraManager()
                info = mgr.enumerate_devices()
                devs = info.get("devices", [])
                self.root.after(0, lambda: self._cam_scan_done(devs, mgr, silent=False))
            except Exception as exc:  # pragma: no cover
                err = str(exc)
                self.root.after(0, lambda err=err: self._cam_scan_fail(err))
        threading.Thread(target=work, daemon=True).start()

    def _cam_scan_done(self, devs: list[dict[str, Any]],
                       mgr: GigECameraManager, silent: bool = False) -> None:
        self.cam_mgr = mgr
        labels = [f"[{d.get('index', i)}] {d.get('model', 'Camera')} "
                  f"({d.get('type', '?')}, {d.get('ip', '?')})"
                  for i, d in enumerate(devs)]
        self.cmb_cam["values"] = labels or ["Chưa tìm thấy camera"]
        self.cmb_cam.current(0)
        if devs:
            self._set_cam_status(f"Tìm thấy {len(devs)} camera.")
            self._set_status(f"Tìm thấy {len(devs)} camera GigE/USB3.")
        elif not silent:
            self._set_cam_status("Kiểm tra cáp mạng và địa chỉ IP của camera.")
            self._set_status("Không tìm thấy camera GigE/USB3.")

    def _cam_scan_fail(self, err: str) -> None:
        self._set_cam_status(f"Lỗi quét camera · {err}")
        messagebox.showerror("Quét camera", err)

    def _cam_connect(self) -> None:
        if not self.cam_mgr:
            self.cam_mgr = GigECameraManager()
        try:
            info_enum = self.cam_mgr.enumerate_devices()
            devs = info_enum.get("devices", [])
            if not devs:
                messagebox.showwarning(
                    "Kết nối camera",
                    "Không tìm thấy camera GigE/USB3.\n"
                    "Kiểm tra cáp và nguồn camera rồi nhấn Quét.",
                )
                return
            self._cam_scan_done(devs, self.cam_mgr, silent=True)
        except Exception as exc:
            messagebox.showerror("Kết nối camera", str(exc))
            return

        idx = max(0, self.cmb_cam.current())
        exp_us, gain_db, sw_trig = self._read_cam_params()
        try:
            info = self.cam_mgr.connect(
                device_index=idx,
                exposure_us=exp_us,
                gain_db=gain_db,
                software_trigger=sw_trig,
            )
            self.cam_connected = True
            dev = info.get("device", {})
            self._set_cam_status(
                f"Đã kết nối · {dev.get('model', 'Camera')} ({dev.get('ip', '?')})"
            )
            self._set_status(
                f"Đã kết nối camera · {dev.get('model', 'Camera')} "
                f"(Exp={exp_us:.0f} µs, Gain={gain_db:.1f} dB)."
            )
            self.cmb_mode.set("Camera GigE")
            if self.weights_path is not None:
                self._reset_pending()
            self.nb.select(1)
            self._schedule_live_preview()
        except Exception as exc:
            messagebox.showerror("Kết nối camera", str(exc))

    def _cam_disconnect(self) -> None:
        if self.camera_quality_mode and (self._busy or self._grabbing_cam):
            self._set_status("Chờ lần chụp hiện tại xong rồi mới ngắt kết nối.")
            return
        self._reset_pending()
        self.cam_connected = False
        if self._live_timer_id is not None:
            try:
                self.root.after_cancel(self._live_timer_id)
            except Exception:
                pass
            self._live_timer_id = None
        try:
            if self.cam_mgr:
                self.cam_mgr.disconnect()
        except Exception:
            pass
        self._set_cam_status("Đã ngắt kết nối camera.")
        self._set_status("Đã ngắt kết nối camera GigE.")

    def _cam_apply_params(self) -> None:
        if self.camera_quality_mode and (self._busy or self._grabbing_cam):
            self._set_status("Chờ lần chụp hiện tại xong rồi mới đổi thông số camera.")
            return
        self._reset_pending()
        exp_us, gain_db, sw_trig = self._read_cam_params()
        if not self.cam_connected or not self.cam_mgr:
            self._set_cam_status(
                f"Đã lưu thông số (Exp={exp_us:.0f} µs, Gain={gain_db:.1f} dB)."
            )
            return
        try:
            self.cam_mgr.update_params(
                exposure_us=exp_us,
                gain_db=gain_db,
                software_trigger=sw_trig,
            )
            self._set_cam_status(
                f"Đã áp dụng: Exp={exp_us:.0f} µs · Gain={gain_db:.1f} dB · "
                f"SW Trigger {'bật' if sw_trig else 'tắt'}"
            )
            self._set_status("Đã cập nhật thông số camera.")
        except Exception as exc:
            messagebox.showerror("Thông số camera", str(exc))


    def _on_live_toggle(self) -> None:
        if self.chk_live.get() and self.cam_connected:
            self._schedule_live_preview()

    def _schedule_live_preview(self) -> None:
        if self._live_timer_id is not None:
            try:
                self.root.after_cancel(self._live_timer_id)
            except Exception:
                pass
            self._live_timer_id = None
        if self.cam_connected and self.chk_live.get():
            self._live_timer_id = self.root.after(220, self._live_preview_tick)

    def _live_preview_tick(self) -> None:
        self._live_timer_id = None
        if not self.cam_connected or not self.cam_mgr or not self.chk_live.get():
            return
        try:
            active_tab = self.nb.index(self.nb.select())
        except Exception:
            active_tab = 1
        if not self._busy and not self._grabbing_cam and active_tab == 1 and self.cmb_mode.get() == "Camera GigE":
            try:
                frame = self.cam_mgr.grab_frame_bgr(timeout_ms=600)
                self._current_frame_bgr = frame
                self._show_raw(frame)
            except Exception:
                pass
        if self.cam_connected and self.chk_live.get():
            self._live_timer_id = self.root.after(220, self._live_preview_tick)

    def _set_cam_status(self, text: str) -> None:
        self.lbl_cam_status.config(text=text)
        state = "Đã kết nối" if self.cam_connected else "Chưa kết nối"
        self.lbl_cam_badge.config(text=f"● Camera: {state}")

    def _open_file(self) -> None:
        if self._busy or self._grabbing_cam or self.camera_quality_mode or self._robot_blocking():
            return
        path = filedialog.askopenfilename(parent=self.root, title="Mở ảnh PCB",
                                         initialdir=str(PROJECT_ROOT / "samples"),
                                         filetypes=[("Ảnh PCB", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        try:
            img = _read_bgr(Path(path))
        except (OSError, ValueError, cv2.error) as exc:
            self._file_frame = None
            self._file_name = ""
            self._current_frame_bgr = None
            self.img_raw._frame_bgr = None
            self.img_raw.configure(image="", text="Chờ ảnh đầu vào")
            self._inspect_error(str(exc))
            return
        self._reset_pending()
        self._file_frame = self._current_frame_bgr = img
        self._file_name = Path(path).name
        self._show_raw(img)
        self._set_status(f"Đã mở {self._file_name}")
        if self.engine_ready:
            self._run_inspect(img, source=self._file_name)

    def _inspect_dispatch(self) -> None:
        if self._busy or self._grabbing_cam or self._robot_blocking():
            return
        if self.engine_error:
            return
        if not self.engine_ready:
            messagebox.showwarning("Kiểm tra", "Đang khởi tạo bộ kiểm tra.")
            return
        if self.cmb_mode.get() == "Ảnh từ file" and not self.camera_quality_mode:
            if self._file_frame is None:
                self._open_file()
            else:
                self._run_inspect(self._file_frame, source=self._file_name)
            return
        if not self.cam_connected or not self.cam_mgr:
            messagebox.showwarning("Kiểm tra", "Hãy kết nối camera trước khi kiểm tra.")
            return
        self._grabbing_cam = True
        self._set_busy(True)
        if self.camera_quality_mode:
            threading.Thread(target=self._inspect_calibrated_camera, daemon=True).start()
        else:
            context = self._real_ab_context if self.weights_path is not None else None
            threading.Thread(target=self._inspect_camera, args=(context,), daemon=True).start()

    def _inspect_calibrated_camera(self) -> None:
        try:
            frame = self.cam_mgr.grab_frame_bgr(timeout_ms=2500)
            result, aligned = self.calibrated_camera.inspect(frame, self._pending_camera_regions)
            self.root.after(0, lambda: self._show_calibrated_camera_result(frame, aligned, result))
        except Exception as exc:
            error = str(exc)
            self.root.after(0, lambda: self._inspect_error(error))
        finally:
            self.root.after(0, self._finish_calibrated_capture)

    def _finish_calibrated_capture(self) -> None:
        self._grabbing_cam = False
        self._set_busy(False)

    def _show_calibrated_camera_result(self, frame, aligned, result) -> None:
        self._pending_camera_regions = result.pending if self.cmb_mode.get() == "Camera GigE" else None
        self._last_inspection_frame = frame.copy()
        self._current_frame_bgr = frame
        self._show_raw(frame)
        action = result.action
        verdict = ("REVIEW" if result.regions else "NO REGIONS") if action == "INSPECT" else action
        reasons = {
            "CONFIRM": "Giữ nguyên bo mạch và nhấn Space để chụp lần hai.",
            "RECAPTURE": "Kiểm tra vị trí bo mạch và chụp lại.",
            "REVIEW": "Kiểm tra các vùng được đánh dấu trên ảnh.",
            "NO REGIONS": "Ảnh đạt điều kiện kiểm tra, không có vùng nghi vấn.",
        }
        self._render_verdict(verdict, reasons[verdict])
        items = [dict(source="branchb", bbox=region.box) for region in result.regions]
        self._render_items(aligned, items)
        self._count_result(verdict)
        self.lbl_metrics.config(text=f"B: {action}")
        self._set_status("Đã kiểm tra ảnh camera.")
        self.nb.select(0)

    def _save_inspection_png(self) -> None:
        frame = self._last_inspection_frame
        if frame is None:
            return
        path = filedialog.asksaveasfilename(
            title="Lưu ảnh gốc của lần kiểm tra", defaultextension=".png",
            filetypes=[("Ảnh PNG", "*.png")])
        if not path:
            return
        target = Path(path)
        try:
            if target.suffix.lower() != ".png":
                raise ValueError("Hãy dùng đuôi .png.")
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                raise RuntimeError("Không mã hoá được ảnh PNG.")
            target.write_bytes(encoded.tobytes())
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            messagebox.showerror("Lưu ảnh kết quả", str(exc))
            return
        self._set_status(f"Đã lưu ảnh gốc · {target}")

    def _inspect_camera(self, context: int | None = None) -> None:
        try:
            frame = self.cam_mgr.grab_frame_bgr(timeout_ms=2500)
        except Exception as exc:
            err = str(exc)
            self.root.after(0, self._finish_calibrated_capture)
            self.root.after(0, lambda: self._inspect_error(err)
                            if context is None or context == self._real_ab_context else None)
            return

        def inspect_current_capture():
            self._finish_calibrated_capture()
            if context is None or context == self._real_ab_context:
                self._current_frame_bgr = frame
                self._show_raw(frame)
                self._run_inspect(frame, source="GigE camera")

        self.root.after(0, inspect_current_capture)

    def _run_inspect(self, img: np.ndarray, source: str = "") -> None:
        self._run_inspect_worker(img, pinned=None, source=source)

    def _run_inspect_worker(self, img: np.ndarray, pinned: str | None,
                            source: str) -> None:
        if self._busy:
            return
        self._set_busy(True)
        if self.weights_path is not None:
            source_key = "GigE camera" if source == "GigE camera" else id(img)
            if source_key != self._real_ab_source:
                self._reset_pending()
                self._real_ab_source = source_key
            context = self._real_ab_context
            pending = self._pending_camera_regions
            use_b = _use_branch_b(self)
            patchcore_design = _patchcore_design(self)

        def show(res):
            if self.weights_path is not None:
                if context != self._real_ab_context:
                    return
                self._pending_camera_regions = res["b_pending"]
            self._show_result(res, source)
            self._after_result(res, source)
            self.nb.select(0)

        def work():
            try:
                if self.weights_path is not None:
                    res = self.real_ab.inspect(img, pending=pending, use_b=use_b,
                                               patchcore_design=patchcore_design)
                else:
                    res = self.engine.inspect(
                        img, fast_align=True, pinned_board_id=pinned,
                        roi_rect=None, allow_yolo_fallback=True,
                        include_visuals=True)
                self.root.after(0, lambda: show(res))
            except Exception as exc:  # pragma: no cover - defensive
                err = str(exc)
                self.root.after(0, lambda err=err: self._inspect_error(err))
            finally:
                self.root.after(0, lambda: self._set_busy(False))

        threading.Thread(target=work, daemon=True).start()

    def _inspect_error(self, err: str) -> None:
        title, color = VERDICTS["UNAVAILABLE"]
        self.banner.config(bg=color)
        self.lbl_verdict.config(text=title, fg="white", bg=color)
        self.lbl_verdict_sub.config(text=err, fg="white", bg=color)
        self.lbl_metrics.config(text="")
        self.tree.delete(*self.tree.get_children())
        self.lbl_defects.config(text="LỖI PHÁT HIỆN (0)")
        self.img_annotated._frame_bgr = None
        self.img_annotated.config(image="")
        self.img_annotated.image = None
        self._last_inspection_frame = None
        self._pending_camera_regions = None
        if hasattr(self, "btn_save_inspection"):
            self.btn_save_inspection.config(state="disabled")
        self._set_status(f"Không kiểm tra được · {err}")
        messagebox.showerror("Kiểm tra", err)

    def _render_verdict(self, verdict: str, reason: str) -> None:
        title, color = VERDICTS[verdict]
        self.banner.config(bg=color)
        self.lbl_verdict.config(text=title, fg="white", bg=color)
        self.lbl_verdict_sub.config(text=reason, fg="white", bg=color)

    def _render_items(self, frame, items) -> None:
        ordered = sorted(items, key=lambda it: -1 if it.get("conf") is None else it["conf"],
                         reverse=True)
        self.tree.delete(*self.tree.get_children())
        for index, item in enumerate(ordered, 1):
            branch = _branch_letter(item)
            conf = item.get("conf")
            self.tree.insert("", "end", values=(
                index, _item_label(item), "—" if conf is None else f"{conf:.2f}", branch))
        # PatchCore regions below the review score are listed but are not defects
        weak = sum(1 for item in ordered if item.get("p_weak"))
        self.lbl_defects.config(text=f"LỖI PHÁT HIỆN ({len(ordered) - weak})"
                                + (f" · {weak} vùng yếu" if weak else ""))
        self.img_annotated._items = ordered
        self._set_image(self.img_annotated, frame)

    def _show_result(self, res: dict[str, Any], source: str) -> None:
        frame = res.get("raw_bgr")
        if frame is None:
            frame = self._current_frame_bgr
        self._show_raw(frame)
        self._render_items(frame, res.get("items", []))
        if self.weights_path is not None:
            verdict = res["conclusion"]
            if verdict == "KHÔNG THẤY LỖI (chưa hiệu chuẩn)":
                verdict = "NO REGIONS"
            if res.get("patchcore_design"):
                strong = sum(r["score"] >= res["p_review_score"] for r in res["p_regions"])
                metrics = (f"PatchCore ({res['patchcore_design']}): {res['p_action']}, {strong} vùng ≥ "
                           f"{res['p_review_score']:.2f} · A {res['a_ms']:.0f} ms · P {res['p_ms']:.0f} ms")
            elif res["b_action"] == "SKIPPED":
                metrics = f"B: không dùng (bo loại khác) · A {res['a_ms']:.0f} ms"
            else:
                metrics = f"B: {res['b_action']} · A {res['a_ms']:.0f} ms · B {res['b_ms']:.0f} ms"
        else:
            verdict = "NO REGIONS" if res["level"] == 4 else "FAIL"
            times = res.get("timings_ms", {})
            branch_b_ms = times.get("align", 0) + times.get("diff", 0)
            metrics = f"B: {res.get('alignment_mode', '—')} · A {times.get('roi_yolo', 0):.0f} ms · B {branch_b_ms:.0f} ms"
        self._render_verdict(verdict, _reason_text(res, verdict))
        self.lbl_metrics.config(text=metrics)
        self._last_inspection_frame = None
        self._count_result(verdict)
        self._set_status(f"Đã kiểm tra · {source}")

    def _count_result(self, verdict: str) -> None:
        bucket = {"FAIL": "fail", "REWORK": "fail", "REVIEW": "review",
                  "NO REGIONS": "pass"}.get(verdict)
        if bucket:
            self._stats = {**self._stats, "total": self._stats["total"] + 1,
                           bucket: self._stats[bucket] + 1}
            for key, widget in self.stat_values.items():
                widget.config(text=str(self._stats[key]))

    def _set_image(self, label: tk.Label, img_bgr: np.ndarray) -> None:
        label._frame_bgr = img_bgr
        width, height = label.winfo_width(), label.winfo_height()
        if img_bgr is None or min(width, height) <= 1:
            return
        photo = _preview_photo(img_bgr, width, height, getattr(label, "_items", ()))
        label.config(image=photo, text="")
        label.image = photo

    def _show_raw(self, img_bgr: np.ndarray | None) -> None:
        if img_bgr is not None:
            self._set_image(self.img_raw, img_bgr)

    # ------------------------------------------------------------------ #
    # Robot sorting (scripts/robot_worker.py reads robot_io/outbox).
    # ------------------------------------------------------------------ #
    def _switch_board(self) -> None:
        self._reset_pending()
        self._set_status("Loại bo · " + _board_text(self.board_mode.get()))

    def _robot_blocking(self) -> bool:
        if getattr(self, "_robot_job", None) is not None:
            self._set_status("Chờ robot gắp xong rồi mới kiểm tra bo tiếp theo.")
            return True
        if getattr(self, "_robot_review", None) is not None:
            self._set_status("Chọn khay Lỗi / Đạt / Không gắp cho bo hiện tại trước.")
            return True
        return False

    def _set_robot_badge(self, text: str) -> None:
        self.lbl_robot.config(text=f"● Robot: {text}")

    def _after_result(self, res: dict[str, Any], source: str) -> None:
        if getattr(self, "robot", None) is None:
            return
        decision = sort_decision(res)
        if decision["action"] == "pick":
            self._robot_send(decision["bin"], res["conclusion"], "auto", source)
        elif decision["action"] == "operator":
            self._robot_review = (res["conclusion"], source)
            self.op_frame.pack(fill="x")
            self._set_robot_badge("chờ người vận hành")
        else:
            self._set_robot_badge("không gắp")

    def _operator_choice(self, bin_name: str | None) -> None:
        if self._robot_review is None:
            return
        conclusion, source = self._robot_review
        self._robot_review = None
        self.op_frame.pack_forget()
        if bin_name is None:
            self._set_robot_badge("Sẵn sàng")
            self._set_status("Không gắp bo này. Lấy bo ra bằng tay.")
            return
        self._robot_send(bin_name, conclusion, "operator", source)

    def _robot_send(self, bin_name: str, conclusion: str, decided_by: str, source: str) -> None:
        try:
            job_id = self.robot.send(bin_name, conclusion, decided_by, source)
        except OSError as exc:
            self._set_robot_badge("lỗi")
            messagebox.showerror("Robot", f"Không ghi được lệnh cho robot: {exc}")
            return
        self._robot_job = {"id": job_id, "bin": bin_name,
                           "deadline": time.monotonic() + ROBOT_TIMEOUT_S}
        self._set_robot_badge(f"đang gắp → khay {BIN_TEXT[bin_name]}")
        self.root.after(250, self._robot_poll)

    def _robot_poll(self) -> None:
        job = self._robot_job
        if job is None:
            return
        try:
            ack = self.robot.poll(job["id"])
        except (OSError, ValueError):
            ack = None
        if ack is None:
            if time.monotonic() < job["deadline"]:
                self.root.after(250, self._robot_poll)
                return
            ack = {"status": "timeout"}
        self._robot_job = None
        status = ack.get("status")
        if status == "done":
            self._set_robot_badge("Sẵn sàng")
            self._set_status(f"Robot đã đặt bo vào khay {BIN_TEXT[job['bin']]}.")
            return
        text = {"error": "lỗi", "expired": "lệnh hết hạn",
                "timeout": "không phản hồi"}.get(status, str(status))
        self._set_robot_badge(text)
        self._set_status(f"Robot {text}. Kiểm tra robot và lấy bo ra bằng tay nếu cần.")
        messagebox.showwarning("Robot", f"Robot {text}. {ack.get('error', '')}\n"
                                        "Kiểm tra robot và lấy bo ra bằng tay nếu cần.")

    def _reset_stats(self) -> None:
        self._stats = dict.fromkeys(self.stat_values, 0)
        for widget in self.stat_values.values():
            widget.config(text="0")

    def _wait_engine(self, timeout_s: float = 180.0) -> bool:
        import time as _t
        t0 = _t.time()
        while _t.time() - t0 < timeout_s:
            if self.engine_ready and (self.engine is not None or self.real_ab is not None):
                return True
            if self.engine_error:
                return False
            _t.sleep(0.5)
        return self.engine_ready

    def run_self_test_blocking(self) -> int:
        """Blocking self-test for --self-test (no Tk mainloop needed)."""
        if not self._wait_engine():
            print(f"SELF_TEST_FAIL: engine failed: {self.engine_error}")
            return 1
        try:
            if self.real_ab is not None:
                frame = _read_bgr(PROJECT_ROOT / "samples" / "A_05_v1_c01.png")
                self._show_result(self.real_ab.inspect(frame), "self-test A_05")
                print("SELF_TEST_OK")
                return 0
            r1 = self.engine.inspect(_read_bgr(DEMO_ROOT / "images" / "Short" / "short01.jpg"),
                                     fast_align=True)
            r2 = self.engine.inspect(_read_bgr(LOCAL_TEMPLATES / "01.JPG"),
                                     fast_align=True)
        except Exception as exc:  # pragma: no cover
            print(f"SELF_TEST_FAIL: {exc}")
            return 1
        ok = (r1["level"] == 1 and r2["level"] == 4
              and r1["board_id"] == "01" and r2["board_id"] == "01"
              and r1["annotated_bgr"] is not None
              and r2["heatmap_bgr"] is not None)
        # Push results through the widget path to verify display code.
        try:
            self._show_result(r1, "self-test short01")
            self._show_result(r2, "self-test GOOD_01")
            clean_shown = self.lbl_verdict.cget("text") == VERDICTS["NO REGIONS"][0]
            self._show_result(r1, "self-test short01 re-check")
            fail_shown = self.lbl_verdict.cget("text") == VERDICTS["FAIL"][0]
            ok = ok and clean_shown and fail_shown and self._stats["fail"] and self._stats["pass"]
        except Exception as exc:
            print(f"SELF_TEST_FAIL: widget error: {exc}")
            return 1
        print("SELF_TEST_OK" if ok else "SELF_TEST_FAIL")
        return 0 if ok else 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="PCB Dual-Path AOI Desktop")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--camera-quality", action="store_true")
    mode.add_argument("--weights", choices=("best.pt", "best2.pt", "real_all_boards.pt", "real_3class.pt"),
                      help="Select an imported checkpoint for full-frame A + calibrated B.")
    parser.add_argument("--robot", action="store_true",
                        help="Send sorting commands to scripts/robot_worker.py via robot_io/.")
    parser.add_argument("--self-test", action="store_true")
    options = parser.parse_args()
    self_test = options.self_test or os.environ.get("AOI_SELF_TEST", "").strip() == "1"
    weights_path = None
    if options.robot and not options.weights:
        parser.error("--robot needs --weights (A + B pipeline).")
    if options.weights:
        weights_path = _RP / "assets" / "weights" / "pcb_camera" / options.weights
        if not weights_path.is_file():
            parser.error(f"Checkpoint not found: {weights_path}")
    if self_test and (not os.environ.get("DISPLAY") and os.name != "nt"):
        os.environ.setdefault("MPLBACKEND", "Agg")
    root = tk.Tk()
    if self_test:
        try:
            root.withdraw()
        except Exception:
            pass
    app = TkAOIApp(root, self_test=self_test, weights_path=weights_path,
                   robot_dir=_RP / "robot_io" if options.robot else None)
    root.report_callback_exception = app._report_tk_exception
    if not self_test and os.environ.get("AOI_LAUNCHER") == "1":
        def _hide_console_when_visible(tries: int = 0) -> None:
            # Hide the launcher console only once the Tk window is mapped,
            # so any crash before this point stays visible to the user.
            try:
                root.update_idletasks()
                visible = bool(root.winfo_viewable())
            except Exception:
                visible = False
            if visible or tries >= 40:
                try:
                    from src.ui.launcher import hide_console
                    hide_console()
                except Exception:
                    pass
                try:
                    root.deiconify()
                    root.lift()
                    root.attributes("-topmost", True)
                    root.after(500, lambda: root.attributes("-topmost", False))
                except Exception:
                    pass
                return
            root.after(250, lambda: _hide_console_when_visible(tries + 1))

        root.after(300, _hide_console_when_visible)
    if self_test:
        exit_box = {"code": 1}

        def _poll_self_test():
            if app.engine_error:
                print(f"SELF_TEST_FAIL: engine failed: {app.engine_error}")
                exit_box["code"] = 1
                root.quit()
                return
            if not app.engine_ready or (app.engine is None and app.real_ab is None):
                root.after(100, _poll_self_test)
                return
            exit_box["code"] = app.run_self_test_blocking()
            root.quit()

        root.after(100, _poll_self_test)
        root.mainloop()
        try:
            root.destroy()
        except Exception:
            pass
        return exit_box["code"]

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
