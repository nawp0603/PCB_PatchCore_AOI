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
import json
import subprocess
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
from configs.station import (FAIL_SLOTS, PASS_SLOTS, SOURCE_SLOTS, TEACH_POINTS,
                              derive_drop_slots, migrate_station_config)  # noqa: E402
from src.core.aoi_engine import DualPathAOIEngine  # noqa: E402
from src.core.calibrated_camera import CalibratedCameraInspection  # noqa: E402
from src.core.real_ab_pipeline import RealABInspection  # noqa: E402
from src.core.board_align import board_region  # noqa: E402
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
BOX_R = "#A855F7"
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


def _load_station_config(path: Path) -> dict:
    """Load deployed configuration and persist the one-way station-name migration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    changed = migrate_station_config(config)
    changed = derive_drop_slots(config) or changed
    if changed:
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config


_SAMPLE_POSES = {
    "A11": (150.0, 180.0, -20.0, 0.0),
    "C11": (150.0, -180.0, -20.0, 0.0),
}


def _sample_station_points(config: dict) -> list[str]:
    """Reject shipped example poses before a real robot can use Automation."""
    points = {slot: (config.get("pick_positions") or {}).get(slot) for slot in SOURCE_SLOTS}
    points.update(config.get("bins") or {})
    return [name for name, sample in _SAMPLE_POSES.items()
            if tuple(points.get(name, ())) == sample]


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


BIN_TEXT = {"A11": "OK (A11)", "A12": "OK (A12)", "A13": "OK (A13)",
            "C11": "NG (C11)", "C12": "NG (C12)", "C13": "NG (C13)"}
# The robot must start a command within RobotBridge.ttl_s (30 s); this leaves ~90 s to move.
ROBOT_TIMEOUT_S = 120.0
ROBOT_PICK_DELAY_MS = 5000
AUTOMATION_POLL_MS = 250
AUTOMATION_MAX_RECAPTURES = 2
PCB_MIN_AREA_RATIO = 0.02


def _pcb_present(frame: np.ndarray) -> bool:
    """Require a substantial green solder-mask region before allowing any robot route."""
    return frame is not None and frame.ndim == 3 and board_region(frame).mean() >= PCB_MIN_AREA_RATIO


def _empty_pcb_result(frame: np.ndarray) -> dict:
    return {"level": 3, "conclusion": "RECAPTURE", "verdict": "RECAPTURE",
            "reasons": ["Không phát hiện PCB trong khung hình; robot không gắp."], "details": [],
            "items": [], "b_regions_unmatched": [], "patchcore_design": None,
            "p_action": "SKIPPED", "p_regions": [], "p_review_score": None, "p_ms": 0.0,
            "a_items": [], "raw_bgr": frame, "annotated_bgr": frame.copy(), "heatmap_bgr": None,
            "alignment_mode": "skipped", "b_action": "SKIPPED", "b_pending": None,
            "branch_b": False, "a_ms": 0.0, "b_ms": 0.0, "timings_ms": {"total": 0.0},
            "pcb_present": False}


def _use_branch_b(app) -> bool:
    """Board of design A uses A + B; another design has no Branch B template."""
    mode = getattr(app, "board_mode", None)
    return mode is None or mode.get() == "A"


def _patchcore_designs() -> dict:
    """Designs with a PatchCore memory: {design: (profile dir, memory dir)}."""
    return patchcore_designs(PROJECT_ROOT)


def _patchcore_design(app):
    """Selected design that uses YOLO + PatchCore, or None."""
    mode = getattr(app, "board_mode", None)
    value = mode.get() if mode is not None else None
    return value if value not in (None, "A", "OTHER") else None


def _board_text(value) -> str:
    if value == "AUTO":
        return "Tự nhận diện (YOLO + PatchCore)"
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
        branch = {"branchb": "B", "patchcore": "P"}.get(item.get("source"), "A")
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
                 weights_path: Path | None = None, robot_dir: Path | None = None,
                 robot_autostart: str | None = None):
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
        self.robot_autostart = robot_autostart
        self._robot_process = None
        self._robot_state = "offline"
        self._robot_control_job = None
        self._robot_teach_mode = False
        self._force_teach_on_next_start = False
        self._last_pose_read_at = 0.0
        self._latest_robot_pose = None
        self._robot_job: dict[str, Any] | None = None
        self._robot_delay_job: dict[str, str] | None = None
        self._robot_delay_after_id = None
        self._robot_review: tuple[str, str] | None = None
        self._automation_enabled = False
        self._automation_phase = "idle"
        self._automation_after_id = None
        self._automation_deadline = None
        self._automation_recaptures = 0
        self._automation_camera_started = False
        self._automation_robot_started = False
        self._automation_slot_pending = {slot: None for slot in SOURCE_SLOTS}
        self._pass_slot_index = 0
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
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

        # Bootstrap engine + warmup on a background thread (window shows < 0.5s).
        init = self._init_calibrated_camera if self.camera_quality_mode else self._init_engine
        if self.weights_path is not None:
            init = self._init_real_ab
        threading.Thread(target=init, daemon=True).start()
        if not self.self_test:
            # Auto-scan LAN/USB3 cameras on startup without blocking UI.
            self.root.after(400, self._cam_scan_silent)
            if self.robot is not None:
                self.root.after(500, self._poll_robot_status)
                if self.robot_autostart:
                    self.root.after(750, self._start_robot)

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
            self.lbl_robot = self._badge(header, "● Robot: Chưa kết nối")

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
        left_outer = tk.Frame(parent, bg=BG, width=300)
        left_outer.grid(row=0, column=0, sticky="ns")
        left_outer.grid_propagate(False)
        scrollbar = ttk.Scrollbar(left_outer, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        canvas = tk.Canvas(left_outer, bg=BG, highlightthickness=0,
                           yscrollcommand=scrollbar.set, width=300)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=canvas.yview)
        left = tk.Frame(canvas, bg=BG, width=300)
        window = canvas.create_window((0, 0), window=left, anchor="nw")

        def resize_content(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(window, width=canvas.winfo_width())

        left.bind("<Configure>", resize_content)
        canvas.bind("<Configure>", resize_content)

        def wheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), "units")

        canvas.bind("<MouseWheel>", wheel)
        left.bind("<MouseWheel>", wheel)
        source = self._panel(left)
        source.pack(fill="x")
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
            # Production inspection always identifies the profile from the captured board.
            self.board_mode = tk.StringVar(value="AUTO")

        actions = tk.Frame(left, bg=BG)
        actions.pack(fill="x", pady=(8, 0))
        self.btn_inspect = self._btn(actions, INSPECT_TEXT, self._inspect_dispatch, accent=True)
        self.btn_inspect.pack(fill="x", ipady=8)
        if self.robot is not None:
            row = tk.Frame(actions, bg=BG)
            row.pack(fill="x", pady=(8, 0))
            row.columnconfigure((0, 1), weight=1, uniform="automation")
            self.btn_automation = self._btn(row, "BẬT AUTOMATION", self._start_automation,
                                            accent=True)
            self.btn_automation.grid(row=0, column=0, sticky="ew", padx=(0, 4))
            self.btn_automation_stop = self._btn(
                row, "DỪNG AUTOMATION", lambda: self._stop_automation(unlock_teaching=True))
            self.btn_automation_stop.grid(row=0, column=1, sticky="ew", padx=(4, 0))
            self._build_robot_controls(actions)
        if self.camera_quality_mode:
            self.btn_save_inspection = self._btn(actions, "Lưu ảnh kết quả", self._save_inspection_png)
            self.btn_save_inspection.configure(state="disabled")
            self.btn_save_inspection.pack(fill="x", pady=(8, 0))

    def _build_robot_controls(self, parent) -> None:
        panel = self._panel(parent)
        panel.pack(fill="x", pady=(0, 8))
        self._heading(panel, "ROBOT PHÂN LOẠI")
        self.robot_port = tk.StringVar()
        row = tk.Frame(panel, bg=CARD)
        row.pack(fill="x")
        tk.Label(row, text="Cổng COM", bg=CARD, fg=MUTED).pack(side="left")
        self.cmb_robot_port = ttk.Combobox(row, textvariable=self.robot_port, state="readonly",
                                           width=11, values=[])
        self.cmb_robot_port.pack(side="left", expand=True, fill="x", padx=8)
        self._btn(row, "Quét COM", self._scan_robot_ports).pack(side="right")
        self._scan_robot_ports()
        self.lbl_robot_pose_inline = tk.Label(
            panel, text="Tọa độ Dobot: chưa đọc", bg=CARD, fg=ACCENT, anchor="w", font=MONO_FONT)
        self.lbl_robot_pose_inline.pack(fill="x", pady=(6, 0))
        self._btn(panel, "Sao chép tọa độ", self._copy_robot_pose).pack(fill="x", pady=(4, 0))
        self.robot_mode = tk.StringVar(value=self.robot_autostart or "sim")
        row = tk.Frame(panel, bg=CARD)
        row.pack(fill="x")
        for value, text in (("sim", "Giả lập"), ("real", "Dobot thật")):
            tk.Radiobutton(row, text=text, variable=self.robot_mode, value=value,
                           bg=CARD, fg=INK, selectcolor=BG, activebackground=NEUTRAL,
                           activeforeground=INK).pack(side="left", expand=True, fill="x")
        row = tk.Frame(panel, bg=CARD)
        row.pack(fill="x", pady=(8, 0))
        row.columnconfigure((0, 1), weight=1, uniform="robot")
        self.btn_robot_connect = self._btn(row, "Kết nối + Home", self._start_robot)
        self.btn_robot_connect.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_robot_disconnect = self._btn(row, "Ngắt robot", self._stop_robot)
        self.btn_robot_disconnect.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self._btn(panel, "Dạy tọa độ (không Home)", self._start_robot_teaching).pack(fill="x", pady=(8, 0))
        row = tk.Frame(panel, bg=CARD)
        row.pack(fill="x", pady=(8, 0))
        row.columnconfigure((0, 1), weight=1, uniform="robot")
        self._btn(row, "Về Home", self._home_robot).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        stop = self._btn(row, "DỪNG KHẨN CẤP", self._emergency_stop)
        stop.configure(bg=ERROR, fg="white")
        stop.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.lbl_robot_status = tk.Label(panel, text="Chưa kết nối", bg=CARD, fg=MUTED,
                                         anchor="w", wraplength=250)
        self.lbl_robot_status.pack(fill="x", pady=(8, 0))
        self.teach_toggle = self._btn(panel, "▸ Dạy tọa độ", self._toggle_teach_panel)
        self.teach_toggle.pack(fill="x", pady=(8, 0))
        teach = tk.Frame(panel, bg=CARD)
        self.teach_panel = teach
        self._heading(teach, "DẠY TỌA ĐỘ")
        self.teach_point = tk.StringVar(value="B11")
        row = tk.Frame(teach, bg=CARD)
        row.pack(fill="x")
        tk.Label(row, text="Lưu điểm", bg=CARD, fg=MUTED).pack(side="left")
        ttk.Combobox(row, textvariable=self.teach_point, state="readonly", width=14,
                      values=("Home", "B11", "A11", "C11", "C12")).pack(
                         side="left", expand=True, fill="x", padx=(8, 0))
        self.teach_fields = []
        row = tk.Frame(teach, bg=CARD)
        row.pack(fill="x", pady=(8, 0))
        for name in ("X", "Y", "Z", "R"):
            cell = tk.Frame(row, bg=CARD)
            cell.pack(side="left", expand=True, fill="x", padx=(0, 4))
            tk.Label(cell, text=name, bg=CARD, fg=MUTED).pack(anchor="w")
            field = tk.Entry(cell, bg=BG, fg=INK, insertbackground=INK, width=7)
            field.insert(0, "0.0")
            field.pack(fill="x")
            self.teach_fields.append(field)
        self.lbl_current_pose = tk.Label(teach, text="Tọa độ hiện tại: chưa đọc", bg=CARD,
                                         fg=MUTED, anchor="w", font=MONO_FONT)
        self.lbl_current_pose.pack(fill="x", pady=(6, 0))
        row = tk.Frame(teach, bg=CARD)
        row.pack(fill="x", pady=(8, 0))
        for text, command in (("Đọc tọa độ", self._read_robot_pose),
                              ("Đi tới", self._move_to_teach_pose),
                              ("Lưu điểm", self._save_teach_pose)):
            self._btn(row, text, command).pack(side="left", expand=True, fill="x", padx=(0, 4))
        self._btn(teach, "HỌC TỌA ĐỘ HIỆN TẠI",
                  self._save_current_robot_pose).pack(fill="x", pady=(6, 0))
        self.teach_point.trace_add("write", lambda *_: self._load_teach_point())
        self._load_teach_point()
        self.teach_live = tk.IntVar(value=0)
        tk.Checkbutton(teach, text="Theo dõi tọa độ", variable=self.teach_live,
                       command=self._toggle_teach_live, bg=CARD, fg=INK, selectcolor=BG,
                       activebackground=NEUTRAL, activeforeground=INK, anchor="w").pack(
                           fill="x", pady=(8, 0))
        self._btn(teach, "Xác nhận trạm đã hiệu chuẩn", self._confirm_station_calibrated).pack(
            fill="x", pady=(6, 0))
        self.lbl_saved_points = tk.Label(teach, text="", bg=CARD, fg=MUTED, anchor="w",
                                         justify="left", font=("Consolas", 8))
        self.lbl_saved_points.pack(fill="x", pady=(8, 0))
        self._refresh_taught_points()

    def _toggle_teach_panel(self) -> None:
        if self.teach_panel.winfo_manager():
            self.teach_panel.pack_forget()
            self.teach_toggle.config(text="▸ Dạy tọa độ")
        else:
            self.teach_panel.pack(fill="x", pady=(8, 0))
            self.teach_toggle.config(text="▾ Dạy tọa độ")

    def _open_pose_window(self) -> None:
        window = getattr(self, "_pose_window", None)
        if window is not None and window.winfo_exists():
            window.deiconify()
            window.lift()
            self._read_robot_pose()
            return
        window = tk.Toplevel(self.root)
        window.title("Dobot · Tọa độ")
        window.configure(bg=CARD)
        window.resizable(False, False)
        window.transient(self.root)
        self._pose_window = window
        frame = tk.Frame(window, bg=CARD, padx=16, pady=16)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="TỌA ĐỘ HIỆN TẠI", bg=CARD, fg=INK,
                 font=TITLE_FONT).pack(anchor="w")
        self.lbl_pose_window = tk.Label(frame, text="Chưa đọc", bg=CARD, fg=ACCENT,
                                        font=("Consolas", 13, "bold"), anchor="w")
        self.lbl_pose_window.pack(fill="x", pady=(8, 16))
        self._btn(frame, "Đọc tọa độ từ Dobot", self._read_robot_pose).pack(fill="x")
        tk.Label(frame, text="ĐIỂM ĐÃ LƯU", bg=CARD, fg=INK,
                 font=TITLE_FONT).pack(anchor="w", pady=(16, 4))
        self.lbl_saved_points_window = tk.Label(frame, text="", bg=CARD, fg=MUTED,
                                                 justify="left", anchor="w", font=("Consolas", 10))
        self.lbl_saved_points_window.pack(fill="x")
        self._refresh_taught_points()
        if self._robot_state == "ready":
            self._read_robot_pose()

    def _scan_robot_ports(self) -> None:
        try:
            from serial.tools import list_ports
            ports = [(port.device, port.description) for port in list_ports.comports()]
        except Exception as exc:
            ports = []
            detail = f"Không quét được COM: {exc}"
        else:
            detail = "Không tìm thấy cổng COM." if not ports else " · ".join(
                f"{device}: {description}" for device, description in ports)
        values = [device for device, _ in ports]
        self.cmb_robot_port.configure(values=values)
        configured = ""
        try:
            configured = str(_load_station_config(
                PROJECT_ROOT / "configs" / "robot_positions.json").get("port", ""))
        except (OSError, ValueError):
            pass
        selected = self.robot_port.get()
        if selected not in values:
            selected = configured if configured in values else (values[0] if values else "")
        self.robot_port.set(selected)
        if hasattr(self, "lbl_robot_status"):
            self.lbl_robot_status.config(text=detail)

    def _teach_pose(self) -> list[float] | None:
        try:
            return [float(field.get()) for field in self.teach_fields]
        except ValueError:
            messagebox.showerror("Dạy tọa độ", "X, Y, Z, R phải là các số hợp lệ.")
            return None

    def _load_teach_point(self) -> None:
        point = {"Home": "home"}.get(self.teach_point.get(), self.teach_point.get())
        try:
            config = _load_station_config(PROJECT_ROOT / "configs" / "robot_positions.json")
            pose = (config[point] if point == "home" else config["pick_positions"][point]
                    if point in SOURCE_SLOTS else config["bins"][point])
        except (OSError, ValueError, KeyError):
            return
        self._set_teach_pose(pose)

    def _set_teach_pose(self, pose) -> None:
        for field, value in zip(self.teach_fields, pose):
            field.delete(0, "end")
            field.insert(0, f"{float(value):.2f}")

    def _set_current_robot_pose(self, pose) -> None:
        if len(pose) != 4:
            return
        self._latest_robot_pose = [float(value) for value in pose]
        if hasattr(self, "lbl_current_pose") and len(pose) == 4:
            self.lbl_current_pose.config(
                text="Tọa độ hiện tại: " + ", ".join(f"{float(value):.2f}" for value in pose))
        if hasattr(self, "lbl_robot_pose_inline") and len(pose) == 4:
            self.lbl_robot_pose_inline.config(
                text="Tọa độ Dobot: X={:.1f}  Y={:.1f}\nZ={:.1f}  R={:.1f}".format(*map(float, pose)))
        if hasattr(self, "lbl_pose_window") and len(pose) == 4:
            self.lbl_pose_window.config(
                text="X={:.2f}  Y={:.2f}  Z={:.2f}  R={:.2f}".format(*map(float, pose)))

    def _read_pose_into_teach_fields(self, pose) -> None:
        self._set_current_robot_pose(pose)
        self._set_teach_pose(pose)

    def _copy_robot_pose(self) -> None:
        pose = self._latest_robot_pose
        if pose is None:
            self._set_status("Chưa có tọa độ Dobot để sao chép.")
            return
        text = "[" + ", ".join(f"{value:.2f}" for value in pose) + "]"
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status(f"Đã sao chép tọa độ: {text}")

    def _refresh_taught_points(self) -> None:
        try:
            config = _load_station_config(PROJECT_ROOT / "configs" / "robot_positions.json")
            points = [("Home", config["home"]),
                       *[(name, config["pick_positions"][name]) for name in SOURCE_SLOTS],
                       *[(name, config["bins"][name]) for name in ("A11", "C11", "C12")]]
        except (OSError, ValueError, KeyError):
            return
        text = "Điểm đã lưu:\n" + "\n".join(
            f"{name}: " + ", ".join(f"{float(value):.1f}" for value in pose)
            for name, pose in points)
        self.lbl_saved_points.config(text=text)
        if hasattr(self, "lbl_saved_points_window"):
            self.lbl_saved_points_window.config(text=text.replace("Điểm đã lưu:\n", ""))

    def _read_robot_pose(self) -> None:
        self._send_robot_control("read_pose", {}, self._read_pose_into_teach_fields)

    def _toggle_teach_live(self) -> None:
        if self.teach_live.get():
            self._poll_teach_pose()

    def _poll_teach_pose(self) -> None:
        if not self.teach_live.get():
            return
        if self._robot_state == "ready" and self._robot_control_job is None:
            self._send_robot_control("read_pose", {}, self._set_current_robot_pose, quiet=True)
        self.root.after(500, self._poll_teach_pose)

    def _move_to_teach_pose(self) -> None:
        pose = self._teach_pose()
        if pose is None:
            return
        if not messagebox.askyesno(
                "Di chuyển robot",
                f"Robot sẽ di chuyển đến X={pose[0]:.2f}, Y={pose[1]:.2f}, "
                f"Z={pose[2]:.2f}, R={pose[3]:.2f}.\nĐảm bảo vùng hoạt động trống."):
            return
        self._send_robot_control("move", {"pose": pose}, self._set_current_robot_pose)

    def _save_teach_pose(self) -> None:
        pose = self._teach_pose()
        if pose is None:
            return
        point = {"Home": "home"}.get(self.teach_point.get(), self.teach_point.get())
        self._persist_teach_pose(point, pose)

    def _save_current_robot_pose(self) -> None:
        if self._robot_state != "ready":
            messagebox.showwarning("Dạy trực tiếp", "Robot chưa sẵn sàng để đọc tọa độ hiện tại.")
            return
        point = {"Home": "home"}.get(self.teach_point.get(), self.teach_point.get())
        def save_current(pose) -> None:
            self._set_current_robot_pose(pose)
            self._set_teach_pose(pose)
            if point != "home":
                try:
                    home = _load_station_config(
                        PROJECT_ROOT / "configs" / "robot_positions.json")["home"]
                except (OSError, ValueError, KeyError):
                    home = ()
                if len(home) == 4 and all(abs(float(value) - float(reference)) <= 3.0
                                          for value, reference in zip(pose, home)):
                    messagebox.showwarning(
                        "Dạy tọa độ",
                        f"Dobot đang ở Home. Không lưu Home vào điểm {point.upper()}. "
                        "Di chuyển tay robot đến vị trí khay trước rồi đọc lại tọa độ.")
                    return
            self._persist_teach_pose(point, pose)

        self._send_robot_control("read_pose", {}, save_current)

    def _persist_teach_pose(self, point: str, pose: list[float]) -> None:
        path = PROJECT_ROOT / "configs" / "robot_positions.json"
        try:
            config = _load_station_config(path)
            if point == "home":
                config[point] = pose
            elif point in SOURCE_SLOTS:
                config["pick_positions"][point] = pose
            else:
                config["bins"][point] = pose
            derive_drop_slots(config)
            # Any changed taught position requires an explicit production re-verification.
            config["station_calibrated"] = False
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Dạy tọa độ", f"Không lưu được tọa độ: {exc}")
            return
        self._refresh_taught_points()
        self._set_status(f"Đã lưu điểm {point}. Ngắt và kết nối lại robot để dùng tọa độ mới.")

    def _confirm_station_calibrated(self) -> None:
        path = PROJECT_ROOT / "configs" / "robot_positions.json"
        try:
            config = _load_station_config(path)
            derive_drop_slots(config)
            points = [config["home"], *[config["pick_positions"][slot] for slot in SOURCE_SLOTS],
                        *[config["bins"][name] for name in ("A11", "C11", "C12")]]
            if any(not isinstance(point, list) or len(point) != 4 for point in points):
                raise ValueError("Thiếu một hoặc nhiều tọa độ Home, B11, A11, C11, C12.")
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Hiệu chuẩn trạm", f"Không thể xác nhận: {exc}")
            return
        if not messagebox.askyesno(
                "Xác nhận hiệu chuẩn",
                "Bạn xác nhận đã kiểm tra Home, B11, A11, C11, C12; vùng robot trống "
                "và E-stop vật lý sẵn sàng?\n\n"
                "Xác nhận sẽ cho phép Automation điều khiển Dobot thật để gắp PCB."):
            return
        config["station_calibrated"] = True
        try:
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Hiệu chuẩn trạm", str(exc))
            return
        self._set_status("Đã xác nhận hiệu chuẩn. Ngắt và kết nối lại robot trước khi bật Automation.")

    def _send_robot_control(self, action: str, payload: dict, on_done, quiet: bool = False) -> None:
        if self.robot is None or self._robot_state != "ready":
            self._set_status("Robot chưa sẵn sàng; không thể dạy tọa độ.")
            return
        if self._robot_control_job is not None:
            self._set_status("Đang chờ lệnh dạy tọa độ trước hoàn thành.")
            return
        try:
            control_id = self.robot.send_control(action, payload)
        except OSError as exc:
            messagebox.showerror("Dạy tọa độ", f"Không gửi được lệnh tới robot: {exc}")
            return
        self._robot_control_job = {"id": control_id, "on_done": on_done, "quiet": quiet,
                                   "deadline": time.monotonic() + ROBOT_TIMEOUT_S}
        if not quiet:
            self._set_robot_badge("đang dạy tọa độ")
        self.root.after(250, self._poll_robot_control)

    def _poll_robot_control(self) -> None:
        job = self._robot_control_job
        if job is None:
            return
        try:
            ack = self.robot.poll_control(job["id"])
        except (OSError, ValueError):
            ack = None
        if ack is None and time.monotonic() < job["deadline"]:
            self.root.after(250, self._poll_robot_control)
            return
        self._robot_control_job = None
        if ack is not None and ack.get("status") == "done":
            job["on_done"](ack.get("result", {}).get("pose", ()))
            if not job["quiet"]:
                self._set_status("Đã hoàn thành lệnh dạy tọa độ.")
            return
        error = "không phản hồi" if ack is None else ack.get("error", ack.get("status", "lỗi"))
        self._set_robot_badge("lỗi")
        messagebox.showwarning("Dạy tọa độ", f"Robot {error}.")

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
        if getattr(self, "_automation_enabled", False):
            self._stop_automation("Automation dừng: camera đã được ngắt.")
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

    def _start_automation(self) -> None:
        if getattr(self, "_automation_enabled", False):
            return
        if self.robot is None:
            messagebox.showwarning("Automation", "Khởi động UI với tùy chọn robot trước khi chạy Automation.")
            return
        if self.weights_path is None:
            messagebox.showwarning("Automation", "Automation yêu cầu mô hình nhận diện đã được chọn.")
            return
        if self.camera_quality_mode:
            messagebox.showwarning("Automation", "Chế độ camera quality chưa hỗ trợ phân loại và gắp tự động.")
            return
        if self.robot_mode.get() == "real":
            config_path = PROJECT_ROOT / "configs" / "robot_positions.json"
            try:
                config = _load_station_config(config_path)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Automation", f"Không đọc được cấu hình robot: {exc}")
                return
            if not config.get("station_calibrated", False):
                messagebox.showwarning(
                    "Automation chưa thể chạy",
                    "Trạm Dobot chưa hiệu chuẩn nên robot đang ở chế độ dạy tọa độ và bị khóa gắp.\n\n"
                    "Hãy dạy và kiểm tra Home, B, A, C11, C12, C13; sau đó xác nhận các tọa độ "
                    "và đặt station_calibrated thành true trước khi chạy Automation.",
                )
                return
            sample_points = _sample_station_points(config)
            if sample_points:
                messagebox.showwarning(
                    "Automation chưa thể chạy",
                    "Các điểm sau vẫn là tọa độ mẫu, chưa được phép gắp thật: "
                    + ", ".join(sample_points) + ".\n\n"
                    "Chọn từng điểm trong Dạy tọa độ, đặt đầu hút đúng vị trí và bấm "
                    "HỌC TỌA ĐỘ HIỆN TẠI.",
                )
                return
        if self._robot_teach_mode:
            if messagebox.askyesno(
                    "Chuyển sang chạy sản xuất",
                    "Tọa độ đã hiệu chuẩn. Automation sẽ dừng worker dạy tọa độ, sau đó khởi động "
                    "lại robot ở chế độ gắp thật và về Home. Tiếp tục?"):
                self._stop_robot()
                self.root.after(1000, self._start_automation)
            return
        self._automation_enabled = True
        self._automation_phase = "chờ bộ nhận diện"
        self._automation_deadline = time.monotonic() + 180.0
        self._automation_recaptures = 0
        self._automation_slot_pending = {slot: None for slot in SOURCE_SLOTS}
        self._pass_slot_index = 0
        self._automation_camera_started = False
        self._automation_robot_started = False
        self.cmb_mode.set("Camera GigE")
        self._switch_mode()
        self._set_status("Automation đã bật. Đang chuẩn bị camera, robot và bộ nhận diện.")
        self._automation_schedule(0)

    def _stop_automation(self, reason: str = "Automation đã dừng.", unlock_teaching: bool = False) -> None:
        if self._automation_after_id is not None:
            try:
                self.root.after_cancel(self._automation_after_id)
            except Exception:
                pass
        self._automation_after_id = None
        if getattr(self, "_robot_delay_after_id", None) is not None:
            try:
                self.root.after_cancel(self._robot_delay_after_id)
            except Exception:
                pass
        self._robot_delay_after_id = None
        self._robot_delay_job = None
        self._automation_enabled = False
        self._automation_phase = "idle"
        self._automation_deadline = None
        self._set_status(reason)
        if unlock_teaching:
            self._unlock_teaching_after_automation()

    def _unlock_teaching_after_automation(self) -> None:
        """Return a real station to teaching mode after an operator stops Automation."""
        if (not hasattr(self, "robot_mode") or self.robot_mode.get() != "real"
                or self._robot_teach_mode):
            return
        path = PROJECT_ROOT / "configs" / "robot_positions.json"
        try:
            config = _load_station_config(path)
            config["station_calibrated"] = False
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dạy tọa độ", f"Không mở được chế độ dạy: {exc}")
            return
        self._force_teach_on_next_start = True
        self._stop_robot()
        self._set_status("Automation đã dừng. Đang mở lại robot ở chế độ dạy tọa độ.")
        self.root.after(1000, self._start_robot)

    def _automation_schedule(self, delay_ms: int = AUTOMATION_POLL_MS) -> None:
        if self._automation_enabled and self._automation_after_id is None:
            self._automation_after_id = self.root.after(delay_ms, self._automation_tick)

    def _automation_waiting_timed_out(self) -> bool:
        return self._automation_deadline is not None and time.monotonic() >= self._automation_deadline

    def _automation_tick(self) -> None:
        self._automation_after_id = None
        if not self._automation_enabled:
            return
        if self.engine_error:
            self._stop_automation("Automation dừng: bộ nhận diện gặp lỗi.")
            return
        if not self.engine_ready:
            if self._automation_waiting_timed_out():
                self._stop_automation("Automation dừng: bộ nhận diện không sẵn sàng.")
                return
            self._automation_schedule()
            return
        if not self.cam_connected:
            if not self._automation_camera_started:
                self._automation_camera_started = True
                self._automation_phase = "đang kết nối camera"
                self._automation_deadline = time.monotonic() + 30.0
                self._set_status("Automation: đang kết nối camera.")
                self._cam_connect()
            if not self.cam_connected:
                if self._automation_waiting_timed_out():
                    self._stop_automation("Automation dừng: không kết nối được camera.")
                    return
                self._automation_schedule()
                return
        if self._robot_teach_mode:
            self._stop_automation("Automation dừng: robot đang ở chế độ dạy tọa độ.")
            return
        if self._robot_state != "ready":
            if not self._automation_robot_started:
                self._automation_robot_started = True
                self._automation_phase = "đang kết nối robot"
                self._automation_deadline = time.monotonic() + 60.0
                self._set_status("Automation: đang kết nối robot và về Home.")
                self._start_robot()
            if self._robot_state == "error" or self._automation_waiting_timed_out():
                self._stop_automation("Automation dừng: robot không sẵn sàng.")
                return
            self._automation_schedule()
            return
        if self._robot_delay_job is not None or self._robot_job is not None:
            self._automation_phase = "chờ robot gắp"
            self._automation_schedule()
            return
        if self._robot_review is not None:
            self._stop_automation("Automation tạm dừng: cần người vận hành chọn khay cho bo hiện tại.")
            return
        if self._busy or self._grabbing_cam:
            self._automation_schedule()
            return
        self._automation_phase = "đang nhận diện"
        self._automation_capture_slot()

    def _automation_capture_slot(self) -> None:
        self._grabbing_cam = True
        self._set_busy(True)
        self._set_status("Automation: chụp PCB tại B11.")

        def work():
            try:
                frame = self.cam_mgr.grab_frame_bgr(timeout_ms=2500)
                self.root.after(0, lambda: self._automation_capture_ready(frame))
            except Exception as exc:
                error = str(exc)
                self.root.after(0, lambda error=error: self._automation_capture_error(error))

        threading.Thread(target=work, daemon=True).start()

    def _automation_capture_error(self, error: str) -> None:
        self._finish_calibrated_capture()
        self._inspect_error(error)

    def _automation_capture_ready(self, frame: np.ndarray) -> None:
        self._finish_calibrated_capture()
        if not self._automation_enabled:
            return
        self._current_frame_bgr = frame
        self._show_raw(frame)
        self._run_inspect(frame, source="GigE camera:B11")

    def _automation_after_result(self, res: dict[str, Any], source: str = "") -> None:
        if not getattr(self, "_automation_enabled", False):
            return
        if res.get("pcb_present") is False:
            self._stop_automation("Automation dừng: không phát hiện PCB tại B11, robot không gắp.")
            return
        conclusion = res.get("conclusion", "")
        slot = source.rsplit(":", 1)[-1] if source.endswith(tuple(f":{name}" for name in SOURCE_SLOTS)) else None
        if sort_decision(res)["action"] == "pick":
            self._automation_recaptures = 0
            if slot:
                self._automation_slot_pending[slot] = None
            self._automation_phase = "chờ robot gắp"
            self._automation_schedule()
            return
        if conclusion in ("CONFIRM", "RECAPTURE", "REWORK"):
            self._automation_recaptures += 1
            if self._automation_recaptures > AUTOMATION_MAX_RECAPTURES:
                self._stop_automation("Automation tạm dừng: chụp lại quá nhiều lần. Kiểm tra PCB/camera.")
                return
            self._automation_phase = "chụp lại"
            self._set_status(f"Automation: chụp lại PCB ({self._automation_recaptures}/{AUTOMATION_MAX_RECAPTURES}).")
            self._automation_schedule(500)
            return
        self._stop_automation(f"Automation tạm dừng: AOI trả {conclusion}; cần người vận hành kiểm tra.")
        messagebox.showwarning(
            "Automation tạm dừng",
            f"AOI trả kết luận {conclusion}, chưa thể tự chọn khay gắp.\n\n"
            "OK sẽ vào khay A; FAIL lớp 0/1/2 sẽ vào C11/C12/C13. "
            "REVIEW/CONFIRM/RECAPTURE không gắp để tránh phân loại nhầm.",
        )

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
            slot = source.rsplit(":", 1)[-1] if source.endswith(tuple(f":{name}" for name in SOURCE_SLOTS)) else None
            source_key = "GigE camera" if source == "GigE camera" else id(img)
            if slot is None and source_key != self._real_ab_source:
                self._reset_pending()
                self._real_ab_source = source_key
            context = self._real_ab_context
            pending = self._automation_slot_pending[slot] if slot else self._pending_camera_regions

        def show(res):
            if self.weights_path is not None:
                if context != self._real_ab_context:
                    return
                if slot:
                    self._automation_slot_pending[slot] = res["b_pending"]
                else:
                    self._pending_camera_regions = res["b_pending"]
            self._show_result(res, source)
            self._after_result(res, source)
            self._automation_after_result(res, source)
            self.nb.select(0)

        def work():
            try:
                if self.weights_path is not None:
                    res = _empty_pcb_result(img) if not _pcb_present(img) else self.real_ab.inspect_auto(img, pending=pending)
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
        if getattr(self, "_automation_enabled", False):
            self._stop_automation(f"Automation dừng: không nhận diện được PCB ({err}).")
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
            branch = {"branchb": "B", "patchcore": "P"}.get(item.get("source"), "A")
            conf = item.get("conf")
            self.tree.insert("", "end", values=(
                index, _item_label(item), "—" if conf is None else f"{conf:.2f}", branch))
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
            verdict = "RECAPTURE" if res.get("pcb_present") is False else res["conclusion"]
            if verdict == "KHÔNG THẤY LỖI (chưa hiệu chuẩn)":
                verdict = "NO REGIONS"
            if res.get("pcb_present") is False:
                metrics = "Không phát hiện PCB · robot không gắp"
            elif res.get("patchcore_design"):
                strong = sum(r["score"] >= res["p_review_score"] for r in res["p_regions"])
                selected = f"Tự nhận diện: {res['selected_profile']} · " if res.get("selected_profile") else ""
                metrics = (f"{selected}PatchCore ({res['patchcore_design']}): {res['p_action']}, {strong} vùng ≥ "
                            f"{res['p_review_score']:.2f} · A {res['a_ms']:.0f} ms · P {res['p_ms']:.0f} ms")
            elif res["b_action"] == "SKIPPED":
                metrics = ("Tự nhận diện: không xác định · " if res.get("selected_profile") == "OTHER" else "") + \
                          f"B: không dùng (bo loại khác) · A {res['a_ms']:.0f} ms"
            else:
                selected = "Tự nhận diện: A · " if res.get("selected_profile") == "A" else ""
                metrics = f"{selected}B: {res['b_action']} · A {res['a_ms']:.0f} ms · B {res['b_ms']:.0f} ms"
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
    def _start_robot(self) -> None:
        if self._robot_process is not None and self._robot_process.poll() is None:
            self._set_status("Robot đang chạy.")
            return
        try:
            existing_status = self.robot.status()
        except (OSError, ValueError):
            existing_status = None
        if (existing_status is not None
                and time.time() - existing_status.get("updated", 0) <= 5.0
                and existing_status.get("status") in ("connecting", "ready", "busy")):
            self._set_robot_state(existing_status["status"], existing_status.get("detail", ""))
            messagebox.showwarning(
                "Robot đã chạy",
                "Một worker robot khác đang hoạt động. Không khởi động worker thứ hai trên cùng cổng COM.")
            return
        mode = self.robot_mode.get()
        config_path = PROJECT_ROOT / "configs" / "robot_positions.json"
        teach_mode = False
        if mode == "real":
            port = self.robot_port.get().strip()
            if not port:
                messagebox.showwarning("Robot", "Hãy quét và chọn cổng COM của Dobot.")
                return
            try:
                config = _load_station_config(config_path)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Robot", f"Không đọc được cấu hình robot: {exc}")
                return
            if not teach_mode and not config.get("station_calibrated", False):
                teach_mode = self._force_teach_on_next_start
                self._force_teach_on_next_start = False
                if not teach_mode:
                    teach_mode = messagebox.askyesno(
                        "Chế độ dạy tọa độ",
                        "Trạm chưa hiệu chuẩn. UI sẽ chỉ kết nối để đọc/di chuyển dạy tọa độ; "
                        "robot không về HOME và không gắp PCB.\nTiếp tục?")
                    if not teach_mode:
                        return
            elif not teach_mode and not messagebox.askyesno(
                    "Kết nối Dobot",
                    "Robot sẽ kết nối qua cổng đã cấu hình và di chuyển về HOME.\n"
                    "Đảm bảo vùng hoạt động trống và nút dừng khẩn cấp sẵn sàng."):
                return
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / "robot_worker.py"),
                   "--io", str(self.robot.root)]
        if mode == "sim":
            command.append("--sim")
        else:
            command.extend(("--port", port))
            if teach_mode:
                command.append("--teach")
        try:
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
            self._robot_process = subprocess.Popen(command, cwd=PROJECT_ROOT, creationflags=flags)
        except OSError as exc:
            self._set_robot_state("error", f"Không khởi động được worker: {exc}")
            return
        self._robot_teach_mode = teach_mode
        self._set_robot_state("connecting", "Đang kết nối chế độ dạy tọa độ..." if teach_mode
                               else "Đang kết nối robot và về HOME...")

    def _start_robot_teaching(self) -> None:
        if self.robot_mode.get() != "real":
            messagebox.showwarning("Dạy tọa độ", "Chọn Dobot thật trước khi dạy tọa độ.")
            return
        path = PROJECT_ROOT / "configs" / "robot_positions.json"
        try:
            config = _load_station_config(path)
            config["station_calibrated"] = False
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dạy tọa độ", f"Không mở được chế độ dạy: {exc}")
            return
        self._force_teach_on_next_start = True
        if self._robot_process is not None and self._robot_process.poll() is None:
            self._stop_robot()
            self.root.after(1000, self._start_robot)
        else:
            self._start_robot()

    def _home_robot(self) -> None:
        if not messagebox.askyesno("Về Home", "Robot sẽ di chuyển về điểm Home đã lưu. Tiếp tục?"):
            return
        self._send_robot_control("home", {}, self._set_current_robot_pose)

    def _emergency_stop(self) -> None:
        if getattr(self, "_automation_enabled", False):
            self._stop_automation("Automation dừng khẩn cấp.")
        port = self.robot_port.get().strip() if hasattr(self, "robot_port") else ""
        process = self._robot_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        self._robot_process = None
        self._robot_teach_mode = False
        if not port:
            self._set_robot_state("error", "Worker đã dừng; không có cổng COM để gửi lệnh stop.")
            return
        try:
            subprocess.Popen([sys.executable, str(PROJECT_ROOT / "scripts" / "robot_stop.py"),
                              "--port", port], cwd=PROJECT_ROOT)
        except OSError as exc:
            self._set_robot_state("error", f"Worker đã dừng, nhưng không gửi được stop: {exc}")
            return
        self._set_robot_state("error", "Đã gửi dừng phần mềm. Dùng E-stop vật lý nếu robot còn di chuyển.")

    def _stop_robot(self) -> None:
        if getattr(self, "_automation_enabled", False):
            self._stop_automation("Automation dừng: robot đã được ngắt.")
        process = self._robot_process
        if process is None or process.poll() is not None:
            self._set_robot_state("offline", "Robot chưa được UI này khởi động.")
            return
        process.terminate()
        self._robot_process = None
        self._robot_teach_mode = False
        self._set_robot_state("offline", "Đã dừng worker robot. Kiểm tra tay robot trước khi thao tác lại.")

    def _set_robot_state(self, state: str, detail: str = "") -> None:
        self._robot_state = state
        labels = {
            "connecting": "đang kết nối",
            "ready": "Sẵn sàng",
            "busy": "đang gắp",
            "error": "lỗi",
            "offline": "chưa kết nối",
        }
        text = labels.get(state, state)
        self._set_robot_badge(text)
        if hasattr(self, "lbl_robot_status"):
            self.lbl_robot_status.config(text=detail or text)

    def _poll_robot_status(self) -> None:
        if self.robot is None:
            return
        try:
            status = self.robot.status()
        except (OSError, ValueError):
            status = None
        if status is not None and time.time() - status.get("updated", 0) <= 5.0:
            self._set_robot_state(status.get("status", "offline"), status.get("detail", ""))
        elif self._robot_process is not None and self._robot_process.poll() is not None:
            self._robot_process = None
            self._set_robot_state("error", "Worker robot đã dừng. Xem terminal để biết lỗi.")
        elif self._robot_state in ("ready", "busy", "connecting"):
            self._set_robot_state("offline", "Không nhận được heartbeat từ worker robot.")
        self.root.after(500, self._poll_robot_status)

    def shutdown(self) -> None:
        self._stop_automation("Automation dừng: đang đóng ứng dụng.")
        self._cam_disconnect()
        self._emergency_stop()
        self.root.destroy()

    def _switch_board(self) -> None:
        self._reset_pending()
        self._set_status("Loại bo · " + _board_text(self.board_mode.get()))

    def _robot_blocking(self) -> bool:
        if getattr(self, "_robot_delay_job", None) is not None:
            self._set_status("Đã có kết luận; chờ 5 giây trước khi robot gắp.")
            return True
        if getattr(self, "_robot_job", None) is not None:
            self._set_status("Chờ robot gắp xong rồi mới kiểm tra bo tiếp theo.")
            return True
        if getattr(self, "_robot_control_job", None) is not None:
            self._set_status("Chờ lệnh dạy tọa độ hoàn thành rồi mới kiểm tra bo tiếp theo.")
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
        if res.get("pcb_present") is False:
            self._set_robot_badge("không gắp")
            self._set_status("Không phát hiện PCB trong khung hình; robot không gắp.")
            return
        decision = sort_decision(res)
        if decision["action"] == "pick":
            source_slot = source.rsplit(":", 1)[-1] if source.endswith(tuple(f":{name}" for name in SOURCE_SLOTS)) else "B11"
            if decision["bin"] == "A11" and getattr(self, "_automation_enabled", False):
                index = getattr(self, "_pass_slot_index", 0)
                if index >= len(PASS_SLOTS):
                    self._stop_automation("Automation dừng: khay A đã đầy. Lấy PCB ra rồi chạy lại.")
                    return
                decision["bin"] = PASS_SLOTS[index]
            self._schedule_robot_pick(decision["bin"], res["conclusion"], "auto", source, source_slot)
        elif decision["action"] == "recapture":
            self._render_verdict("RECAPTURE", "Lỗi cần sửa. Kiểm tra bo và chụp lại; robot không gắp.")
            self._set_robot_badge("không gắp")
        else:
            self._set_robot_badge("không gắp")
            conclusion = res.get("conclusion", "không xác định")
            if conclusion == "FAIL":
                self._set_status("AOI báo FAIL nhưng không có lớp lỗi 0/1/2 để chọn khay C; robot không gắp.")
            elif conclusion in ("REVIEW", "CONFIRM", "RECAPTURE"):
                self._set_status(f"AOI trả {conclusion}; robot không gắp cho đến khi ảnh được xác nhận/chụp lại.")
            else:
                self._set_status(f"AOI trả {conclusion}; không có quy tắc gắp tương ứng.")

    def _schedule_robot_pick(self, bin_name: str, conclusion: str, decided_by: str, source: str,
                              source_slot: str = "B11") -> None:
        self._robot_delay_job = {"bin": bin_name, "conclusion": conclusion,
                                 "decided_by": decided_by, "source": source, "source_slot": source_slot}
        self._set_robot_badge(f"chờ 5 giây → khay {BIN_TEXT[bin_name]}")
        self._set_status("Kết luận đã xác nhận. Robot sẽ gắp sau 5 giây.")
        self._robot_delay_after_id = self.root.after(
            ROBOT_PICK_DELAY_MS, self._dispatch_delayed_robot_pick)

    def _dispatch_delayed_robot_pick(self) -> None:
        self._robot_delay_after_id = None
        job = self._robot_delay_job
        self._robot_delay_job = None
        if job is not None:
            self._robot_send(job["bin"], job["conclusion"], job["decided_by"], job["source"],
                             job["source_slot"])

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

    def _robot_send(self, bin_name: str, conclusion: str, decided_by: str, source: str,
                    source_slot: str = "B11") -> None:
        if getattr(self, "_robot_teach_mode", False):
            self._set_status("Robot đang ở chế độ dạy tọa độ; không gửi lệnh gắp.")
            return
        if getattr(self, "_robot_state", "ready") != "ready":
            self._set_robot_badge("chưa sẵn sàng")
            self._set_status("Robot chưa sẵn sàng; PCB giữ tại B, không gửi lệnh gắp.")
            return
        try:
            job_id = self.robot.send(bin_name, conclusion, decided_by, source, source_slot)
        except OSError as exc:
            self._set_robot_badge("lỗi")
            messagebox.showerror("Robot", f"Không ghi được lệnh cho robot: {exc}")
            return
        self._robot_job = {"id": job_id, "bin": bin_name, "is_pass": bin_name in PASS_SLOTS,
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
            if job["is_pass"] and getattr(self, "_automation_enabled", False):
                self._pass_slot_index += 1
            self._set_robot_badge("Sẵn sàng")
            self._set_status(f"Robot đã đặt bo vào khay {BIN_TEXT[job['bin']]}.")
            if getattr(self, "_automation_enabled", False):
                self._automation_phase = "đang nhận diện"
                self._automation_schedule()
            return
        text = {"error": "lỗi", "expired": "lệnh hết hạn",
                "timeout": "không phản hồi"}.get(status, str(status))
        self._set_robot_badge(text)
        self._set_status(f"Robot {text}. Kiểm tra robot và lấy bo ra bằng tay nếu cần.")
        if getattr(self, "_automation_enabled", False):
            self._stop_automation(f"Automation dừng: robot {text}; không thử gắp lại PCB hiện tại.")
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
    parser.add_argument("--robot-autostart", choices=("sim", "real"),
                        help="Start the robot worker from the UI after it opens.")
    parser.add_argument("--self-test", action="store_true")
    options = parser.parse_args()
    self_test = options.self_test or os.environ.get("AOI_SELF_TEST", "").strip() == "1"
    weights_path = None
    if (options.robot or options.robot_autostart) and not options.weights:
        parser.error("--robot needs --weights (A + B pipeline).")
    if options.robot_autostart and not options.robot:
        parser.error("--robot-autostart needs --robot.")
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
                   robot_dir=_RP / "robot_io" if options.robot else None,
                   robot_autostart=options.robot_autostart)
    root.protocol("WM_DELETE_WINDOW", app.shutdown)
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
