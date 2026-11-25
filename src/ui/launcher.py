"""Crash-safe GUI launcher for the packaged PCB_DualPath_AOI.exe.

The packaged .exe is a renamed (Authenticode-signed) python.exe whose
`Lib/sitecustomize.py` calls `run()` below. This module guarantees that a
startup failure is NEVER silent:

  * stdout/stderr are tee'd into a UTF-8 log file
    (`<app>/logs/aoi_YYYYmmdd_HHMMSS.log`, fallback `%LOCALAPPDATA%`),
    with `errors="replace"` so printing paths/unicode can never crash.
  * `faulthandler` writes native crashes (CUDA / torch DLL faults) to the log.
  * `sys.excepthook` + `threading.excepthook` log uncaught exceptions from
    the main thread and worker threads (engine init, camera, inspection).
  * The console stays VISIBLE while torch/YOLO load; it is hidden only after
    the Tk window is actually on screen (`hide_console()` called by the app).
    On any fatal error the console is re-shown and a MessageBox points to the
    log file.

Flags:
  --console            keep the console window visible (debugging)
  AOI_KEEP_CONSOLE=1   same, via environment
"""
from __future__ import annotations

import datetime as _dt
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

APP_TITLE = "PCB Dual-Path AOI"
_LOG_KEEP = 20

_log_fh: Any = None
_log_path: Path | None = None


# --------------------------------------------------------------------------- #
# Console helpers (Windows only; no-ops elsewhere).
# --------------------------------------------------------------------------- #
def _console_hwnd() -> int:
    if os.name != "nt":
        return 0
    try:
        import ctypes
        return int(ctypes.windll.kernel32.GetConsoleWindow() or 0)
    except Exception:
        return 0


def keep_console_requested() -> bool:
    return ("--console" in sys.argv
            or os.environ.get("AOI_KEEP_CONSOLE", "").strip() == "1")


def hide_console() -> None:
    """Hide the console window (called once the Tk window is visible)."""
    if keep_console_requested():
        return
    hwnd = _console_hwnd()
    if hwnd:
        try:
            import ctypes
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
        except Exception:
            pass


def show_console() -> None:
    hwnd = _console_hwnd()
    if hwnd:
        try:
            import ctypes
            ctypes.windll.user32.ShowWindow(hwnd, 5)  # SW_SHOW
        except Exception:
            pass


def log_file() -> Path | None:
    return _log_path


# --------------------------------------------------------------------------- #
# Logging (tee stdout/stderr -> UTF-8 file).
# --------------------------------------------------------------------------- #
class _Tee:
    """Write-through stream: original console stream + log file."""

    def __init__(self, primary: Any, logfh: Any) -> None:
        self._primary = primary
        self._log = logfh

    def write(self, s: Any) -> int:
        if not isinstance(s, str):
            s = str(s)
        for st in (self._primary, self._log):
            if st is None:
                continue
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:
        for st in (self._primary, self._log):
            if st is None:
                continue
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        if self._primary is not None:
            return self._primary.fileno()
        return self._log.fileno()

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def errors(self) -> str:
        return "replace"

    def __getattr__(self, name: str) -> Any:
        target = self._primary if self._primary is not None else self._log
        return getattr(target, name)


def _pick_log_dir(root: Path) -> Path:
    cands = [root / "logs"]
    lad = os.environ.get("LOCALAPPDATA")
    if lad:
        cands.append(Path(lad) / "PCB_DualPath_AOI" / "logs")
    import tempfile
    cands.append(Path(tempfile.gettempdir()) / "PCB_DualPath_AOI_logs")
    for d in cands:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return d
        except Exception:
            continue
    return Path.cwd()


def _prune_logs(d: Path) -> None:
    try:
        logs = sorted(d.glob("aoi_*.log"), key=lambda p: p.stat().st_mtime)
        for p in logs[:-_LOG_KEEP]:
            try:
                p.unlink()
            except Exception:
                pass
    except Exception:
        pass


def setup_logging(root: Path) -> Path:
    global _log_fh, _log_path
    if _log_path is not None:
        return _log_path
    d = _pick_log_dir(root)
    _prune_logs(d)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = d / f"aoi_{stamp}.log"
    _log_fh = open(_log_path, "a", encoding="utf-8", errors="replace",
                   buffering=1)

    for name in ("stdout", "stderr"):
        orig = getattr(sys, name, None)
        if orig is not None:
            try:
                orig.reconfigure(errors="replace")
            except Exception:
                pass
        setattr(sys, name, _Tee(orig, _log_fh))

    try:
        import faulthandler
        faulthandler.enable(file=_log_fh, all_threads=True)
    except Exception:
        pass

    def _excepthook(etype, value, tb):
        print("[launcher] UNCAUGHT EXCEPTION (main thread):", file=sys.stderr)
        traceback.print_exception(etype, value, tb, file=sys.stderr)

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        tname = getattr(args.thread, "name", "?")
        print(f"[launcher] UNCAUGHT EXCEPTION in thread '{tname}':",
              file=sys.stderr)
        traceback.print_exception(args.exc_type, args.exc_value,
                                  args.exc_traceback, file=sys.stderr)

    sys.excepthook = _excepthook
    threading.excepthook = _thread_hook
    os.environ["AOI_LOG_FILE"] = str(_log_path)
    return _log_path


# --------------------------------------------------------------------------- #
# Fatal error reporting.
# --------------------------------------------------------------------------- #
def fatal(message: str, self_test: bool = False) -> None:
    """Log + re-show console + MessageBox (GUI mode only)."""
    log = _log_path
    print(f"[launcher] FATAL: {message}", file=sys.stderr)
    try:
        sys.stderr.flush()
    except Exception:
        pass
    show_console()
    if self_test or os.name != "nt":
        return
    text = (f"{message}\n\nFull details were saved to:\n{log}\n\n"
            "Tip: run 'PCB_DualPath_AOI.exe --console' to keep the "
            "console window open.")
    try:
        import ctypes
        # MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST
        ctypes.windll.user32.MessageBoxW(0, text, f"{APP_TITLE} - Error",
                                         0x10 | 0x10000 | 0x40000)
    except Exception:
        pass


def _banner(root: Path, log: Path) -> None:
    import platform
    print("=" * 72)
    print(f" {APP_TITLE} - starting ({_dt.datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f" exe     : {sys.executable}")
    print(f" root    : {root}")
    print(f" python  : {sys.version.split()[0]} ({platform.platform()})")
    print(f" log     : {log}")
    print(" Loading AI engine (torch + YOLO11n + ORB cache)... please wait.")
    print(" This console hides automatically once the window is ready.")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def run(root: Path | str | None = None) -> int:
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    self_test = ("--self-test" in sys.argv
                 or os.environ.get("AOI_SELF_TEST", "").strip() == "1")
    log = setup_logging(root)
    os.environ["AOI_LAUNCHER"] = "1"
    _banner(root, log)

    try:
        from src.ui.tkinter_app import main as app_main
    except BaseException as exc:  # noqa: BLE001 - must never be silent
        traceback.print_exc()
        fatal("Failed to load AOI modules (torch / ultralytics / OpenCV / "
              f"Tkinter):\n{type(exc).__name__}: {exc}", self_test)
        return 1

    try:
        if os.environ.get("AOI_SIMULATE_STARTUP_ERROR", "").strip() == "1":
            raise RuntimeError("SimulatedStartupError (AOI_SIMULATE_STARTUP_ERROR=1)")
        rc = int(app_main() or 0)
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        fatal(f"Application crashed:\n{type(exc).__name__}: {exc}", self_test)
        return 1
    print(f"[launcher] exit code {rc}")
    return rc
