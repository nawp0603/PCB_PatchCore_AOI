"""Direct HIK MVS GigE Vision / USB3 Vision Camera Manager (MV-GE500C 5MP).

Uses the official Hikrobot MVS SDK (MvCameraControl.dll) via ctypes to enumerate
GigE/USB cameras, configure optimal GigE packet size, ExposureTime, Gain, and
TriggerMode, and grab 5MP frames directly into BGR8 numpy arrays in RAM.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def _prepare_mvs_dll_and_import() -> Any:
    """Locate signed MvCameraControl.dll on Windows and import MvCameraControl_class."""
    mv_import_dir = Path(__file__).resolve().parent / "mv_import"
    if str(mv_import_dir) not in sys.path:
        sys.path.insert(0, str(mv_import_dir))

    candidates: List[str] = []
    runenv = os.getenv("MVCAM_COMMON_RUNENV")
    if runenv:
        candidates.append(os.path.join(runenv, "Win64_x64"))
    for env_key in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.getenv(env_key)
        if base:
            candidates.append(os.path.join(base, "Common Files", "MVS", "Runtime", "Win64_x64"))

    resolved_dll: Optional[str] = None
    for cand in candidates:
        dll_file = os.path.join(cand, "MvCameraControl.dll")
        if os.path.isfile(dll_file):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(cand)
            os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")
            resolved_dll = dll_file
            break

    orig_windll = ctypes.WinDLL

    class _PatchedWinDLL(orig_windll):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            name: str,
            mode: int = ctypes.DEFAULT_MODE,
            handle: Any = None,
            use_errno: bool = False,
            use_last_error: bool = False,
            winmode: Any = None,
        ) -> None:
            target = resolved_dll if (name == "MvCameraControl.dll" and resolved_dll) else name
            super().__init__(
                target,
                mode=mode,
                handle=handle,
                use_errno=use_errno,
                use_last_error=use_last_error,
                winmode=None,
            )

    ctypes.WinDLL = _PatchedWinDLL  # type: ignore[misc]
    try:
        import MvCameraControl_class as mv_mod
    finally:
        ctypes.WinDLL = orig_windll  # type: ignore[misc]

    return mv_mod


def _decode_char_array(arr: Any) -> str:
    raw = memoryview(arr).tobytes()
    null_idx = raw.find(b"\x00")
    if null_idx != -1:
        raw = raw[:null_idx]
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace").strip()


def _format_ip(ip_uint: int) -> str:
    return (
        f"{(ip_uint >> 24) & 0xFF}."
        f"{(ip_uint >> 16) & 0xFF}."
        f"{(ip_uint >> 8) & 0xFF}."
        f"{ip_uint & 0xFF}"
    )


class GigECameraManager:
    """Thread-safe HIK MVS GigE / USB3 Camera Manager."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mv: Any = None
        self._cam: Any = None
        self._connected: bool = False
        self._grabbing: bool = False
        self._software_trigger: bool = True
        self._device_info: Dict[str, Any] = {}
        self._sdk_initialized: bool = False
        self._last_grab_ts: float = 0.0
        self._last_frame_bgr: Optional[np.ndarray] = None

    def _ensure_sdk(self) -> Any:
        if self._mv is None:
            self._mv = _prepare_mvs_dll_and_import()
        if not self._sdk_initialized:
            self._mv.MvCamera.MV_CC_Initialize()
            self._sdk_initialized = True
        return self._mv

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            try:
                mv = self._ensure_sdk()
                ver = mv.MvCamera.MV_CC_GetSDKVersion()
                sdk_ver = f"{(ver >> 24) & 0xFF}.{(ver >> 16) & 0xFF}.{(ver >> 8) & 0xFF}.{ver & 0xFF}"
            except Exception as exc:
                return {
                    "sdk_available": False,
                    "sdk_error": str(exc),
                    "connected": False,
                    "device": None,
                }
            return {
                "sdk_available": True,
                "sdk_version": sdk_ver,
                "connected": self._connected,
                "software_trigger": self._software_trigger,
                "device": self._device_info if self._connected else None,
            }

    def enumerate_devices(self) -> Dict[str, Any]:
        with self._lock:
            mv = self._ensure_sdk()
            dev_list = mv.MV_CC_DEVICE_INFO_LIST()
            tlayer = mv.MV_GIGE_DEVICE | mv.MV_USB_DEVICE
            ret = mv.MvCamera.MV_CC_EnumDevices(tlayer, dev_list)
            if ret != 0:
                raise RuntimeError(f"MV_CC_EnumDevices failed: 0x{ret:08X}")

            devices: List[Dict[str, Any]] = []
            for idx in range(int(dev_list.nDeviceNum)):
                info = cast(dev_list.pDeviceInfo[idx], POINTER(mv.MV_CC_DEVICE_INFO)).contents
                if info.nTLayerType in (mv.MV_GIGE_DEVICE, getattr(mv, "MV_GENTL_GIGE_DEVICE", 64)):
                    gige = info.SpecialInfo.stGigEInfo
                    model = _decode_char_array(gige.chModelName) or "GigE Camera"
                    serial = _decode_char_array(gige.chSerialNumber)
                    user_name = _decode_char_array(gige.chUserDefinedName)
                    ip_str = _format_ip(int(gige.nCurrentIp))
                    label = user_name or model
                    devices.append({
                        "index": idx,
                        "type": "GigE",
                        "model": model,
                        "serial": serial,
                        "ip": ip_str,
                        "display_name": f"[{idx}] {label} (SN: {serial}, IP: {ip_str})",
                    })
                elif info.nTLayerType == mv.MV_USB_DEVICE:
                    usb = info.SpecialInfo.stUsb3VInfo
                    model = _decode_char_array(usb.chModelName) or "USB3 Camera"
                    serial = _decode_char_array(usb.chSerialNumber)
                    user_name = _decode_char_array(usb.chUserDefinedName)
                    label = user_name or model
                    devices.append({
                        "index": idx,
                        "type": "USB3",
                        "model": model,
                        "serial": serial,
                        "ip": "USB3",
                        "display_name": f"[{idx}] {label} (SN: {serial}, USB3)",
                    })

            ver = mv.MvCamera.MV_CC_GetSDKVersion()
            sdk_ver = f"{(ver >> 24) & 0xFF}.{(ver >> 16) & 0xFF}.{(ver >> 8) & 0xFF}.{ver & 0xFF}"
            return {
                "sdk_version": sdk_ver,
                "count": len(devices),
                "devices": devices,
            }

    def connect(
        self,
        device_index: int = 0,
        exposure_us: float = 15000.0,
        gain_db: float = 2.0,
        software_trigger: bool = True,
    ) -> Dict[str, Any]:
        with self._lock:
            self._disconnect_unlocked()
            mv = self._ensure_sdk()

            dev_list = mv.MV_CC_DEVICE_INFO_LIST()
            ret = mv.MvCamera.MV_CC_EnumDevices(mv.MV_GIGE_DEVICE | mv.MV_USB_DEVICE, dev_list)
            if ret != 0:
                raise RuntimeError(f"MV_CC_EnumDevices failed: 0x{ret:08X}")
            if int(dev_list.nDeviceNum) == 0:
                raise RuntimeError("No GigE/USB3 camera detected on LAN/USB.")
            if device_index < 0 or device_index >= int(dev_list.nDeviceNum):
                device_index = 0

            st_dev = cast(dev_list.pDeviceInfo[device_index], POINTER(mv.MV_CC_DEVICE_INFO)).contents
            cam = mv.MvCamera()
            ret = cam.MV_CC_CreateHandle(st_dev)
            if ret != 0:
                raise RuntimeError(f"MV_CC_CreateHandle failed: 0x{ret:08X}")

            ret = cam.MV_CC_OpenDevice(mv.MV_ACCESS_Exclusive, 0)
            if ret != 0:
                cam.MV_CC_DestroyHandle()
                raise RuntimeError(f"MV_CC_OpenDevice failed: 0x{ret:08X} (close MVS Client if open)")

            packet_size = 0
            if st_dev.nTLayerType == mv.MV_GIGE_DEVICE:
                pkt = cam.MV_CC_GetOptimalPacketSize()
                if int(pkt) > 0:
                    cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(pkt))
                    packet_size = int(pkt)
                cam.MV_CC_SetIntValueEx("GevSCPD", 400)

            cam.MV_CC_SetBayerCvtQuality(1)
            self._cam = cam
            self._connected = True
            self._apply_params_unlocked(exposure_us, gain_db, software_trigger)

            ret = cam.MV_CC_StartGrabbing()
            if ret != 0:
                self._disconnect_unlocked()
                raise RuntimeError(f"MV_CC_StartGrabbing failed: 0x{ret:08X}")
            self._grabbing = True

            if st_dev.nTLayerType == mv.MV_GIGE_DEVICE:
                gige = st_dev.SpecialInfo.stGigEInfo
                self._device_info = {
                    "index": device_index,
                    "model": _decode_char_array(gige.chModelName),
                    "serial": _decode_char_array(gige.chSerialNumber),
                    "ip": _format_ip(int(gige.nCurrentIp)),
                    "packet_size": packet_size,
                }
            else:
                usb = st_dev.SpecialInfo.stUsb3VInfo
                self._device_info = {
                    "index": device_index,
                    "model": _decode_char_array(usb.chModelName),
                    "serial": _decode_char_array(usb.chSerialNumber),
                    "ip": "USB3",
                    "packet_size": 0,
                }

            return {
                "connected": True,
                "device": self._device_info,
                "exposure_us": exposure_us,
                "gain_db": gain_db,
                "software_trigger": software_trigger,
            }

    def _apply_params_unlocked(
        self,
        exposure_us: float,
        gain_db: float,
        software_trigger: bool,
    ) -> None:
        if not self._cam or not self._mv:
            return
        mv = self._mv
        cam = self._cam
        cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        cam.MV_CC_SetFloatValue("ExposureTime", float(max(50.0, exposure_us)))
        cam.MV_CC_SetEnumValue("GainAuto", 0)
        cam.MV_CC_SetFloatValue("Gain", float(max(0.0, gain_db)))

        self._software_trigger = bool(software_trigger)
        if self._software_trigger:
            cam.MV_CC_SetEnumValue("TriggerMode", mv.MV_TRIGGER_MODE_ON)
            cam.MV_CC_SetEnumValue("TriggerSource", mv.MV_TRIGGER_SOURCE_SOFTWARE)
        else:
            cam.MV_CC_SetEnumValue("TriggerMode", mv.MV_TRIGGER_MODE_OFF)

    def update_params(
        self,
        exposure_us: float = 15000.0,
        gain_db: float = 2.0,
        software_trigger: bool = True,
    ) -> Dict[str, Any]:
        with self._lock:
            if not self._connected or not self._cam:
                raise RuntimeError("Camera is not connected.")
            self._apply_params_unlocked(exposure_us, gain_db, software_trigger)
            return {
                "connected": True,
                "exposure_us": exposure_us,
                "gain_db": gain_db,
                "software_trigger": software_trigger,
            }

    def grab_frame_bgr(self, timeout_ms: int = 2500) -> np.ndarray:
        """Trigger (if in software trigger mode) and grab a single BGR8 numpy frame."""
        with self._lock:
            if not self._connected or not self._cam or not self._mv:
                raise RuntimeError("GigE Camera is not connected.")
            mv = self._mv
            cam = self._cam

            st_frame = mv.MV_FRAME_OUT()
            ret = -1
            max_tries = 3 if self._software_trigger else 1
            per_try_ms = max(500, int(timeout_ms) // max_tries) if max_tries > 1 else int(timeout_ms)

            for attempt in range(max_tries):
                # Ensure sensor readout settle gap between back-to-back software triggers
                if self._software_trigger:
                    elapsed = time.perf_counter() - self._last_grab_ts
                    if elapsed < 0.045:
                        time.sleep(0.045 - elapsed)
                    cam.MV_CC_SetCommandValue("TriggerSoftware")

                memset(byref(st_frame), 0, sizeof(st_frame))
                ret = cam.MV_CC_GetImageBuffer(st_frame, per_try_ms)
                self._last_grab_ts = time.perf_counter()
                if ret == 0 and st_frame.pBufAddr:
                    break
                if attempt + 1 < max_tries:
                    time.sleep(0.04)

            if ret != 0 or not st_frame.pBufAddr:
                raise RuntimeError(f"MV_CC_GetImageBuffer timeout/fail: 0x{ret & 0xFFFFFFFF:08X}")

            try:
                w = int(st_frame.stFrameInfo.nWidth)
                h = int(st_frame.stFrameInfo.nHeight)
                bgr_size = w * h * 3
                dst_buf = (c_ubyte * bgr_size)()

                st_cvt = mv.MV_CC_PIXEL_CONVERT_PARAM_EX()
                memset(byref(st_cvt), 0, sizeof(st_cvt))
                st_cvt.nWidth = w
                st_cvt.nHeight = h
                st_cvt.pSrcData = st_frame.pBufAddr
                st_cvt.nSrcDataLen = st_frame.stFrameInfo.nFrameLen
                st_cvt.enSrcPixelType = st_frame.stFrameInfo.enPixelType
                st_cvt.enDstPixelType = mv.PixelType_Gvsp_BGR8_Packed
                st_cvt.pDstBuffer = dst_buf
                st_cvt.nDstBufferSize = bgr_size

                ret_cvt = cam.MV_CC_ConvertPixelTypeEx(st_cvt)
                if ret_cvt != 0:
                    raise RuntimeError(f"MV_CC_ConvertPixelTypeEx failed: 0x{ret_cvt & 0xFFFFFFFF:08X}")

                arr = np.ctypeslib.as_array(dst_buf).reshape((h, w, 3)).copy()
                self._last_frame_bgr = arr
                return arr
            finally:
                cam.MV_CC_FreeImageBuffer(st_frame)

    def _disconnect_unlocked(self) -> None:
        if self._cam is not None:
            try:
                if self._grabbing:
                    self._cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                self._cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self._cam.MV_CC_DestroyHandle()
            except Exception:
                pass
        self._cam = None
        self._grabbing = False
        self._connected = False
        self._device_info = {}

    def disconnect(self) -> Dict[str, Any]:
        with self._lock:
            self._disconnect_unlocked()
            return {"connected": False}
