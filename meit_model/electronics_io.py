from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

import serial


@dataclass
class SensorPacket:
    external_depth_m: float
    internal_depth_m: Optional[float]

    # Unknown states must never be silently converted to False.
    evacuation_complete: Optional[bool] = None
    equalization_active: Optional[bool] = None


class ElectronicsSerial:
    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 1.0,
    ):
        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
        )

    @staticmethod
    def parse_sensor_line(text: str) -> SensorPacket:
        """Board RX: one JSON object per line, depths already in metres."""
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        required = ("external_depth_m", "internal_depth_m",
                    "evacuation_complete", "equalization_active")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing sensor fields: {missing}")
        depths = {}
        for key in required[:2]:
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number in metres")
            try:
                value = float(value)
            except OverflowError as exc:
                raise ValueError(f"{key} is too large") from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
            depths[key] = value
        for key in required[2:]:
            if not isinstance(data[key], bool):
                raise ValueError(f"{key} must be a JSON boolean, not a string or number")
        return SensorPacket(**depths,
                            evacuation_complete=data["evacuation_complete"],
                            equalization_active=data["equalization_active"])

    def read_sensor_packet(self) -> Optional[SensorPacket]:
        raw = self.serial.readline()
        if not raw:
            return None
        # readline may return only part of a packet on timeout. Accumulate
        # until newline so a truncated numeric value cannot become a depth.
        pending = getattr(self, "_pending_rx", b"") + raw
        if len(pending) > 4096:
            self._pending_rx = b""
            self._discard_rx = not pending.endswith(b"\n")
            print("[SERIAL INPUT ERROR] Sensor line too long")
            return None
        if getattr(self, "_discard_rx", False):
            self._discard_rx = not pending.endswith(b"\n")
            self._pending_rx = b""
            return None
        if not pending.endswith(b"\n"):
            self._pending_rx = pending
            return None
        self._pending_rx = b""
        try:
            text = pending.decode("utf-8").strip()
            if not text:
                return None
            print("[RAW RX]", repr(text))
            return self.parse_sensor_line(text)
        except (UnicodeDecodeError, TypeError, ValueError) as exc:
            print(f"[SERIAL INPUT ERROR] {exc}")
            return None

    def send_ai_result(
        self,
        payload: dict,
    ):
        text = json.dumps(
            payload,
            ensure_ascii=False,
        )

        self.serial.write(
            (text + "\n").encode("utf-8")
        )

    def close(self):
        self.serial.close()