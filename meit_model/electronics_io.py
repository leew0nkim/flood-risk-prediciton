from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import serial


@dataclass
class SensorPacket:
    external_depth_m: float
    internal_depth_m: Optional[float]

    evacuation_complete: bool
    equalization_active: bool


class ElectronicsSerial:
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 1.0,
    ):
        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
        )

    @staticmethod
    def _require_bool(name: str, value):
        if not isinstance(value, bool):
            raise TypeError(
                f"{name} must be bool, "
                f"got {type(value).__name__}: {value!r}"
            )

        return value

    def read_sensor_packet(
        self,
    ) -> Optional[SensorPacket]:

        raw = self.serial.readline()

        if not raw:
            return None

        try:
            text = raw.decode("utf-8").strip()

            if not text:
                return None

            data = json.loads(text)

            external_depth_m = float(
                data["external_depth_m"]
            )

            internal_raw = data.get(
                "internal_depth_m"
            )

            internal_depth_m = (
                None
                if internal_raw is None
                else float(internal_raw)
            )

            evacuation_complete = (
                self._require_bool(
                    "evacuation_complete",
                    data["evacuation_complete"],
                )
            )

            equalization_active = (
                self._require_bool(
                    "equalization_active",
                    data["equalization_active"],
                )
            )

            if external_depth_m < 0:
                raise ValueError(
                    "external_depth_m cannot be negative"
                )

            if (
                internal_depth_m is not None
                and internal_depth_m < 0
            ):
                raise ValueError(
                    "internal_depth_m cannot be negative"
                )

            return SensorPacket(
                external_depth_m=external_depth_m,
                internal_depth_m=internal_depth_m,
                evacuation_complete=evacuation_complete,
                equalization_active=equalization_active,
            )

        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            print(
                f"[SERIAL INPUT ERROR] {exc}"
            )

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