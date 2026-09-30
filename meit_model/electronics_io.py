"""Newline-delimited JSON transport for electronics bench tests."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class SensorPacket:
    external_depth_m: float
    internal_depth_m: Optional[float]
    evacuation_complete: bool
    equalization_active: bool


class ElectronicsSerial:
    MAX_LINE_BYTES = 4096

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0,
                 *, connection=None):
        if connection is None:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError('Install electronics dependencies: pip install -e ".[electronics]"') from exc
            connection = serial.Serial(port=port, baudrate=baudrate, timeout=timeout,
                                       write_timeout=max(1.0, self.MAX_LINE_BYTES * 10 / baudrate + 1))
        self.serial = connection
        self.pending = bytearray()
        self.discarding = False
        self.last_error = None
        self.last_line = None
        self.last_event = None
        self.board_ready = False
        self.board_sensor_status = 'UNKNOWN'
        self.board_reported_state = None
        self.board_reported_led = None
        self.board_echo_status = None

    def _ensure_input_state(self):
        """Initialize state for tests that construct this class with __new__."""
        defaults = {
            'pending': bytearray(), 'discarding': False, 'last_error': None,
            'last_line': None, 'last_event': None, 'board_ready': False,
            'board_sensor_status': 'UNKNOWN', 'board_reported_state': None,
            'board_reported_led': None, 'board_echo_status': None,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def _parse_board_message(self, text: str) -> bool:
        """Interpret known Arduino diagnostic/acknowledgement lines.

        Returns True when the line belongs to the board protocol rather than
        being a sensor JSON packet. Known board messages are structured events,
        not JSON parse errors.
        """
        if text == 'Arduino Ready':
            self.board_ready = True
            self.last_event = {'type': 'BOARD_READY'}
        elif text == 'SENSOR ERROR' or (len(text) >= 5 and 'SENSOR ERROR'.endswith(text)):
            self.board_sensor_status = 'ERROR'
            self.last_event = {'type': 'SENSOR_ERROR'}
        elif text.startswith('AI DATA:'):
            echoed = text.removeprefix('AI DATA:').strip()
            try:
                decoded = json.loads(echoed)
                complete = isinstance(decoded, dict)
            except (json.JSONDecodeError, TypeError):
                complete = False
            self.board_echo_status = 'COMPLETE' if complete else 'TRUNCATED'
            self.last_event = {'type': 'AI_ECHO', 'status': self.board_echo_status}
        elif text.startswith('STATE ='):
            state = text.partition('=')[2].strip()
            if state not in {'SAFE', 'WARNING', 'DANGER', 'UNKNOWN'}:
                return False
            self.board_reported_state = state
            self.last_event = {'type': 'BOARD_STATE', 'value': state}
        elif text.startswith('LED ='):
            led = text.partition('=')[2].strip()
            if led not in {'BLUE', 'GREEN', 'YELLOW', 'RED', 'OFF', 'UNKNOWN'}:
                return False
            self.board_reported_led = led
            self.last_event = {'type': 'BOARD_LED', 'value': led}
        else:
            return False
        self.last_error = None
        return True

    @staticmethod
    def _require_bool(name, value):
        if not isinstance(value, bool):
            raise ValueError(f'{name} must be bool')
        return value

    @staticmethod
    def _depth(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a JSON number')
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError(f'{name} is out of range') from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
        return value

    @staticmethod
    def parse_sensor_line(text: str) -> SensorPacket:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('Sensor packet must be a JSON object')
        try:
            return SensorPacket(
                external_depth_m=ElectronicsSerial._depth('external_depth_m', data['external_depth_m']),
                internal_depth_m=ElectronicsSerial._depth('internal_depth_m', data['internal_depth_m']),
                evacuation_complete=ElectronicsSerial._require_bool('evacuation_complete', data['evacuation_complete']),
                equalization_active=ElectronicsSerial._require_bool('equalization_active', data['equalization_active']),
            )
        except KeyError as exc:
            raise ValueError(f'Missing required sensor field: {exc.args[0]}') from exc

    def read_sensor_packet(self) -> Optional[SensorPacket]:
        self._ensure_input_state()
        self.last_line = None  # Only a complete line from this read, never a stale echo.
        # Keep last_event across read timeouts so status snapshots describe the
        # most recent actual input instead of an empty polling iteration.
        # readline can return an incomplete line when its timeout expires.
        raw = self.serial.readline(self.MAX_LINE_BYTES + 1)
        if not raw:
            return None
        if self.discarding:
            self.discarding = not raw.endswith(b'\n')
            return None
        self.pending.extend(raw)
        if len(self.pending) > self.MAX_LINE_BYTES:
            self.pending.clear()
            self.discarding = not raw.endswith(b'\n')
            self.last_error = 'Sensor line exceeds 4096 bytes'
            self.last_event = {'type': 'INPUT_TOO_LONG'}
            return None
        if not self.pending.endswith(b'\n'):
            return None
        line = bytes(self.pending)
        self.pending.clear()
        self.last_line = line.decode('utf-8', errors='backslashreplace').rstrip('\r\n')
        if self._parse_board_message(self.last_line):
            return None
        try:
            packet = self.parse_sensor_line(line.decode('utf-8'))
        except (UnicodeDecodeError, ValueError, TypeError, KeyError, OverflowError) as exc:
            if self.last_line.lstrip().startswith(('{', '[', 'null')):
                self.last_error = f'Invalid sensor JSON: {exc}'
                event_type = 'INVALID_SENSOR_JSON'
            else:
                self.last_error = f'Unrecognized board input: {self.last_line!r}'
                event_type = 'UNRECOGNIZED_TEXT'
            self.last_event = {'type': event_type}
            return None
        self.last_error = None
        self.board_sensor_status = 'OK'
        self.last_event = {
            'type': 'SENSOR_DATA',
            'external_depth_m': packet.external_depth_m,
            'internal_depth_m': packet.internal_depth_m,
        }
        return packet

    def input_status(self) -> dict:
        self._ensure_input_state()
        return {
            'serial_input_event': self.last_event,
            'board_ready': self.board_ready,
            'board_sensor_status': self.board_sensor_status,
            'board_reported_state': self.board_reported_state,
            'board_reported_led': self.board_reported_led,
            'board_echo_status': self.board_echo_status,
            'serial_input_error': self.last_error,
        }

    def send_ai_result(self, payload: dict):
        wire = (json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
        if self.serial.write(wire) != len(wire):
            raise OSError('Incomplete serial write')
        return len(wire)

    def close(self):
        self.serial.close()
