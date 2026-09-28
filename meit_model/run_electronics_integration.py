"""Live LSTM and model-free electronics tests. All commands are NONE."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from meit_model.electronics_io import ElectronicsSerial
from meit_model.integration_test_modes import IntegrationSession


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('live', 'quick', 'accumulate'), default='live')
    parser.add_argument('--port', default='COM3', help='Serial port, e.g. COM3 or /dev/cu.usbmodem...')
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/risk_lstm_cv_v2/fold_1/best_model.pt'))
    parser.add_argument('--baudrate', type=int, default=9600)
    parser.add_argument('--scenario', choices=('cycle', 'safe', 'warning', 'danger', 'equalized', 'inside-high'), default='cycle')
    parser.add_argument('--scenario-seconds', type=float, default=5)
    parser.add_argument('--config', type=Path, default=Path('configs/pipeline_config.json'))
    parser.add_argument('--dry-run', action='store_true', help='Quick mode only: print JSON without opening a port')
    parser.add_argument('--reply-on-input', action='store_true',
                        help='Quick mode only: log received lines and reply with compact mock JSON for each valid packet')
    parser.add_argument('--full-wire-payload', action='store_true',
                        help='Send the full diagnostic JSON to the board (requires a sufficiently large receiver)')
    parser.add_argument('--max-messages', type=int, default=0, help='0 runs until Ctrl+C')
    args = parser.parse_args(argv)
    if args.dry_run and args.mode != 'quick':
        parser.error('--dry-run is only available with --mode quick')
    if args.reply_on_input and (args.mode != 'quick' or args.dry_run):
        parser.error('--reply-on-input requires --mode quick and a serial port (no --dry-run)')
    if not args.dry_run and not args.port:
        parser.error('--port is required unless --dry-run is set')
    if not math.isfinite(args.scenario_seconds) or args.scenario_seconds <= 0:
        parser.error('--scenario-seconds must be positive and finite')
    if args.max_messages < 0 or args.baudrate <= 0:
        parser.error('invalid message count or baudrate')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    predictor = None
    if args.mode == 'live':
        from meit_model.predictor import ForecastPredictor
        predictor = ForecastPredictor(args.checkpoint)
    # Monotonic elapsed time avoids wall-clock corrections resetting the grid.
    epoch, tick = time.time(), time.monotonic()
    session = IntegrationSession(args.mode, config, epoch, args.scenario, args.scenario_seconds, predictor=predictor)
    transport = None
    try:
        if not args.dry_run:
            transport = ElectronicsSerial(args.port, args.baudrate, timeout=0.1)
        sent, next_send = 0, time.monotonic()
        reported_error = None
        while not args.max_messages or sent < args.max_messages:
            packet = transport.read_sensor_packet() if transport else None
            if args.reply_on_input:
                if transport.last_line is not None:
                    tag = 'RX OK' if packet is not None else (
                        'RX ERROR' if transport.last_error else 'RX BOARD')
                    print(f'[{tag}] {transport.last_line!r}', file=sys.stderr, flush=True)
                if transport.last_error and (transport.last_line is not None or transport.last_error != reported_error):
                    print(f'[RX ERROR] {transport.last_error}', file=sys.stderr, flush=True)
                reported_error = transport.last_error
                # One request -> one response. No unsolicited reply, and no
                # success response to malformed or partial sensor input.
                if packet is None:
                    time.sleep(0.01)
                    continue
            payload = session.update(packet, epoch + time.monotonic() - tick)
            if args.reply_on_input or time.monotonic() >= next_send:
                if not args.reply_on_input:
                    payload.update(transport.input_status() if transport else {
                        'serial_input_event': None, 'board_ready': False,
                        'board_sensor_status': 'UNKNOWN', 'board_reported_state': None,
                        'board_reported_led': None, 'board_echo_status': None,
                        'serial_input_error': None,
                    })
                send_interval = 1.0
                wire_payload = payload if args.full_wire_payload else {
                    # Keep the default Arduino message below its small line buffer.
                    # Detailed diagnostics remain in the host console payload.
                    'state': payload['state'],
                    'mock': payload['prediction_source'] == 'mock',
                    'command': payload['command'],
                }
                if transport:
                    sent_bytes = transport.send_ai_result(wire_payload)
                    # 8N1 uses ten wire bits per byte. New pressure fields can
                    # exceed 9600 baud's one-second capacity; avoid a backlog.
                    send_interval = max(1.0, sent_bytes * 10 / args.baudrate * 1.1)
                rendered = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
                wire_rendered = json.dumps(wire_payload, ensure_ascii=False, allow_nan=False,
                                           separators=(',', ':'))
                print(f'[TX MOCK] {wire_rendered}' if args.reply_on_input else rendered, flush=True)
                sent += 1
                next_send = time.monotonic() + send_interval
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'Integration stopped: {exc}', file=sys.stderr)
        return 1
    finally:
        if transport:
            transport.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
