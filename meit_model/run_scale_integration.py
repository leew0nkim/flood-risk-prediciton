"""Run the current-water-level policy for the physical scale model."""
from __future__ import annotations

import argparse
import json
import sys
import time

from meit_model.electronics_io import ElectronicsSerial
from meit_model.scale_integration_model import ScaleIntegrationModel, StableScaleIntegrationModel


def _wire_payload(result, *, mock: bool) -> dict:
    """Small newline-delimited response for the Arduino receive buffer."""
    return {
        'state': result.state,
        'mock': mock,
        'command': result.command,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM3', help='COM3 or /dev/cu.usbmodem...')
    parser.add_argument('--baudrate', type=int, default=9600)
    parser.add_argument('--max-messages', type=int, default=0, help='0 runs until Ctrl+C')
    parser.add_argument('--full-wire-payload', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Evaluate one pair without opening a port')
    parser.add_argument('--external-depth-cm', type=float, default=2.0,
                        help='Dry-run external level in centimetres')
    parser.add_argument('--internal-depth-cm', type=float, default=2.0,
                        help='Dry-run internal level in centimetres')
    parser.add_argument('--equalization-active', action='store_true',
                        help='Dry-run only: board reports the opening is already active')
    parser.add_argument('--observe-only', action='store_true',
                        help='Use real sensors but force command=NONE for preflight')
    parser.add_argument('--warning-cm', type=float, default=2.0)
    parser.add_argument('--danger-cm', type=float, default=3.0)
    parser.add_argument('--open-cm', type=float, default=3.0)
    parser.add_argument('--warning-gap-cm', type=float, default=2.0)
    parser.add_argument('--open-gap-cm', type=float, default=3.0)
    parser.add_argument('--internal-warning-cm', type=float, default=2.5)
    parser.add_argument('--internal-limit-cm', type=float, default=3.0)
    args = parser.parse_args(argv)

    if args.baudrate <= 0 or args.max_messages < 0:
        parser.error('baudrate must be positive and max-messages nonnegative')
    if args.observe_only and args.dry_run:
        parser.error('--observe-only is only for real-sensor serial preflight')

    try:
        policy = ScaleIntegrationModel(
            warning_depth_m=args.warning_cm / 100,
            danger_depth_m=args.danger_cm / 100,
            open_depth_m=args.open_cm / 100,
            warning_depth_gap_m=args.warning_gap_cm / 100,
            open_depth_gap_m=args.open_gap_cm / 100,
            internal_warning_depth_m=args.internal_warning_cm / 100,
            internal_limit_depth_m=args.internal_limit_cm / 100,
        )
    except ValueError as exc:
        parser.error(str(exc))
    model = StableScaleIntegrationModel(policy)
    if args.dry_run:
        try:
            # Dry-run represents a stable level, so feed the same paired
            # measurement enough times to satisfy the actuation filter.
            for _ in range(model.minimum_samples):
                result = model.update(
                    args.external_depth_cm / 100,
                    args.internal_depth_cm / 100,
                    equalization_active=args.equalization_active,
                )
        except ValueError as exc:
            parser.error(str(exc))
        payload = result.to_dict() if args.full_wire_payload else _wire_payload(result, mock=True)
        if args.full_wire_payload:
            payload['mock'] = True
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
        return 0

    transport = None
    sent = 0
    reported_error = None
    try:
        transport = ElectronicsSerial(args.port, args.baudrate, timeout=0.1)
        print(
            f'[POLICY] warning={args.warning_cm:g}cm danger={args.danger_cm:g}cm '
            f'open={args.open_cm:g}cm warning_gap={args.warning_gap_cm:g}cm '
            f'danger/open_gap={args.open_gap_cm:g}cm '
            f'internal_limit={args.internal_limit_cm:g}cm',
            file=sys.stderr,
            flush=True,
        )
        while not args.max_messages or sent < args.max_messages:
            packet = transport.read_sensor_packet()
            if transport.last_line is not None:
                tag = 'RX OK' if packet is not None else (
                    'RX ERROR' if transport.last_error else 'RX BOARD')
                print(f'[{tag}] {transport.last_line!r}', file=sys.stderr, flush=True)
            if transport.last_error and (
                    transport.last_line is not None or transport.last_error != reported_error):
                print(f'[RX ERROR] {transport.last_error}', file=sys.stderr, flush=True)
            reported_error = transport.last_error
            if packet is None:
                time.sleep(0.01)
                continue

            result = model.update(
                packet.external_depth_m,
                packet.internal_depth_m,
                equalization_active=packet.equalization_active,
            )
            print(
                f'[LIVE] raw_external={packet.external_depth_m:.3f}m '
                f'raw_internal={packet.internal_depth_m:.3f}m '
                f'filtered_external={result.external_depth_m:.3f}m '
                f'filtered_internal={result.internal_depth_m:.3f}m '
                f'gap={result.depth_gap_m:.3f}m samples={result.filter_sample_count} '
                f'ready={str(result.filter_ready).lower()}',
                file=sys.stderr,
                flush=True,
            )
            diagnostic = result.to_dict()
            diagnostic['mock'] = False
            wire = diagnostic if args.full_wire_payload else _wire_payload(
                result, mock=False)
            if args.observe_only:
                wire['command'] = 'NONE'
                print(
                    f'[OBSERVE ONLY] candidate_command={result.command}; transmitted=NONE',
                    file=sys.stderr,
                    flush=True,
                )
            transport.send_ai_result(wire)
            print('[TX] ' + json.dumps(wire, ensure_ascii=False, allow_nan=False,
                                       separators=(',', ':')), flush=True)
            if args.full_wire_payload:
                print('[RESULT] ' + json.dumps(diagnostic, ensure_ascii=False, allow_nan=False,
                                               separators=(',', ':')), file=sys.stderr, flush=True)
            sent += 1
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'Scale integration stopped: {exc}', file=sys.stderr)
        return 1
    finally:
        if transport:
            transport.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
