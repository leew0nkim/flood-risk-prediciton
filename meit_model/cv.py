from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from meit_data.pipeline import sha256
from meit_data.torch_data import depth_to_m
from meit_model.fold_scaling import data_fingerprint


# ============================================================
# CV 기본 설정
# ============================================================

# 최근 30분을 입력으로 사용
HISTORY_MINUTES = 30

# 미래 15분을 예측
HORIZON_MINUTES = 15

# 팀에서 새로 정한 위험 임계수심
DEFAULT_CRITICAL_DEPTH_M = 0.30
CV_SCHEMA_VERSION = 3


@dataclass
class DevelopmentData:
    X: np.ndarray
    Y: np.ndarray
    Y_mask: np.ndarray

    origin_time: np.ndarray
    event_index: np.ndarray

    train_size: int


@dataclass
class Fold:
    fold_number: int

    train_indices: np.ndarray
    val_indices: np.ndarray

    train_event_ids: np.ndarray
    val_event_ids: np.ndarray

    val_start: int
    val_end: int


# ============================================================
# 데이터 불러오기
# ============================================================

def load_development_data(
    data_dir: Path,
) -> DevelopmentData:

    train_path = data_dir / "train.npz"
    val_path = data_dir / "validation.npz"

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not val_path.exists():
        raise FileNotFoundError(val_path)

    train = np.load(
        train_path,
        allow_pickle=False,
    )

    validation = np.load(
        val_path,
        allow_pickle=False,
    )

    # --------------------------------------------------------
    # Event ID가 train/validation에서 겹치는지 먼저 검사
    # --------------------------------------------------------

    train_events = np.unique(
        train["event_index"]
    )

    val_events = np.unique(
        validation["event_index"]
    )

    overlap = np.intersect1d(
        train_events,
        val_events,
    )

    if len(overlap) > 0:
        raise RuntimeError(
            f"Existing train/validation event overlap detected: "
            f"{len(overlap)} events"
        )

    # --------------------------------------------------------
    # CV에서는 기존 train + validation을 development set으로 사용
    #
    # 순서는 반드시 train → validation
    # --------------------------------------------------------

    X = np.concatenate(
        [
            train["X"],
            validation["X"],
        ],
        axis=0,
    )

    Y = np.concatenate(
        [
            train["Y"],
            validation["Y"],
        ],
        axis=0,
    )

    Y_mask = np.concatenate(
        [
            train["Y_mask"],
            validation["Y_mask"],
        ],
        axis=0,
    )

    origin_time = np.concatenate(
        [
            train["origin_time_unix_seconds"],
            validation["origin_time_unix_seconds"],
        ],
        axis=0,
    )

    event_index = np.concatenate(
        [
            train["event_index"],
            validation["event_index"],
        ],
        axis=0,
    )

    return DevelopmentData(
        X=X,
        Y=Y,
        Y_mask=Y_mask,
        origin_time=origin_time,
        event_index=event_index,
        train_size=len(train["X"]),
    )


# ============================================================
# 각 event가 실제로 사용하는 전체 시간 범위 계산
# ============================================================

def build_event_table(
    data: DevelopmentData,
):

    origins = data.origin_time
    events = data.event_index

    # --------------------------------------------------------
    # 한 sample의 정보 범위
    #
    # origin - 30분
    #        ~
    # origin + 15분
    #
    # 총 약 45분
    # --------------------------------------------------------

    information_start = (
        origins
        - HISTORY_MINUTES * 60
    )

    information_end = (
        origins
        + HORIZON_MINUTES * 60
    )

    event_ids, inverse = np.unique(
        events,
        return_inverse=True,
    )

    n_events = len(event_ids)

    event_start = np.full(
        n_events,
        np.iinfo(np.int64).max,
        dtype=np.int64,
    )

    event_end = np.full(
        n_events,
        np.iinfo(np.int64).min,
        dtype=np.int64,
    )

    np.minimum.at(
        event_start,
        inverse,
        information_start,
    )

    np.maximum.at(
        event_end,
        inverse,
        information_end,
    )

    event_sample_count = np.bincount(
        inverse
    )

    # 시간순으로 정렬
    order = np.argsort(
        event_start,
        kind="stable",
    )

    return (
        event_ids[order],
        event_start[order],
        event_end[order],
        event_sample_count[order],
    )


# ============================================================
# Purged Rolling-Origin Fold 생성
# ============================================================

def make_folds(
    data: DevelopmentData,
    n_splits: int = 4,
    initial_train_fraction: float = 0.50,
):

    (
        event_ids,
        event_start,
        event_end,
        event_sample_count,
    ) = build_event_table(data)

    total_events = len(event_ids)

    initial_train_events = int(
        total_events
        * initial_train_fraction
    )

    if initial_train_events <= 0:
        raise ValueError(
            "Initial train period is empty."
        )

    candidate_indices = np.arange(
        initial_train_events,
        total_events,
    )

    if len(candidate_indices) < n_splits:
        raise ValueError(
            "Not enough events for requested CV folds."
        )

    # --------------------------------------------------------
    # 미래 event들을 시간 순서 그대로
    # validation block으로 나눈다.
    #
    # random shuffle 절대 X
    # --------------------------------------------------------

    validation_blocks = np.array_split(
        candidate_indices,
        n_splits,
    )

    folds = []

    for fold_number, block in enumerate(
        validation_blocks,
        start=1,
    ):

        val_event_ids = event_ids[block]

        # 이 Validation fold가 사용하는 가장 이른 정보 시점
        val_information_start = int(
            event_start[block].min()
        )

        val_information_end = int(
            event_end[block].max()
        )

        # ----------------------------------------------------
        # PURGE
        #
        # Validation 정보 시작 이전에
        # 완전히 끝난 event만 Train에 사용
        #
        # event_end < val_information_start
        #
        # 따라서 train target과 validation history가
        # 겹치는 event는 통째로 제외됨
        # ----------------------------------------------------

        train_event_mask = (
            event_end
            < val_information_start
        )

        train_event_ids = event_ids[
            train_event_mask
        ]

        # ----------------------------------------------------
        # Sample index 생성
        # ----------------------------------------------------

        train_indices = np.flatnonzero(
            np.isin(
                data.event_index,
                train_event_ids,
            )
        )

        val_indices = np.flatnonzero(
            np.isin(
                data.event_index,
                val_event_ids,
            )
        )

        if len(train_indices) == 0:
            raise RuntimeError(
                f"Fold {fold_number}: empty train set"
            )

        if len(val_indices) == 0:
            raise RuntimeError(
                f"Fold {fold_number}: empty validation set"
            )

        # ----------------------------------------------------
        # EVENT-AWARE 검증
        # ----------------------------------------------------

        event_overlap = np.intersect1d(
            train_event_ids,
            val_event_ids,
        )

        if len(event_overlap) != 0:
            raise AssertionError(
                f"Fold {fold_number}: "
                f"event leakage detected"
            )

        # ----------------------------------------------------
        # PURGE 검증
        # ----------------------------------------------------

        train_information_end = (
            data.origin_time[train_indices]
            + HORIZON_MINUTES * 60
        )

        val_information_start_samples = (
            data.origin_time[val_indices]
            - HISTORY_MINUTES * 60
        )

        latest_train_information = int(
            train_information_end.max()
        )

        earliest_val_information = int(
            val_information_start_samples.min()
        )

        if (
            latest_train_information
            >= earliest_val_information
        ):
            raise AssertionError(
                f"Fold {fold_number}: "
                f"time leakage detected"
            )

        fold = Fold(
            fold_number=fold_number,
            train_indices=train_indices,
            val_indices=val_indices,
            train_event_ids=train_event_ids,
            val_event_ids=val_event_ids,
            val_start=val_information_start,
            val_end=val_information_end,
        )

        folds.append(fold)

    return folds


# ============================================================
# Risk-aware Fold 진단
# ============================================================

def load_scaler(
    data_dir: Path,
):

    scaler_path = (
        data_dir
        / "scaler.json"
    )

    with open(
        scaler_path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def calculate_risk_statistics(
    data: DevelopmentData,
    indices: np.ndarray,
    scaler,
    critical_depth_m: float,
):

    # --------------------------------------------------------
    # 현재 외부수위
    # --------------------------------------------------------

    current_standardized = (
        data.X[indices, -1:, :]
    )

    current_m = depth_to_m(
        torch.from_numpy(
            current_standardized
        ).float(),
        scaler,
        clip_negative=True,
    )

    current_m = (
        current_m
        .cpu()
        .numpy()[:, 0, 0]
    )

    # --------------------------------------------------------
    # 미래 1~15분
    # --------------------------------------------------------

    future_standardized = (
        data.Y[indices]
    )

    future_m = depth_to_m(
        torch.from_numpy(
            future_standardized
        ).float(),
        scaler,
        clip_negative=True,
    )

    future_m = (
        future_m
        .cpu()
        .numpy()[:, :, 0]
    )

    mask = (
        data.Y_mask[indices]
        [:, :, 0]
    )

    # 결측 target은 위험 판정에서 제외
    valid_future = np.where(
        mask,
        future_m,
        -np.inf,
    )

    future_max = np.max(
        valid_future,
        axis=1,
    )

    # 현재 이미 critical
    critical_now = (
        current_m
        >= critical_depth_m
    )

    # 현재는 critical 미만인데
    # 향후 15분 이내 critical 도달
    crossing = (
        (current_m < critical_depth_m)
        &
        (future_max >= critical_depth_m)
    )

    # 현재 또는 미래 중 한 번이라도 위험
    any_critical = (
        critical_now
        |
        (future_max >= critical_depth_m)
    )

    return {
        "samples": int(
            len(indices)
        ),

        "critical_now": int(
            critical_now.sum()
        ),

        "critical_crossing_15m": int(
            crossing.sum()
        ),

        "any_critical": int(
            any_critical.sum()
        ),
    }


# ============================================================
# 출력 helper
# ============================================================

def utc_string(
    unix_seconds: int,
):

    return datetime.fromtimestamp(
        unix_seconds,
        tz=timezone.utc,
    ).isoformat()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "data/processed"
        ),
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--initial-train-fraction",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--critical-depth",
        type=float,
        default=DEFAULT_CRITICAL_DEPTH_M,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/cv"
        ),
    )

    args = parser.parse_args()

    if not np.isclose(args.critical_depth, DEFAULT_CRITICAL_DEPTH_M, rtol=0., atol=1e-12):
        parser.error("The canonical external crossing criterion is fixed at 0.30 m")

    # --------------------------------------------------------
    # 1. Development data
    # --------------------------------------------------------

    data = load_development_data(
        args.data_dir
    )

    scaler = load_scaler(
        args.data_dir
    )
    fingerprint = data_fingerprint(args.data_dir)

    print(
        "Development samples:",
        len(data.X),
    )

    print(
        "Critical depth:",
        args.critical_depth,
        "m",
    )

    # --------------------------------------------------------
    # 2. Fold 생성
    # --------------------------------------------------------

    folds = make_folds(
        data=data,
        n_splits=args.n_splits,
        initial_train_fraction=(
            args.initial_train_fraction
        ),
    )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("CV output directory is not empty; choose a new directory")
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    # --------------------------------------------------------
    # 3. Fold 검사 + 저장
    # --------------------------------------------------------

    for fold in folds:

        risk_stats = (
            calculate_risk_statistics(
                data=data,
                indices=fold.val_indices,
                scaler=scaler,
                critical_depth_m=(
                    args.critical_depth
                ),
            )
        )

        train_origin = (
            data.origin_time[
                fold.train_indices
            ]
        )

        val_origin = (
            data.origin_time[
                fold.val_indices
            ]
        )

        summary = {

            "fold": fold.fold_number,

            "train_samples": int(
                len(
                    fold.train_indices
                )
            ),

            "validation_samples": int(
                len(
                    fold.val_indices
                )
            ),

            "train_events": int(
                len(
                    fold.train_event_ids
                )
            ),

            "validation_events": int(
                len(
                    fold.val_event_ids
                )
            ),

            "train_origin_start_utc":
                utc_string(
                    int(
                        train_origin.min()
                    )
                ),

            "train_origin_end_utc":
                utc_string(
                    int(
                        train_origin.max()
                    )
                ),

            "validation_origin_start_utc":
                utc_string(
                    int(
                        val_origin.min()
                    )
                ),

            "validation_origin_end_utc":
                utc_string(
                    int(
                        val_origin.max()
                    )
                ),

            "risk": risk_stats,
        }

        summaries.append(
            summary
        )

        # 이 indices는 나중에
        # ConcatDataset([train, validation])
        # 에 그대로 사용 가능
        fold_output = args.output_dir / f"fold_{fold.fold_number}.npz"
        np.savez_compressed(
            fold_output,

            train_indices=(
                fold.train_indices
            ),

            val_indices=(
                fold.val_indices
            ),
        )
        summary["artifact"] = {
            "path": fold_output.name,
            "sha256": sha256(fold_output),
        }

        print(
            "\n"
            f"===== FOLD "
            f"{fold.fold_number} ====="
        )

        print(
            "Train:",
            len(
                fold.train_indices
            ),
            "samples /",
            len(
                fold.train_event_ids
            ),
            "events",
        )

        print(
            "Validation:",
            len(
                fold.val_indices
            ),
            "samples /",
            len(
                fold.val_event_ids
            ),
            "events",
        )

        print(
            "Validation risk:",
            risk_stats,
        )

        print(
            "Event overlap: 0"
        )

        print(
            "Information overlap: 0"
        )

    # --------------------------------------------------------
    # Summary 저장
    # --------------------------------------------------------

    with open(
        args.output_dir
        / "cv_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "schema_version": CV_SCHEMA_VERSION,
                "data_fingerprint": fingerprint,
                "evaluation_scope": "development_only",
                "sealed_test_used": False,
                "critical_threshold_role": "external_depth_crossing",
                "n_splits": args.n_splits,
                "initial_train_fraction": args.initial_train_fraction,
                "critical_depth_m": args.critical_depth,
                "folds": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n"
        "================================"
    )

    print(
        "CV FOLD GENERATION PASSED"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()
