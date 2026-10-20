"""Checks the synthetic sensor table before it is used for training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_data import generate_sensor_frame

FEATURES = [
    "hours_run",
    "vibration_mm_s",
    "temperature_c",
    "pressure_psi",
    "current_amps",
]


def test_frame_has_4000_rows_and_the_expected_columns():
    frame = generate_sensor_frame()
    assert len(frame) == 4000
    assert list(frame.columns) == FEATURES + ["failed_within_7_days"]
    assert frame[FEATURES].isna().sum().sum() == 0
    assert set(frame["failed_within_7_days"].unique()) <= {0, 1}


def test_same_seed_produces_the_same_table():
    first = generate_sensor_frame()
    second = generate_sensor_frame()
    assert first.equals(second)
