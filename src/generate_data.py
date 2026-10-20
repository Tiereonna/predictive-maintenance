# fake sensor data (not from a real machine)
# there's a planted failure rule so i can check if the model actually learns it

from __future__ import annotations

import numpy as np
import pandas as pd

# fixed seed so the data is the same every run
RNG_SEED = 20260909


def generate_sensor_frame(n_rows: int = 4000, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hours_run = rng.uniform(50, 8000, size=n_rows)
    vibration_mm_s = rng.normal(2.4, 0.9, size=n_rows)
    temperature_c = rng.normal(68, 12, size=n_rows)
    pressure_psi = rng.normal(120, 18, size=n_rows)
    current_amps = rng.normal(14, 3.2, size=n_rows)

    # Planted risk: old + hot + shaky machines fail more often.
    # The logit is the "true" story. Models only see the columns above.
    logit = (
        -3.4
        + 0.00055 * hours_run
        + 0.85 * np.clip(vibration_mm_s - 2.0, 0, None)
        + 0.10 * np.clip(temperature_c - 70, 0, None)
        + 0.08 * np.clip(current_amps - 15, 0, None)
        - 0.012 * np.abs(pressure_psi - 120)
    )
    fail_prob = 1 / (1 + np.exp(-logit))
    failed_within_7_days = rng.random(n_rows) < fail_prob

    return pd.DataFrame(
        {
            "hours_run": hours_run.round(1),
            "vibration_mm_s": vibration_mm_s.round(3),
            "temperature_c": temperature_c.round(2),
            "pressure_psi": pressure_psi.round(2),
            "current_amps": current_amps.round(2),
            "failed_within_7_days": failed_within_7_days.astype(int),
        }
    )
