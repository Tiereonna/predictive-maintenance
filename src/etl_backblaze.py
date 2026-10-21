# load backblaze hard drive csvs in chunks
# real public data: https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data
# failure is rare, dates are messy, columns change by year

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

KEEP_ALWAYS = ["date", "serial_number", "model", "failure"]

# these show up in most yearly files. missing ones just get skipped.
SMART_CANDIDATES = [
    "smart_5_raw",   # reallocated sectors
    "smart_9_raw",   # power-on hours
    "smart_12_raw",  # power cycles
    "smart_187_raw", # uncorrectable errors
    "smart_197_raw", # current pending sector
    "smart_198_raw", # offline uncorrectable
]


def list_csvs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.csv") if "__MACOSX" not in p.parts)


def clean_chunk(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"rows_read": int(len(frame)), "rows_rejected": 0}
    missing = [c for c in KEEP_ALWAYS if c not in frame.columns]
    if missing:
        raise ValueError(f"csv missing columns: {missing}")

    smart = [c for c in SMART_CANDIDATES if c in frame.columns]
    cols = KEEP_ALWAYS + smart
    out = frame[cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["serial_number"] = out["serial_number"].astype(str).str.strip()
    out["model"] = out["model"].astype(str).str.strip()
    out["failure"] = pd.to_numeric(out["failure"], errors="coerce")

    bad = (
        out["date"].isna()
        | out["serial_number"].eq("")
        | out["serial_number"].eq("nan")
        | out["failure"].isna()
        | ~out["failure"].isin([0, 1])
    )
    report["rows_rejected"] = int(bad.sum())
    out = out.loc[~bad].copy()
    out["failure"] = out["failure"].astype(int)
    for col in smart:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    report["rows_accepted"] = int(len(out))
    return out, report


def merge_reports(total: dict, piece: dict) -> dict:
    for key in ("rows_read", "rows_rejected", "rows_accepted"):
        total[key] = int(total.get(key, 0) + piece[key])
    return total


def load_files(path: Path, chunksize: int = 50_000):
    total = {"rows_read": 0, "rows_rejected": 0, "rows_accepted": 0}
    parts: list[pd.DataFrame] = []
    started = time.perf_counter()
    for csv_path in list_csvs(path):
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
            cleaned, piece = clean_chunk(chunk)
            merge_reports(total, piece)
            if not cleaned.empty:
                parts.append(cleaned)
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    total["seconds"] = round(time.perf_counter() - started, 2)
    total["files"] = len(list_csvs(path))
    return frame, total


STAGING_DDL = """
CREATE TABLE IF NOT EXISTS drive_daily (
    date date NOT NULL,
    serial_number text NOT NULL,
    model text NOT NULL,
    failure integer NOT NULL,
    smart_5_raw double precision,
    smart_9_raw double precision,
    smart_12_raw double precision,
    smart_187_raw double precision,
    smart_197_raw double precision,
    smart_198_raw double precision
)
"""

UNIQUE_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS drive_daily_key
ON drive_daily (date, serial_number)
"""

INSERT_SQL = """
INSERT INTO drive_daily (
    date, serial_number, model, failure,
    smart_5_raw, smart_9_raw, smart_12_raw,
    smart_187_raw, smart_197_raw, smart_198_raw
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (date, serial_number) DO UPDATE SET
    model = EXCLUDED.model,
    failure = EXCLUDED.failure,
    smart_5_raw = EXCLUDED.smart_5_raw,
    smart_9_raw = EXCLUDED.smart_9_raw,
    smart_12_raw = EXCLUDED.smart_12_raw,
    smart_187_raw = EXCLUDED.smart_187_raw,
    smart_197_raw = EXCLUDED.smart_197_raw,
    smart_198_raw = EXCLUDED.smart_198_raw
"""


def _tuple(row: dict) -> tuple:
    def num(name: str):
        value = row.get(name)
        if value is None or pd.isna(value):
            return None
        return float(value)

    return (
        row["date"].date() if hasattr(row["date"], "date") else row["date"],
        str(row["serial_number"]),
        str(row["model"]),
        int(row["failure"]),
        num("smart_5_raw"),
        num("smart_9_raw"),
        num("smart_12_raw"),
        num("smart_187_raw"),
        num("smart_197_raw"),
        num("smart_198_raw"),
    )


def load_postgres(frame: pd.DataFrame, database_url: str, batch_size: int = 5000) -> int:
    import psycopg

    if frame.empty:
        return 0
    rows = [_tuple(r) for r in frame.to_dict(orient="records")]
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(STAGING_DDL)
            cursor.execute(UNIQUE_SQL)
            for i in range(0, len(rows), batch_size):
                cursor.executemany(INSERT_SQL, rows[i : i + batch_size])
        connection.commit()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True, help="csv file or folder of csvs")
    parser.add_argument("--report", type=Path, default=Path("artifacts/backblaze_report.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/backblaze_clean.csv"))
    args = parser.parse_args()
    frame, report = load_files(args.path)
    report["loaded_rows"] = 0
    url = os.environ.get("DATABASE_URL")
    if url:
        report["loaded_rows"] = load_postgres(frame, url)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not frame.empty:
        frame.to_csv(args.out, index=False)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
