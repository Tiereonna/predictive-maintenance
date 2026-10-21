import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from etl_backblaze import clean_chunk, load_files


def test_bad_date_and_blank_serial_drop():
    frame = pd.DataFrame(
        {
            "date": ["2013-04-10", "not-a-date", "2013-04-11"],
            "serial_number": ["A1", "B2", " "],
            "model": ["ST", "ST", "ST"],
            "failure": [0, 0, 1],
            "smart_9_raw": [100, 200, 300],
        }
    )
    cleaned, report = clean_chunk(frame)
    assert report["rows_read"] == 3
    assert report["rows_rejected"] == 2
    assert list(cleaned["serial_number"]) == ["A1"]
    assert cleaned.iloc[0]["failure"] == 0


def test_chunked_folder_load(tmp_path: Path):
    one = tmp_path / "a.csv"
    two = tmp_path / "b.csv"
    one.write_text(
        "date,serial_number,model,failure,smart_5_raw\n"
        "2013-01-01,S1,M,0,1\n"
        "2013-01-01,,M,0,1\n"
    )
    two.write_text(
        "date,serial_number,model,failure,smart_5_raw\n"
        "2013-01-02,S2,M,1,4\n"
    )
    frame, report = load_files(tmp_path, chunksize=1)
    assert report["rows_read"] == 3
    assert report["rows_rejected"] == 1
    assert report["rows_accepted"] == 2
    assert set(frame["serial_number"]) == {"S1", "S2"}
