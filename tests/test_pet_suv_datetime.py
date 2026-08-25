from datetime import datetime

import pytest

from zrad.io.pet_suv import parse_time


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("20250101110000+0100", datetime(2025, 1, 1, 11, 0)),
        (
            "20250101111459.998530+0100",
            datetime(2025, 1, 1, 11, 14, 59, 998530),
        ),
    ],
)
def test_parse_time_accepts_dicom_datetime_utc_offsets(value, expected):
    assert parse_time(value) == expected
