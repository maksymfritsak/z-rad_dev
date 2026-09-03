from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from zrad.image import Image
from zrad.io.pet_suv import parse_time

VALID_IBSI_SUV_DROS = (
    "DRO_0_0",
    "DRO_1_0",
    "DRO_2_0",
    "DRO_2_1_0",
    "DRO_2_1_1",
    "DRO_2_1_2",
    "DRO_2_2_0",
    "DRO_2_2_1",
    "DRO_2_2_2",
    "DRO_2_3",
    "DRO_2_4",
    "DRO_2_5",
    "DRO_2_6_0",
    "DRO_2_6_1",
    "DRO_2_6_2",
    "DRO_3_0",
    "DRO_3_1",
    "DRO_3_2_0",
    "DRO_3_2_1",
    "DRO_3_2_2",
    "DRO_3_2_3",
    "DRO_3_3_0",
    "DRO_3_3_1",
    "DRO_3_4_0",
    "DRO_3_4_1",
    "DRO_3_4_2",
    "DRO_3_4_3",
    "DRO_3_5_0",
    "DRO_3_5_1",
    "DRO_3_5_2",
    "DRO_3_5_3",
    "DRO_4_0",
    "DRO_4_1",
    "DRO_4_2",
    "DRO_4_3",
    "DRO_4_4",
    "DRO_4_5",
    "DRO_5_0",
    "DRO_7_0_0",
    "DRO_7_1_0",
    "DRO_7_2_0",
    "DRO_7_3_0",
    "DRO_7_3_1",
)

ERROR_IBSI_SUV_DROS = (
    "DRO_error_2_0",
    "DRO_error_2_1",
    "DRO_error_2_2",
    "DRO_error_2_3",
    "DRO_error_2_4",
    "DRO_error_2_5",
    "DRO_error_2_6",
    "DRO_error_2_7",
    "DRO_error_3_0",
    "DRO_error_3_1",
    "DRO_error_3_2",
    "DRO_error_4_0",
    "DRO_error_4_1",
    "DRO_error_4_2",
    "DRO_error_5_0",
)


@pytest.fixture(scope="session")
def ibsi_suv_mask(ibsi_suv_data_dir):
    return Image.from_nifti(ibsi_suv_data_dir / "DRO_error_5_0" / "mask" / "DRO_mask.nii.gz")


@pytest.mark.integration
@pytest.mark.parametrize("phantom_name", VALID_IBSI_SUV_DROS + ERROR_IBSI_SUV_DROS)
def test_official_ibsi_suv_dro(ibsi_suv_data_dir, ibsi_suv_mask, phantom_name):
    dicom_dir = ibsi_suv_data_dir / phantom_name / "PT"

    if "error" in phantom_name:
        with pytest.raises(Exception):
            Image.from_dicom(dicom_dir=dicom_dir, modality="PET")
        return

    image = Image.from_dicom(dicom_dir=dicom_dir, modality="PET")
    masked_image = np.where(ibsi_suv_mask.array > 0, image.array, np.nan)
    actual = tuple(round(float(summary(masked_image)), 2) for summary in (np.nanmin, np.nanmedian, np.nanmax))

    assert actual == (0.2, 1.0, 4.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("20250101110000+0100", datetime(2025, 1, 1, 11, 0, tzinfo=timezone(timedelta(hours=1)))),
        (
            "20250101111459.998530+0100",
            datetime(2025, 1, 1, 11, 14, 59, 998530, tzinfo=timezone(timedelta(hours=1))),
        ),
    ],
)
def test_parse_time_accepts_dicom_datetime_utc_offsets(value, expected):
    assert parse_time(value) == expected


def test_parse_time_preserves_offsets_for_instant_comparison():
    injection = parse_time("20250330013000+0100")
    reference = parse_time("20250330030000+0200")

    assert reference - injection == timedelta(minutes=30)
