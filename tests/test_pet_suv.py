from datetime import datetime, timedelta, timezone

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

from zrad.exceptions import DataStructureError
from zrad.image import Image
from zrad.io.pet_suv import _apply_enhanced_suv_correction, _enhanced_suv_factor, parse_time

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


def _enhanced_pet(pixel_values):
    values = np.asarray(pixel_values, dtype=np.uint16)
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = pydicom.dataset.FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.130"
    ds.Rows = values.shape[-2]
    ds.Columns = values.shape[-1]
    ds.NumberOfFrames = values.shape[0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = values.tobytes()
    return ds


def _mapping(slope, unit="g/ml{SUVbw}"):
    mapping = Dataset()
    mapping.RealWorldValueSlope = slope
    mapping.RealWorldValueIntercept = 0
    units = Dataset()
    units.CodeValue = unit
    mapping.MeasurementUnitsCodeSequence = Sequence([units])
    return mapping


def test_enhanced_pet_uses_per_frame_real_world_value_slopes():
    ds = _enhanced_pet([[[2]], [[2]]])
    ds.PerFrameFunctionalGroupsSequence = Sequence([])
    for slope in (3, 5):
        group = Dataset()
        group.RealWorldValueMappingSequence = Sequence([_mapping(slope)])
        ds.PerFrameFunctionalGroupsSequence.append(group)

    source = sitk.GetImageFromArray(np.zeros((2, 1, 1), dtype=np.uint16))
    result = _apply_enhanced_suv_correction(ds, source)

    np.testing.assert_allclose(sitk.GetArrayFromImage(result)[:, 0, 0], [6, 10])


@pytest.mark.parametrize("suv_type", ["LBM", "LBMJAMES128", "LBMJANMA", "IBW"])
def test_enhanced_pet_gml_fallback_honors_suv_type(suv_type):
    ds = _enhanced_pet([[[10]]])
    ds.PatientWeight = 80
    ds.PatientSize = 1.8
    ds.PatientSex = "M"
    ds.add_new((0x0054, 0x1006), "CS", suv_type)
    transformation = Dataset()
    transformation.RescaleSlope = 2
    transformation.RescaleIntercept = 0
    transformation.RescaleType = "GML"
    group = Dataset()
    group.PixelValueTransformationSequence = Sequence([transformation])
    ds.PerFrameFunctionalGroupsSequence = Sequence([group])

    source = sitk.GetImageFromArray(np.zeros((1, 1, 1), dtype=np.uint16))
    result = _apply_enhanced_suv_correction(ds, source)

    expected = 20 * _enhanced_suv_factor(ds, suv_type)
    assert sitk.GetArrayFromImage(result)[0, 0, 0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("decay_corrected", "reference_keyword", "expected"),
    [
        ("YES", "DecayCorrectionDateTime", 1000.0),
        ("NO", "FrameReferenceDateTime", 1000.0 / np.sqrt(2)),
    ],
)
def test_enhanced_pet_bqml_uses_decay_reference_datetime(decay_corrected, reference_keyword, expected):
    ds = _enhanced_pet([[[1000]]])
    ds.PatientWeight = 50
    ds.DecayCorrected = decay_corrected
    group = Dataset()
    group.RealWorldValueMappingSequence = Sequence([_mapping(1, "Bq/ml")])
    if reference_keyword == "FrameReferenceDateTime":
        group.FrameReferenceDateTime = "20200101110000"
    else:
        ds.DecayCorrectionDateTime = "20200101120000"
    ds.PerFrameFunctionalGroupsSequence = Sequence([group])
    radiopharmaceutical = Dataset()
    radiopharmaceutical.RadionuclideHalfLife = 7200
    radiopharmaceutical.RadionuclideTotalDose = 100000
    radiopharmaceutical.RadiopharmaceuticalStartDateTime = "20200101100000"
    ds.RadiopharmaceuticalInformationSequence = Sequence([radiopharmaceutical])

    source = sitk.GetImageFromArray(np.zeros((1, 1, 1), dtype=np.uint16))
    result = _apply_enhanced_suv_correction(ds, source)

    assert sitk.GetArrayFromImage(result)[0, 0, 0] == pytest.approx(expected)


def test_enhanced_pet_bqml_compares_datetime_instants_across_utc_offsets():
    ds = _enhanced_pet([[[1000]]])
    ds.PatientWeight = 50
    ds.DecayCorrected = "YES"
    ds.DecayCorrectionDateTime = "20250330030000+0200"
    group = Dataset()
    group.RealWorldValueMappingSequence = Sequence([_mapping(1, "Bq/ml")])
    ds.PerFrameFunctionalGroupsSequence = Sequence([group])
    radiopharmaceutical = Dataset()
    radiopharmaceutical.RadionuclideHalfLife = 3600
    radiopharmaceutical.RadionuclideTotalDose = 100000
    radiopharmaceutical.RadiopharmaceuticalStartDateTime = "20250330013000+0100"
    ds.RadiopharmaceuticalInformationSequence = Sequence([radiopharmaceutical])

    source = sitk.GetImageFromArray(np.zeros((1, 1, 1), dtype=np.uint16))
    result = _apply_enhanced_suv_correction(ds, source)

    assert sitk.GetArrayFromImage(result)[0, 0, 0] == pytest.approx(1000.0 / np.sqrt(2))


def test_enhanced_pet_bqml_rejects_implausible_reconstructed_injection_datetime():
    ds = _enhanced_pet([[[1000]]])
    ds.PatientWeight = 50
    ds.DecayCorrected = "YES"
    ds.DecayCorrectionDateTime = "20200101100000"
    group = Dataset()
    group.RealWorldValueMappingSequence = Sequence([_mapping(1, "Bq/ml")])
    ds.PerFrameFunctionalGroupsSequence = Sequence([group])
    radiopharmaceutical = Dataset()
    radiopharmaceutical.RadionuclideHalfLife = 3600
    radiopharmaceutical.RadionuclideTotalDose = 100000
    radiopharmaceutical.RadiopharmaceuticalStartDateTime = "19900101120000"
    ds.RadiopharmaceuticalInformationSequence = Sequence([radiopharmaceutical])

    source = sitk.GetImageFromArray(np.zeros((1, 1, 1), dtype=np.uint16))

    with pytest.raises(DataStructureError, match="Reconstructed administration datetime"):
        _apply_enhanced_suv_correction(ds, source)
