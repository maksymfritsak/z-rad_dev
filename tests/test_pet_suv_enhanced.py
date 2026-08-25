import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

from zrad.io.pet_suv import _apply_enhanced_suv_correction


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


@pytest.mark.parametrize(
    ("decay_corrected", "reference_keyword", "expected"),
    [
        ("YES", "DecayCorrectionDateTime", 2000.0),
        ("NO", "FrameReferenceDateTime", 1000.0),
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
    radiopharmaceutical.RadionuclideHalfLife = 3600
    radiopharmaceutical.RadionuclideTotalDose = 100000
    radiopharmaceutical.RadiopharmaceuticalStartDateTime = "20200101100000"
    ds.RadiopharmaceuticalInformationSequence = Sequence([radiopharmaceutical])

    source = sitk.GetImageFromArray(np.zeros((1, 1, 1), dtype=np.uint16))
    result = _apply_enhanced_suv_correction(ds, source)

    assert sitk.GetArrayFromImage(result)[0, 0, 0] == pytest.approx(expected)
