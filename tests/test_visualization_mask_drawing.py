import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog  # noqa: E402

from zrad.image import Image  # noqa: E402
from zrad.visualization.visualization import Visualization  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(qapp):
    image = Image(
        array=np.zeros((4, 5, 6), dtype=np.float32),
        origin=(11.0, 12.0, 13.0),
        spacing=(0.7, 0.8, 1.5),
        direction=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        shape=(6, 5, 4),
    )
    window = Visualization([{"image": image, "image_name": "case", "masks": []}])
    yield window
    window.close()


def test_created_mask_can_be_drawn_in_axial_view(viewer, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("target", True))
    viewer._create_mask()
    viewer.brush_size.setValue(1)

    viewer._on_view_drawn(viewer.axi_view, 2 * viewer.sx, 2 * viewer.sy)

    mask = viewer.masks[0]["data"]
    assert mask.dtype == np.uint8
    assert mask[viewer.current_axial].sum() == 5


def test_saved_mask_uses_loaded_image_geometry(viewer, monkeypatch, tmp_path):
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("target", True))
    viewer._create_mask()
    output_path = tmp_path / "drawn-mask.nii.gz"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(output_path), "NIfTI"))

    viewer._save_active_mask()

    saved = sitk.ReadImage(str(output_path))
    assert saved.GetOrigin() == pytest.approx(viewer.current_image.origin)
    assert saved.GetSpacing() == pytest.approx(viewer.current_image.spacing)
    assert saved.GetDirection() == pytest.approx(viewer.current_image.direction)
    assert saved.GetSize() == viewer.current_image.shape
    assert saved.GetPixelID() == sitk.sitkUInt8
