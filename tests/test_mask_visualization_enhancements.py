import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from PyQt5.QtCore import QEvent, Qt  # noqa: E402
from PyQt5.QtGui import QColor, QKeyEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication, QColorDialog, QFileDialog, QInputDialog  # noqa: E402

from zrad.image import Image  # noqa: E402
from zrad.visualization.interactive_mask_visualization import Visualization  # noqa: E402


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
    assert mask[viewer.current_axial].sum() == 1


def test_fast_cursor_movement_produces_continuous_line(viewer, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("target", True))
    viewer._create_mask()
    viewer.brush_size.setValue(1)
    viewer._on_draw_started()

    viewer._on_view_drawn(viewer.axi_view, 0, 2 * viewer.sy)
    viewer._on_view_drawn(viewer.axi_view, 5 * viewer.sx, 2 * viewer.sy)

    assert viewer.masks[0]["data"][viewer.current_axial, 2, :].sum() == 6


def test_fill_bucket_fills_enclosed_area(viewer, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("target", True))
    viewer._create_mask()
    plane = viewer.masks[0]["data"][viewer.current_axial]
    plane[1:4, 1] = 1
    plane[1:4, 4] = 1
    plane[1, 1:5] = 1
    plane[3, 1:5] = 1
    viewer.fill_mode = True

    viewer._on_view_drawn(viewer.axi_view, 3 * viewer.sx, 2 * viewer.sy)

    assert np.all(plane[2, 2:4] == 1)
    assert plane[0, 0] == 0


def test_arrow_slice_step_uses_most_recent_view(viewer):
    viewer._active_slice_view = viewer.cor_view
    starting_slice = viewer.current_coronal

    viewer.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier))
    viewer.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier))

    assert viewer.current_coronal == starting_slice


def test_copy_paste_contour_between_slices(viewer, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("target", True))
    viewer._create_mask()
    viewer._active_slice_view = viewer.axi_view
    source_slice = viewer.current_axial
    viewer.masks[0]["data"][source_slice, 1:3, 2:4] = 1

    viewer.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_C, Qt.ControlModifier))
    viewer._step_active_slice(1)
    destination_slice = viewer.current_axial
    viewer.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_V, Qt.ControlModifier))

    assert destination_slice != source_slice
    assert np.array_equal(
        viewer.masks[0]["data"][destination_slice],
        viewer.masks[0]["data"][source_slice],
    )


def test_selected_mask_can_be_recolored(viewer, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("target", True))
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args: QColor(12, 34, 56))
    viewer._create_mask()

    viewer._choose_mask_color()

    assert viewer.masks[0]["color"] == (12, 34, 56)


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


def test_mask_selector_saves_a_mask_other_than_the_last(viewer, monkeypatch, tmp_path):
    names = iter((("first", True), ("second", True)))
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: next(names))
    viewer._create_mask()
    viewer.masks[0]["data"][0, 0, 0] = 1
    viewer._create_mask()
    viewer.masks[1]["data"][1, 1, 1] = 1
    viewer.mask_selector.setCurrentIndex(0)
    output_path = tmp_path / "selected-mask.nii.gz"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(output_path), "NIfTI"))

    viewer._save_active_mask()

    saved = sitk.GetArrayFromImage(sitk.ReadImage(str(output_path)))
    assert saved[0, 0, 0] == 1
    assert saved[1, 1, 1] == 0
