"""Conflict-free GUI entry point for the interactive mask viewer."""

from ..visualization.interactive_mask_visualization import Visualization
from . import visual_tab


class MaskVisualizationTab(visual_tab.VisualizationTab):
    """Visualization tab that opens the interactive mask-capable viewer."""

    def run_selection(self):
        # ``VisualizationTab.run_selection`` resolves the viewer from its own
        # module. Replace that dependency only while using this specialized tab,
        # without modifying the shared ``visual_tab.py`` implementation.
        visual_tab.Visualization = Visualization
        return super().run_selection()
