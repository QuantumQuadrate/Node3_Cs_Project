"""Deterministic-loading SPCM count histogram applet.

Plots an overlaid histogram of:
  * MOT-only counts (from det_first_unloaded_counts) -- one per shot.
  * Atom-loaded counts (from det_spcm1_at_load + det_spcm2_at_load,
    filtered to where the loader succeeded).

Uses bright cyan + magenta brushes visible against pyqtgraph's default
black background.

API contract (verified against this ARTIQ version's simple.py):
  - widget __init__(self, args)        # ONE positional arg, no `req`
  - data_changed(self, data, mods)     # data is a dict {dataset_name:
                                       #   (persist, value, metadata)}
  - dataset value access: data[name][1]

LAUNCH COMMAND -- paste into the ARTIQ dashboard's applet "Command" box
(one line, the three dataset names are required positional args because
SimpleApplet.add_dataset in this ARTIQ build does NOT support default=):

  ${python} "C:/Networking Experiment/Artiq7_TEST_NEW/useful_codes/repository/applets/plot_loading_histogram.py" det_first_unloaded_counts det_spcm1_at_load det_spcm2_at_load
"""

import sys
import numpy as np
from artiq.applets.simple import SimpleApplet
import pyqtgraph as pg


def _log(msg):
    """Diagnostic output to stderr. Not stdout, which can interact with
    parent-process IPC in some configurations."""
    print(msg, file=sys.stderr, flush=True)


class LoadingHistogramPlot(pg.PlotWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setTitle("Deterministic loading: SPCM counts per gate")
        self.setLabel("bottom", "Counts (SPCM1 + SPCM2)")
        self.setLabel("left", "# shots")
        _log("[plot_loading_histogram] __init__ done")

    def data_changed(self, data, mods):
        try:
            # data is a dict mapping dataset name -> (persist, value, metadata)
            # Use .get with sentinel so missing datasets degrade gracefully.
            unloaded = data.get(self.args.unloaded, (False, None))[1] or []
            s1 = data.get(self.args.spcm1_loaded, (False, None))[1] or []
            s2 = data.get(self.args.spcm2_loaded, (False, None))[1] or []

            unloaded = np.asarray(unloaded, dtype=float)
            s1 = np.asarray(s1, dtype=float)
            s2 = np.asarray(s2, dtype=float)

            # Atom-loaded sum, dropping timeouts (where both s1 and s2 are 0)
            if len(s1) == len(s2) and len(s1) > 0:
                tot = s1 + s2
                loaded = tot[(s1 > 0) | (s2 > 0)]
            else:
                loaded = np.array([], dtype=float)

            _log(f"[plot_loading_histogram] data_changed: "
                 f"mot_only={len(unloaded)} pts, loaded={len(loaded)} pts")

            if len(unloaded) == 0 and len(loaded) == 0:
                return

            self.clear()

            # MOT-only histogram (cyan)
            if len(unloaded) > 0:
                n_bins = min(50, max(10, int(np.sqrt(len(unloaded)))))
                hist, edges = np.histogram(unloaded, bins=n_bins)
                centers = 0.5 * (edges[:-1] + edges[1:])
                width = edges[1] - edges[0] if len(edges) > 1 else 1.0
                self.addItem(pg.BarGraphItem(
                    x=centers, height=hist, width=width * 0.8, brush="c"))

            # Atom-loaded histogram (magenta)
            if len(loaded) > 0:
                n_bins = min(50, max(10, int(np.sqrt(len(loaded)))))
                hist, edges = np.histogram(loaded, bins=n_bins)
                centers = 0.5 * (edges[:-1] + edges[1:])
                width = edges[1] - edges[0] if len(edges) > 1 else 1.0
                self.addItem(pg.BarGraphItem(
                    x=centers, height=hist, width=width * 0.8, brush="m"))

        except Exception as e:
            import traceback
            _log(f"[plot_loading_histogram] {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)


def main():
    applet = SimpleApplet(LoadingHistogramPlot)
    applet.add_dataset("unloaded",
                       "MOT-only counts dataset (e.g. det_first_unloaded_counts)")
    applet.add_dataset("spcm1_loaded",
                       "SPCM1 counts-at-load dataset (e.g. det_spcm1_at_load)")
    applet.add_dataset("spcm2_loaded",
                       "SPCM2 counts-at-load dataset (e.g. det_spcm2_at_load)")
    applet.run()


if __name__ == "__main__":
    main()
