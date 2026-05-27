"""Confirmation-shot SPCM count histogram applet for deterministic loading.

Companion to plot_loading_histogram.py. Where that one shows the
DETECTION-stage signal (during polling), this one shows the
CONFIRMATION-stage signal (the 4 ms fluorescence imaging shot after
the loader returns).

Two histograms overlaid:
  * Background  (cyan)  -- bg1+bg2 (cooling + repump on, FORT off,
                            measured before each shot). A "no atom"
                            reference.
  * Confirmation (red)  -- count1+count2 (same gate, FORT on, after the
                            loader said success). For real atoms this
                            should sit clearly above the background.

Interpretation:
  * If red distribution is well separated from cyan and shifted high
    (typically 50-300 counts per 4 ms gate) -- atoms are real.
  * If red overlaps cyan or both are at the same low counts (~5-10) --
    loader is firing on background noise; the apparent "100% success"
    is a false positive.

API contract (verified against this ARTIQ version's simple.py):
  - widget __init__(self, args)
  - data_changed(self, data, mods)
  - dataset value access: data[name][1]

LAUNCH COMMAND -- paste into the dashboard's applet box (one line, the
four dataset names are required positional args):

  ${python} "C:/Networking Experiment/Artiq7_TEST_NEW/useful_codes/repository/applets/plot_confirmation_histogram.py" det_confirm_count1 det_confirm_count2 det_confirm_bg1 det_confirm_bg2
"""

import sys
import numpy as np
from artiq.applets.simple import SimpleApplet
import pyqtgraph as pg


def _log(msg):
    """Diagnostic output to stderr -- not stdout."""
    print(msg, file=sys.stderr, flush=True)


class ConfirmationHistogramPlot(pg.PlotWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setTitle("Confirmation imaging: SPCM counts per 4 ms gate")
        self.setLabel("bottom", "Counts (SPCM1 + SPCM2)")
        self.setLabel("left", "# shots")
        _log("[plot_confirmation_histogram] __init__ done")

    def data_changed(self, data, mods):
        try:
            c1 = data.get(self.args.conf1, (False, None))[1] or []
            c2 = data.get(self.args.conf2, (False, None))[1] or []
            b1 = data.get(self.args.bg1, (False, None))[1] or []
            b2 = data.get(self.args.bg2, (False, None))[1] or []

            c1 = np.asarray(c1, dtype=float)
            c2 = np.asarray(c2, dtype=float)
            b1 = np.asarray(b1, dtype=float)
            b2 = np.asarray(b2, dtype=float)

            # Sum SPCM1 + SPCM2 for each distribution. Require equal lengths
            # (one per shot) to align the sum; if mismatched, fall back to
            # empty.
            if len(c1) == len(c2) and len(c1) > 0:
                confirmation = c1 + c2
            else:
                confirmation = np.array([], dtype=float)

            if len(b1) == len(b2) and len(b1) > 0:
                background = b1 + b2
            else:
                background = np.array([], dtype=float)

            _log(f"[plot_confirmation_histogram] data_changed: "
                 f"confirmation={len(confirmation)} pts, "
                 f"background={len(background)} pts")

            if len(confirmation) == 0 and len(background) == 0:
                return

            self.clear()

            # Shared binning so the two distributions overlay cleanly
            if len(confirmation) > 0 and len(background) > 0:
                all_data = np.concatenate([confirmation, background])
            elif len(confirmation) > 0:
                all_data = confirmation
            else:
                all_data = background

            n_bins = min(50, max(10, int(np.sqrt(max(1, len(all_data))))))
            lo = float(np.min(all_data))
            hi = float(np.max(all_data))
            if hi <= lo:
                hi = lo + 1.0
            edges = np.linspace(lo, hi, n_bins + 1)
            width = edges[1] - edges[0] if len(edges) > 1 else 1.0
            centers = 0.5 * (edges[:-1] + edges[1:])

            # Background (cyan, slight left offset)
            if len(background) > 0:
                hist, _ = np.histogram(background, bins=edges)
                self.addItem(pg.BarGraphItem(
                    x=centers - width * 0.2, height=hist,
                    width=width * 0.4, brush="c"))

            # Confirmation (red, slight right offset)
            if len(confirmation) > 0:
                hist, _ = np.histogram(confirmation, bins=edges)
                self.addItem(pg.BarGraphItem(
                    x=centers + width * 0.2, height=hist,
                    width=width * 0.4, brush="r"))

            # Quick at-a-glance status in the title: medians + the
            # success/fail diagnostic (median confirmation - median bg).
            n_c = int(len(confirmation))
            n_b = int(len(background))
            med_c = float(np.median(confirmation)) if n_c > 0 else 0.0
            med_b = float(np.median(background)) if n_b > 0 else 0.0
            gap = med_c - med_b
            self.setTitle(
                f"Confirmation imaging  --  cyan = bg (n={n_b}, median={med_b:.1f}),  "
                f"red = atom-loaded (n={n_c}, median={med_c:.1f}),  "
                f"gap = {gap:.1f} counts"
            )

        except Exception as e:
            import traceback
            _log(f"[plot_confirmation_histogram] {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)


def main():
    applet = SimpleApplet(ConfirmationHistogramPlot)
    applet.add_dataset("conf1", "Confirmation SPCM1 counts dataset "
                                "(typically det_confirm_count1)")
    applet.add_dataset("conf2", "Confirmation SPCM2 counts dataset "
                                "(typically det_confirm_count2)")
    applet.add_dataset("bg1",   "Background SPCM1 counts dataset "
                                "(typically det_confirm_bg1)")
    applet.add_dataset("bg2",   "Background SPCM2 counts dataset "
                                "(typically det_confirm_bg2)")
    applet.run()


if __name__ == "__main__":
    main()
