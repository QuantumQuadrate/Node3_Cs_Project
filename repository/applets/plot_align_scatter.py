"""SPCM3 alignment scatter applet.

Scatter of SPCM3 counts (y) vs the SPCM1+SPCM2 reference (x). When SPCM3 is
aligned onto the atom, high reference (atom present) coincides with high SPCM3,
so the cloud tilts into a rising line and the Pearson correlation -> 1. When
SPCM3 is misaligned, SPCM3 is flat/noise regardless of the reference (a blob,
r ~ 0). The live correlation r is shown in the title -- maximize it while moving
the SPCM3 detector.

Usage in the ARTIQ dashboard:
  Command:   ${python} "<repo>/applets/plot_align_scatter.py" spcm_counts3
  Arguments: --x spcm_sum12
"""

import numpy as np
import PyQt5  # ensure Qt5
import pyqtgraph

from artiq.applets.simple import TitleApplet


class AlignScatter(pyqtgraph.PlotWidget):
    def __init__(self, args):
        pyqtgraph.PlotWidget.__init__(self)
        self.args = args
        self.setLabel("bottom", "SPCM1 + SPCM2  (reference)")
        self.setLabel("left", "SPCM3 counts")
        self.showGrid(x=True, y=True, alpha=0.3)

    def data_changed(self, data, mods, title):
        try:
            y = data[self.args.y][1]
        except KeyError:
            return
        x = data.get(self.args.x, (False, None))[1]
        if y is None or not len(y) or x is None or not len(x):
            return

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = min(len(x), len(y))
        if n < 1:
            return
        x = x[-n:]
        y = y[-n:]

        # Pearson correlation (guard against zero variance).
        r = 0.0
        if n >= 2:
            sx = x.std(); sy = y.std()
            if sx > 0 and sy > 0:
                r = float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))

        self.clear()
        self.plot(x, y, pen=None, symbol="o", symbolSize=5,
                  symbolBrush=(80, 220, 100), symbolPen=None)
        self.setTitle(f"SPCM3 vs reference   |   r = {r:+.3f}   (N={n})")

        # Keep the y-floor at 0 (counts are non-negative), top auto-scales.
        ymax = float(y.max()) if y.size else 1.0
        self.enableAutoRange(axis="y", enable=False)
        self.setYRange(0.0, ymax * 1.15 if ymax > 0 else 1.0, padding=0)


def main():
    applet = TitleApplet(AlignScatter)
    applet.add_dataset("y", "SPCM3 counts (Y values)")
    applet.add_dataset("x", "reference = SPCM1+SPCM2 (X values)", required=False)
    applet.run()


if __name__ == "__main__":
    main()
