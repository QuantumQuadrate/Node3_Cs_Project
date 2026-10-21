"""Live monitor of spcm_tot -- summed SPCM1+2 atom fluorescence (trapped windows).

Companion applet for atom_align_spcm12_atom_trapped_only.py. Scrolling trace of
the SPCM1+SPCM2 total, y-floor pinned at 0 with a sticky (non-shrinking) top so
the window stays stable. The title shows the latest value and the mean of the
visible points -- maximize both while aligning the SPCM1/2 collection optics.

Usage in the ARTIQ dashboard:
  Command: ${python} "<repo>/applets/plot_spcm_tot.py" spcm_tot
"""

import numpy as np
import PyQt5  # ensure Qt5
import pyqtgraph

from artiq.applets.simple import TitleApplet


class SpcmTotPlot(pyqtgraph.PlotWidget):
    def __init__(self, args):
        pyqtgraph.PlotWidget.__init__(self)
        self.args = args
        self.setLabel("bottom", "trapped-window number")
        self.setLabel("left", "SPCM1+2 total  [trapped only]")
        self._ytop = None   # sticky y-axis top (grows only) -> stable window

    def data_changed(self, data, mods, title):
        try:
            y = data[self.args.y][1]
        except KeyError:
            return
        if y is None or not len(y):
            return

        y = np.asarray(y, dtype=float)
        x = np.arange(len(y))

        self.clear()
        self.plot(x, y,
                  pen=pyqtgraph.mkPen((0, 180, 255), width=1),
                  symbol="o", symbolSize=4,
                  symbolBrush=(0, 180, 255), symbolPen=None,
                  name="spcm_tot")
        self.setTitle(f"spcm_tot   last = {y[-1]:.0f}   mean = {y.mean():.1f}   (N={len(y)})")

        # Pin the floor at 0 (counts are non-negative); top auto-scales but only
        # ever grows, so the window does not jump around as points scroll.
        ymax = float(y.max()) if y.size else 1.0
        top = ymax * 1.15 if ymax > 0 else 1.0
        if self._ytop is None or top > self._ytop:
            self._ytop = top
        self.enableAutoRange(axis="y", enable=False)
        self.setYRange(0.0, float(self._ytop), padding=0)


def main():
    applet = TitleApplet(SpcmTotPlot)
    applet.add_dataset("y", "spcm_tot dataset (SPCM1+2 totals, trapped only)")
    applet.run()


if __name__ == "__main__":
    main()
