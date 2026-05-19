"""Real-time dual-SPCM count monitor applet.

Shows SPCM1 and SPCM2 counts on the same plot with a rolling window.

Usage in ARTIQ dashboard (add as applet):
  Command: ${python} -m applets.plot_spcm_counts spcm_counts1
  Arguments: --y2 spcm_counts2 --pts 300

Or for a single channel:
  Command: ${python} -m applets.plot_spcm_counts spcm_counts1 --pts 300
"""

import numpy as np
import PyQt5  # ensure Qt5
import pyqtgraph

from artiq.applets.simple import TitleApplet


class DualSPCMPlot(pyqtgraph.PlotWidget):
    def __init__(self, args):
        pyqtgraph.PlotWidget.__init__(self)
        self.args = args
        self.setLabel("bottom", "Shot number")
        self.setLabel("left", "Counts")
        self.addLegend()

    def data_changed(self, data, mods, title):
        try:
            y1 = data[self.args.y][1]
        except KeyError:
            return
        if y1 is None or not len(y1):
            return

        pts = data.get(self.args.pts, (False, None))[1]
        y2 = data.get(self.args.y2, (False, None))[1]

        # Apply rolling window
        if pts is not None:
            pts = int(pts)
            y1 = y1[-pts:]
        x1 = np.arange(len(y1))

        self.clear()

        # SPCM1
        self.plot(x1, np.array(y1),
                  pen=pyqtgraph.mkPen((0, 180, 255), width=1),
                  symbol="o",
                  symbolSize=4,
                  symbolBrush=(0, 180, 255),
                  symbolPen=None,
                  name="SPCM1")

        # SPCM2 (if provided)
        if y2 is not None and len(y2):
            if pts is not None:
                y2 = y2[-pts:]
            x2 = np.arange(len(y2))
            self.plot(x2, np.array(y2),
                      pen=pyqtgraph.mkPen((255, 100, 200), width=1),
                      symbol="o",
                      symbolSize=4,
                      symbolBrush=(255, 100, 200),
                      symbolPen=None,
                      name="SPCM2")

        self.setTitle(title)


def main():
    applet = TitleApplet(DualSPCMPlot)
    applet.add_dataset("y", "SPCM1 counts (Y values)")
    applet.add_dataset("y2", "SPCM2 counts", required=False)
    applet.add_dataset("pts", "Max points to display", required=False)
    applet.run()


if __name__ == "__main__":
    main()
