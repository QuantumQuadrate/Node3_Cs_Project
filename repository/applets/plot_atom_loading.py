"""Atom loading rate monitor applet.

Subscribes to spcm_counts1 and spcm_counts2, plots a scatter of
total counts per shot. Shows threshold line.
"""

import numpy as np
import PyQt5
from PyQt5 import QtWidgets
from artiq.applets.simple import SimpleApplet
import pyqtgraph as pg


class AtomLoadingPlot(pg.PlotWidget):
    def __init__(self, args, req):
        super().__init__()
        self.args = args
        self.req = req
        self.setTitle("Atom Loading")
        self.setLabel("bottom", "Shot number")
        self.setLabel("left", "Total SPCM counts")
        self.scatter = self.plot(pen=None, symbol="o", symbolSize=4,
                                 symbolBrush="c")
        self.threshold_line = pg.InfiniteLine(pos=12, angle=0,
                                               pen=pg.mkPen("r", style=2))
        self.addItem(self.threshold_line)

    def data_changed(self, value, metadata, persist, mods):
        try:
            c1 = np.array(self.req.get_dataset(self.args.dataset1) or [])
            c2 = np.array(self.req.get_dataset(self.args.dataset2) or [])
            if len(c1) > 0 and len(c2) > 0:
                n = min(len(c1), len(c2))
                total = c1[:n] + c2[:n]
                self.scatter.setData(np.arange(n), total)

            try:
                thresh = float(self.req.get_dataset("atom_threshold") or 12)
                self.threshold_line.setValue(thresh)
            except Exception:
                pass
        except Exception:
            pass


def main():
    applet = SimpleApplet(AtomLoadingPlot)
    applet.add_dataset("dataset1", "SPCM1 dataset", default="spcm_counts1")
    applet.add_dataset("dataset2", "SPCM2 dataset", default="spcm_counts2")
    applet.run()


if __name__ == "__main__":
    main()
