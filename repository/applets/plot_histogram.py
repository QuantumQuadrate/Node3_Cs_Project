"""SPCM count histogram applet with threshold marker.

Subscribes to spcm_counts1 and spcm_counts2, displays a combined
histogram with the atom detection threshold marked.
"""

import numpy as np
import PyQt5
from PyQt5 import QtWidgets
from artiq.applets.simple import SimpleApplet
import pyqtgraph as pg


class HistogramPlot(pg.PlotWidget):
    def __init__(self, args, req):
        super().__init__()
        self.args = args
        self.req = req
        self.setTitle("SPCM Count Histogram")
        self.setLabel("bottom", "Counts")
        self.setLabel("left", "Frequency")
        self.threshold_line = pg.InfiniteLine(pos=12, angle=90,
                                               pen=pg.mkPen("r", width=2, style=2))
        self.addItem(self.threshold_line)

    def data_changed(self, value, metadata, persist, mods):
        try:
            c1 = np.array(self.req.get_dataset(self.args.dataset1) or [])
            c2 = np.array(self.req.get_dataset(self.args.dataset2) or [])
            combined = np.concatenate([c1, c2]) if len(c1) > 0 or len(c2) > 0 else np.array([])

            if len(combined) > 0:
                self.clear()
                self.addItem(self.threshold_line)

                n_bins = min(50, max(10, int(np.sqrt(len(combined)))))
                hist, bin_edges = np.histogram(combined, bins=n_bins)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                width = bin_edges[1] - bin_edges[0]
                bars = pg.BarGraphItem(x=bin_centers, height=hist,
                                        width=width * 0.8, brush="c")
                self.addItem(bars)

            try:
                thresh = float(self.req.get_dataset("atom_threshold") or 12)
                self.threshold_line.setValue(thresh)
            except Exception:
                pass
        except Exception:
            pass


def main():
    applet = SimpleApplet(HistogramPlot)
    applet.add_dataset("dataset1", "SPCM1 dataset", default="spcm_counts1")
    applet.add_dataset("dataset2", "SPCM2 dataset", default="spcm_counts2")
    applet.run()


if __name__ == "__main__":
    main()
