"""Parity oscillation live plot applet.

Subscribes to parity measurement datasets and plots
counts vs HWP angle, updated in real time during scans.
"""

import numpy as np
import PyQt5
from PyQt5 import QtWidgets
from artiq.applets.simple import SimpleApplet
import pyqtgraph as pg


class ParityPlot(pg.PlotWidget):
    def __init__(self, args, req):
        super().__init__()
        self.args = args
        self.req = req
        self.setTitle("Parity Oscillation")
        self.setLabel("bottom", "HWP angle (deg)")
        self.setLabel("left", "Signal counts")
        self.addLegend()
        self.scatter = self.plot(pen=None, symbol="o", symbolSize=6,
                                 symbolBrush="c", name="Counts")

    def data_changed(self, value, metadata, persist, mods):
        try:
            hwp = np.array(self.req.get_dataset("HWP_angles") or [])
            counts = np.array(self.req.get_dataset("counts_signal") or [])
            if len(hwp) > 0 and len(counts) > 0:
                n = min(len(hwp), len(counts))
                self.scatter.setData(hwp[:n], counts[:n])
        except Exception:
            pass


def main():
    applet = SimpleApplet(ParityPlot)
    applet.run()


if __name__ == "__main__":
    main()
