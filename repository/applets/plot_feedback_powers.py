"""AOM feedback power monitor bar chart applet.

Displays a bar chart of the 5 feedback channel powers normalized
to their setpoints. Green = within tolerance, red = drifting.

Subscribes to: feedback_MOT_X, feedback_MOT_Y, feedback_MOT_Z,
               feedback_FORT, feedback_excitation
"""

import numpy as np
import PyQt5
from PyQt5 import QtWidgets
from artiq.applets.simple import SimpleApplet
import pyqtgraph as pg


CHANNEL_NAMES = ["MOT X", "MOT Y", "MOT Z", "FORT", "Excitation"]
DATASET_NAMES = [
    "feedback_MOT_X", "feedback_MOT_Y", "feedback_MOT_Z",
    "feedback_FORT", "feedback_excitation",
]


class FeedbackPowersPlot(pg.PlotWidget):
    def __init__(self, args, req):
        super().__init__()
        self.args = args
        self.req = req
        self.setTitle("AOM Feedback Powers")
        self.setLabel("left", "Normalized Power")

        x = np.arange(len(CHANNEL_NAMES))
        self.bars = pg.BarGraphItem(x=x, height=np.zeros(5), width=0.6,
                                     brush="g")
        self.addItem(self.bars)

        ax = self.getAxis("bottom")
        ax.setTicks([list(zip(x, CHANNEL_NAMES))])

    def data_changed(self, value, metadata, persist, mods):
        try:
            heights = []
            for ds_name in DATASET_NAMES:
                try:
                    val = self.req.get_dataset(ds_name)
                    heights.append(float(val) if val is not None else 0.0)
                except Exception:
                    heights.append(0.0)

            colors = []
            for h in heights:
                if 0.95 <= h <= 1.05:
                    colors.append("g")
                elif 0.9 <= h <= 1.1:
                    colors.append("y")
                else:
                    colors.append("r")

            x = np.arange(len(heights))
            self.removeItem(self.bars)
            self.bars = pg.BarGraphItem(x=x, height=heights, width=0.6,
                                         brushes=colors)
            self.addItem(self.bars)
        except Exception:
            pass


def main():
    applet = SimpleApplet(FeedbackPowersPlot)
    applet.run()


if __name__ == "__main__":
    main()
