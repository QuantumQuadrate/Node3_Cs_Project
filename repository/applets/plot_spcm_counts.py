"""Real-time SPCM count monitor applet (up to three traces).

The FIRST dataset is POSITIONAL (no flag); the rest are optional flags.

  <applet> spcm_counts1 --y2 spcm_counts2 --y3 spcm_counts3 --ymin 0 --ymax 250

Options:
  --ymin / --ymax   pin the y-axis (floor defaults to 0; without --ymax the top
                    auto-scales but only ever grows, so the window stays stable)
  --bin N           plot the MEAN of every N consecutive shots as ONE point --
                    smooths shot noise when you are watching for a small change.
                    Display only: the dataset keeps every shot. N=1 (default) is
                    the raw per-shot trace. Bin boundaries are FIXED (shots
                    [0,N), [N,2N), ...) so a point never moves once drawn; the
                    trace gains one point every N shots, and the partial bin at
                    the end is not shown until full. The bottom axis still shows
                    the shot number. (If the plotted dataset is a rolling window
                    that has hit its cap, the array itself slides and the points
                    will drift -- keep the experiment's max_points above the run
                    length.)
  --pts <dataset>   limit the trace to the last N points, where N comes from a
                    DATASET (e.g. --pts spcm_plot_pts). `--pts 300` does NOT
                    work: it looks for a dataset literally named "300", finds
                    nothing, and silently plots everything. Experiments that cap
                    their own buffer (max_points) need no --pts at all.
"""

import numpy as np
import PyQt5  # ensure Qt5
import pyqtgraph

from artiq.applets.simple import TitleApplet


def _bin_avg(y, n, offset=0):
    """Block-average `y` in groups of `n`: one plotted point per n shots.

    `offset` is the ABSOLUTE shot number of y[0]. Pass it (via --total) whenever
    the dataset is a rolling window, or the bins cannot stay still -- see the
    CAVEAT below.

    DISPLAY ONLY -- the dataset itself is untouched. Returns (x, y_binned),
    where x is the shot number at each bin's centre, so the bottom axis keeps
    meaning the same thing whether or not binning is on.

    Bins are aligned to the START with FIXED boundaries: bin k always covers
    shots [k*n, (k+1)*n). A point therefore never moves once drawn -- it is the
    mean of the same n shots forever. The trailing partial bin is NOT plotted
    until it is full, so the trace grows one finished point every n shots
    instead of showing an average that drifts while it fills.

    (Aligning to the END instead -- keeping the newest shots in a full bin and
    dropping the oldest leftover -- re-partitions the WHOLE history on every new
    shot, so every past point visibly changes. Don't.)

    CAVEAT -- why `offset` is needed. A ROLLING dataset (the experiment's
    `max_points` cap) slides left by one every shot once it is full, so array
    POSITION stops meaning shot number: bins defined on the position regroup
    different shots each update and the points drift again. With `offset` the
    boundaries are anchored to ABSOLUTE multiples of n instead, so a bin always
    averages the same shots however far the window has slid -- old points scroll
    off the left edge but never change value. offset=0 (no --total) keeps the
    old position-based behaviour, which is only static while the dataset has not
    yet reached its cap.
    """
    a = np.asarray(y, dtype=float)
    offset = int(offset)
    if n <= 1:
        return offset + np.arange(a.size, dtype=float), a
    # First element whose ABSOLUTE index is a multiple of n starts bin 0 here.
    start = (-offset) % n
    nbins = max(0, (a.size - start) // n)    # trailing partial bin: not plotted
    if nbins == 0:
        return np.array([]), np.array([])
    yb = a[start:start + nbins * n].reshape(nbins, n).mean(axis=1)
    xb = offset + start + np.arange(nbins) * n + 0.5 * (n - 1)
    return xb, yb


class DualSPCMPlot(pyqtgraph.PlotWidget):
    def __init__(self, args):
        pyqtgraph.PlotWidget.__init__(self)
        self.args = args
        self.setLabel("bottom", "Shot number")
        # Distinctive label: if the plot's y-axis does NOT read "[y pinned]",
        # you are running a DIFFERENT applet than this patched file.
        nbin = int(getattr(args, "bin", 1) or 1)
        label = "Counts  [y pinned]"
        if nbin > 1:
            label += "   (mean of {} shots)".format(nbin)
        self.setLabel("left", label)
        self.addLegend()
        self._ytop = None   # sticky y-axis top (grows only) -> stable window

    def data_changed(self, data, mods, title):
        try:
            y1 = data[self.args.y][1]
        except KeyError:
            return
        if y1 is None or not len(y1):
            return

        pts = data.get(self.args.pts, (False, None))[1]
        y2 = data.get(self.args.y2, (False, None))[1]
        y3 = data.get(self.args.y3, (False, None))[1]

        # Absolute shot number of the FIRST element still in the dataset.
        # --total names a dataset holding the TOTAL shots recorded so far; the
        # difference is how many have already scrolled out of a capped buffer.
        # Without it, offset 0 -- fine only until the buffer fills.
        total = data.get(getattr(self.args, "total", None), (False, None))[1]
        nbin = int(getattr(self.args, "bin", 1) or 1)

        def _prep(y):
            """Apply the --pts window, then bin, tracking the absolute offset."""
            raw_n = len(y)
            off = (int(total) - raw_n) if total is not None else 0
            if pts is not None:
                y = y[-int(pts):]
                off += raw_n - len(y)        # --pts drops older points too
            return _bin_avg(y, nbin, off)

        x1, y1 = _prep(y1)

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
            x2, y2 = _prep(y2)
            self.plot(x2, np.array(y2),
                      pen=pyqtgraph.mkPen((255, 100, 200), width=1),
                      symbol="o",
                      symbolSize=4,
                      symbolBrush=(255, 100, 200),
                      symbolPen=None,
                      name="SPCM2")

        # SPCM3 (if provided) -- the new detector being aligned
        if y3 is not None and len(y3):
            x3, y3 = _prep(y3)
            self.plot(x3, np.array(y3),
                      pen=pyqtgraph.mkPen((80, 220, 100), width=1),
                      symbol="o",
                      symbolSize=4,
                      symbolBrush=(80, 220, 100),
                      symbolPen=None,
                      name="SPCM3")

        self.setTitle(title)

        # Pin the y-axis floor. SPCM counts are non-negative, so we ALWAYS anchor
        # the bottom (default 0) and let the top auto-scale — otherwise pyqtgraph
        # auto-ranges and the floor "floats up" to ~100 when the atom is bright.
        # --ymin overrides the floor if the argument reaches the applet; if it
        # doesn't (older ARTIQ), the 0 default still applies. Top tracks the data.
        ymin_val = getattr(self.args, "ymin", None)
        if ymin_val is None:
            ymin_val = 0.0
        ymax = float(np.array(y1, dtype=float).max()) if len(y1) else 1.0
        if y2 is not None and len(y2):
            ymax = max(ymax, float(np.array(y2, dtype=float).max()))
        if y3 is not None and len(y3):
            ymax = max(ymax, float(np.array(y3, dtype=float).max()))
        top = ymax * 1.15 if ymax > 0 else 1.0

        # Sticky top: only grow, never shrink -> the y-window stops jumping
        # around as points scroll through the rolling buffer.
        if self._ytop is None or top > self._ytop:
            self._ytop = top
        top = self._ytop

        # Optional hard-fixed top via --ymax (fully static window with --ymin/--ymax).
        ymax_arg = getattr(self.args, "ymax", None)
        if ymax_arg is not None:
            top = float(ymax_arg)

        self.enableAutoRange(axis="y", enable=False)
        self.setYRange(float(ymin_val), float(top), padding=0)


def main():
    applet = TitleApplet(DualSPCMPlot)
    applet.add_dataset("y", "SPCM1 counts (Y values)")
    applet.add_dataset("y2", "SPCM2 counts", required=False)
    applet.add_dataset("y3", "SPCM3 counts", required=False)
    applet.add_dataset("pts", "Max points to display", required=False)
    applet.add_dataset("total", "Dataset holding the TOTAL shots recorded so "
                                "far -- absolute anchor that keeps --bin points "
                                "static once the plotted buffer starts sliding",
                       required=False)
    applet.argparser.add_argument("--ymin", type=float, default=None,
                                  help="Fix the y-axis minimum (default 0)")
    applet.argparser.add_argument("--ymax", type=float, default=None,
                                  help="Fix the y-axis maximum for a fully static window")
    applet.argparser.add_argument("--bin", type=int, default=1,
                                  help="Plot the MEAN of every N consecutive "
                                       "shots as one point (display only; "
                                       "default 1 = raw points)")
    applet.run()


if __name__ == "__main__":
    main()
