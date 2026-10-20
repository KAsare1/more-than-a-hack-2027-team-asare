"""Aggregate feature statistics against the public 5-minute ground truth.

Prints quantiles and rates only -- never a frame-level value -- so its output is safe to share.

    python -m solution.diagnostics
"""

import os

import numpy as np
import pandas as pd

from games import GAME, GAMES
from solution import features, load

GT = os.path.join(load.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def q(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return "n/a"
    return "p10 {:6.2f}  p50 {:6.2f}  p90 {:6.2f}".format(*np.percentile(x, [10, 50, 90]))


def main():
    f = features.compute(load.load_tracking(), GAMES[GAME]["pitch"])
    gt = pd.read_csv(GT)["label"].values
    for lab in "HAD":
        m = gt == lab
        print("GT {}  n={:5d}  ball visible {:4.0%}  out {:4.0%}".format(
            lab, m.sum(), f["vis"][m].mean(), f["out"][m].mean()))
        print("    team_speed  ", q(f["team_speed"][m]))
        print("    ball_speed  ", q(f["ball_speed"][m]))
        print("    min dist    ", q(np.fmin(f["d_home"], f["d_away"])[m]))
        print("    gap_len (s) ", q(f["gap_len"][m & ~f["vis"]] / 25.0))
    for lab, own, other in (("H", "d_home", "d_away"), ("A", "d_away", "d_home")):
        m = (gt == lab) & f["vis"]
        print("GT {}: nearest player is own team on {:.0%} of visible frames".format(
            lab, np.mean(f[own][m] < f[other][m])))
    graded = gt != "X"
    print("whole match: ball visible {:.0%}, team_speed {}".format(f["vis"].mean(), q(f["team_speed"])))
    print("set pieces tagged: {}  with team: {}".format(
        len(load.load_set_pieces()), sum(sp[2] is not None for sp in load.load_set_pieces())))
    print("graded public frames {}".format(graded.sum()))


if __name__ == "__main__":
    main()
