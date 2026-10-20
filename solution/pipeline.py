"""Data in, submission out.

    python -m solution.pipeline                 # writes evaluation/results/team_asare.csv

Prints aggregate counts only: possessions, durations, label shares.
"""

import argparse
import os

import numpy as np

from games import GAME, GAMES, half_video_blocks, source_frames
from solution import features, load, model

OUT = os.path.join(load.ROOT, "evaluation", "results", "team_asare.csv")
LETTERS = np.array(["H", "A", "D"])

# The second half is graded from 110808 but kicks off at 113742 (FAQ.md): everything before the
# kickoff is a dead ball by definition. Published boundary, not a label.
KICKOFFS = {1: 11185, 2: 113742}


def predict(params=None, game=GAME):
    params = dict(model.DEFAULTS, **(params or {}))
    trk = load.load_tracking(game)
    f = features.compute(trk, GAMES[game]["pitch"])
    e = model.evidence(f, load.load_tags(game), params)

    labels = np.full(source_frames("tactical", game), "X", dtype="<U1")
    for half, first, last in half_video_blocks(game):
        kickoff = KICKOFFS[half]
        labels[first:kickoff] = "D"
        labels[kickoff:last + 1] = LETTERS[model.viterbi(e[kickoff:last + 1], params)]
    return labels


def summary(labels, game=GAME):
    spells, dead = [], []
    for _, first, last in half_video_blocks(game):
        seg = labels[first:last + 1]
        cut = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(seg)]):
            (dead if seg[a] == "D" else spells).append(b - a)
    graded = labels != "X"
    return ("graded {}  H {:.1%}  A {:.1%}  D {:.1%}  |  possessions {}  mean {:.1f}s  "
            "median {:.1f}s  |  dead stretches {}  mean {:.1f}s").format(
        graded.sum(), *(np.mean(labels[graded] == c) for c in "HAD"),
        len(spells), np.mean(spells) / 25, np.median(spells) / 25, len(dead), np.mean(dead) / 25)


def write(labels, path=OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    labels = predict()
    print(summary(labels))
    print("wrote", os.path.relpath(write(labels, args.out), load.ROOT))
