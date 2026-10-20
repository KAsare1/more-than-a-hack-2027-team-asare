"""A full-match proxy check, built from Smart Tagging phases the model does not use.

The public ground truth covers 17 possessions, too few to tell two good submissions apart. Some
Smart Tagging phases name the team in possession almost perfectly on that window (Build Up and
Creating Chances 100%, Long Ball 90%) and occur across the whole match. The model never reads
them, so they work as independent spot checks.

For each such tag, the proportion of the submission's in-play frames inside it that carry the
tagged team:
    agree  mean of that proportion over tags       -- is the team right?
    clean  share of tags where it is >= 90%        -- is the possession unbroken through it?

`clean` is the one that should track spellF1: a submission that flickers inside a phase fails it
even when its average team is right. It is a proxy, not the score: the tags are automatic and
imperfect, and they say nothing about dead balls.

    python -m solution.yardstick                     # every CSV in evaluation/results/
    python -m solution.yardstick path/to/file.csv

Prints aggregates only.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from solution import load

CHECK_PHASES = ("build up", "creating chances", "long ball")
CLEAN = 0.9
PUBLIC = (11185, 18685)


def check_tags():
    return [(s, e, t) for code, s, e, t, _ in load.load_tags() if code in CHECK_PHASES and t is not None]


def score(labels, tags, exclude_public=False):
    props = []
    for s, e, team in tags:
        if exclude_public and s < PUBLIC[1] and e > PUBLIC[0]:
            continue
        seg = labels[s:e]
        alive = seg[(seg == "H") | (seg == "A")]
        if len(alive) < 25:          # the submission calls (almost) the whole phase dead
            continue
        props.append(np.mean(alive == "HA"[team]))
    props = np.array(props)
    return len(props), props.mean(), np.mean(props >= CLEAN)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    files = args.files or sorted(glob.glob(os.path.join(load.ROOT, "evaluation", "results", "*.csv")))
    tags = check_tags()
    print("{} check tags ({}), {} outside the public window".format(
        len(tags), ", ".join(CHECK_PHASES),
        sum(1 for s, e, _ in tags if not (s < PUBLIC[1] and e > PUBLIC[0]))))
    print("{:<26} {:>5} {:>7} {:>7}   {:>7} {:>7}".format(
        "submission", "tags", "agree", "clean", "agree*", "clean*"))
    for path in files:
        labels = pd.read_csv(path)["label"].astype(str).str.strip().str.upper().values
        n, agree, clean = score(labels, tags)
        _, agree_x, clean_x = score(labels, tags, exclude_public=True)
        print("{:<26} {:>5} {:>7.3f} {:>7.3f}   {:>7.3f} {:>7.3f}".format(
            os.path.basename(path)[:-4], n, agree, clean, agree_x, clean_x))
    print("* = excluding tags that overlap the public 5-minute window")


if __name__ == "__main__":
    main()
