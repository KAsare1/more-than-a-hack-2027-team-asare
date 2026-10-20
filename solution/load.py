"""Read the tactical ATD and Smart Tagging into numpy arrays, cached under data/cache/.

The cache holds frame-level tracking, so it lives inside data/ -- gitignored, never published, the
same status as the raw files it was built from.

Nothing in this module prints data values. Errors name a line number, never a line's content.
"""

import difflib
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402

CACHE_VERSION = 1
HOME, AWAY = 0, 1


def cache_path(game=GAME):
    return os.path.join(DATA_DIR, "cache", "{}_tactical_atd_v{}.npz".format(game, CACHE_VERSION))


def team_of_each_track(meta_path):
    """0 for a home slot, 1 for away. First team in the metadata is home (FIFA/EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in the ATD metadata, found {}".format(len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _parse_raw(raw_path, n_frames, n_slots):
    """Stream "<frame>:<p1>;<p2>;...:<ball>" into arrays. NaN where the tracker has nothing."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)   # x, y, speed (0-1, m/s)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)              # x, y (0-1)

    with open(raw_path) as handle:
        for lineno, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                raise ValueError("ATD line {}: expected 3 ':'-separated fields".format(lineno))
            row = int(parts[0]) - 1                     # frame_id counts from 1, video from 0
            if not 0 <= row < n_frames:
                raise ValueError("ATD line {}: frame id out of range".format(lineno))

            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                if group and group[0] != "N":
                    values = group.split(",")
                    players[row, slot, 0] = float(values[0])
                    players[row, slot, 1] = float(values[1])
                    if len(values) > 2 and values[2] and values[2][0] != "N":
                        players[row, slot, 2] = float(values[2])

            if parts[2] and parts[2][0] != "N":
                values = parts[2].split(",")
                ball[row, 0] = float(values[0])
                ball[row, 1] = float(values[1])
    return players, ball


def load_tracking(game=GAME, rebuild=False):
    """dict: players (N,S,3), ball (N,2), team (S,), all coordinates in metres."""
    path = cache_path(game)
    if os.path.isfile(path) and not rebuild:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    team = team_of_each_track(meta_path)
    n_frames = source_frames("tactical", game)
    print("parsing tracking (first run only, ~1-2 min) ...")
    players, ball = _parse_raw(raw_path, n_frames, len(team))

    length, width = GAMES[game]["pitch"]
    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, players=players, ball=ball, team=team)
    return {"players": players, "ball": ball, "team": team}


def load_set_pieces(game=GAME):
    """Smart Tagging SET PIECES as [(start_frame, end_frame, team_code_or_None, type)]."""
    return [(s, e, t, ty) for code, s, e, t, ty in load_tags(game) if code == "set pieces"]


def load_tags(game=GAME):
    """Every Smart Tagging instance as (code, start_frame, end_frame, team_code_or_None, type).

    team code: 0 home, 1 away, None for N/A (every throw-in). Home is the club named first in the
    ATD metadata; Smart Tagging spells club names its own way ("Atletic Club"), so the match is fuzzy.
    """
    fps = GAMES[game]["fps"]
    meta_path, tag_path = require(["atd_meta", "smart_tagging"], game)
    clubs = [(t.findtext("Name") or "").strip().lower() for t in ET.parse(meta_path).getroot().iter("Team")]

    def team_code(name):
        name = (name or "").strip().lower()
        if not name or name == "n/a":
            return None
        scores = [difflib.SequenceMatcher(None, name, club).ratio() for club in clubs]
        return int(np.argmax(scores)) if max(scores) >= 0.5 else None

    out = []
    for inst in ET.parse(tag_path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        out.append(((inst.findtext("code") or "").strip().lower(),
                    int(round(float(inst.findtext("start")) * fps)),
                    int(round(float(inst.findtext("end")) * fps)),
                    team_code(tags.get("Team")),
                    (tags.get("Type") or "").strip().lower()))
    return out
