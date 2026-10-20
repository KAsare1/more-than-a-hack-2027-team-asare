"""Per-frame features from the tracking. Everything is a length-N array over video frames."""

import warnings

import numpy as np

FPS = 25
OUT_MARGIN_M = 1.0   # tracking noise on a ball sitting on the line; the whole ball must cross


def _nan_smooth(x, half):
    """Centred moving mean over +/- half frames, ignoring NaN. NaN only where the window is empty."""
    valid = np.isfinite(x)
    filled = np.where(valid, x, 0.0)
    c = np.concatenate(([0.0], np.cumsum(filled)))
    n = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    count = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (c[hi] - c[lo]) / np.maximum(count, 1), np.nan)


def _run_lengths(mask):
    """For each frame, the length of the run of equal mask values it sits in."""
    n = len(mask)
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    return np.repeat(ends - starts, ends - starts)


def compute(trk, pitch):
    # Frames with nobody tracked (halftime) make nanmin/nanmean warn; NaN is the right answer there.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _compute(trk, pitch)


def _compute(trk, pitch):
    players, ball, team = trk["players"], trk["ball"], trk["team"]
    length, width = pitch
    n = len(ball)

    vis = np.isfinite(ball[:, 0])
    bx, by = ball[:, 0], ball[:, 1]
    inside = (bx > -OUT_MARGIN_M) & (bx < length + OUT_MARGIN_M) & \
             (by > -OUT_MARGIN_M) & (by < width + OUT_MARGIN_M)
    out = vis & ~inside

    dist = np.hypot(players[:, :, 0] - bx[:, None], players[:, :, 1] - by[:, None])
    with np.errstate(all="ignore"):
        d_home = np.nanmin(np.where(team[None, :] == 0, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(team[None, :] == 1, dist, np.nan), axis=1)

    # Ball speed from consecutive visible positions, m/s.
    step = np.full(n, np.nan)
    step[1:] = np.hypot(np.diff(bx), np.diff(by)) * FPS
    ball_speed = _nan_smooth(step, 2)

    # Collective player speed: the tracker's own speed channel, else finite differences.
    speed = players[:, :, 2].astype(np.float64)
    if not np.isfinite(speed).any():
        speed = np.full(players.shape[:2], np.nan)
        speed[1:] = np.hypot(np.diff(players[:, :, 0], axis=0),
                             np.diff(players[:, :, 1], axis=0)) * FPS
    with np.errstate(all="ignore"):
        mean_speed = np.nanmean(speed, axis=1)
    team_speed = _nan_smooth(mean_speed, FPS)          # +/- 1 s

    gap_len = np.where(vis, 0, _run_lengths(vis))       # length of the ball-missing run, frames

    # Distance to the nearest touchline or goal line of the ball as last seen before, and as first
    # seen after, the current frame. Through a gap these say where the ball vanished and where it
    # came back -- a ball that reappears on a line is a throw-in, a corner or a goal kick.
    edge = np.fmin(np.fmin(bx, length - bx), np.fmin(by, width - by))
    frame = np.arange(n)
    last = np.maximum.accumulate(np.where(vis, frame, -1))
    nxt = np.minimum.accumulate(np.where(vis, frame, n)[::-1])[::-1]
    last_edge = np.where(last >= 0, edge[np.clip(last, 0, n - 1)], np.nan)
    next_edge = np.where(nxt < n, edge[np.clip(nxt, 0, n - 1)], np.nan)

    carried, carried_age = _last_control(vis & ~out, d_home, d_away)

    return {
        "carried": carried, "carried_age": carried_age,
        "vis": vis, "out": out, "d_home": d_home, "d_away": d_away,
        "ball_speed": ball_speed, "team_speed": team_speed, "gap_len": gap_len,
        "last_edge": last_edge, "next_edge": next_edge,
        "n_players": np.isfinite(players[:, :, 0]).sum(axis=1),
    }


CONTROL_RADIUS_M = 1.2
CONTROL_MIN_FRAMES = 8


def _last_control(usable, d_home, d_away):
    """Team of the most recent control episode, and frames since it ended.

    A control episode is a run of at least CONTROL_MIN_FRAMES where the ball sits within
    CONTROL_RADIUS_M of a player and that player's team is the closer one. The ball track is
    smooth enough that kicks do not show up as velocity spikes, but a ball staying at a player's
    feet does show up -- so control is read from dwell time, not from impacts.

    Carrying the last episode's team forward covers passes in flight and gaps in the ball track,
    where the nearest player is often an opponent or there is no ball at all. carried is -1 before
    the first episode; during an episode it is that episode's team with age 0.
    """
    n = len(usable)
    end_team = np.full(n, -1, dtype=np.int8)
    in_episode = np.full(n, -1, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        for team, own, other in ((0, d_home, d_away), (1, d_away, d_home)):
            mask = usable & (own < CONTROL_RADIUS_M) & (own < other)
            edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
            for start, end in zip(edges[0::2], edges[1::2]):
                if end - start >= CONTROL_MIN_FRAMES:
                    in_episode[start:end] = team
                    end_team[end - 1] = team
    frame = np.arange(n)
    marked = (end_team >= 0) | (in_episode >= 0)
    last = np.maximum.accumulate(np.where(marked, frame, -1))
    source = np.where(in_episode >= 0, in_episode, end_team)
    carried = np.where(last >= 0, source[np.clip(last, 0, n - 1)], -1)
    age = np.where(last >= 0, frame - last, n)
    return carried, age
