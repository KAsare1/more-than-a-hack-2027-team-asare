"""Per-frame evidence -> Viterbi decoding over three states, H / A / D.

Each frame gets a score for each state (a log-likelihood up to a constant). The decoder picks the
label sequence maximising the summed score minus a fixed cost for every change of state. That cost
is what turns a flickering per-frame guess into possessions: a change of team has to be backed by
enough consistent evidence to pay for it, a touch or a deflection does not, and a stretch with no
ball in view simply holds the current state because it scores the same for every state.
"""

import numpy as np

H, A, D = 0, 1, 2

DEFAULTS = {
    # team evidence (ball visible and on the pitch)
    # Shaped on the public window: the nearest player's team is the owner 96% of the time within
    # 0.5 m, 91% within 1 m, 82% within 4 m, 63% beyond, and 60% with the ball in flight. The
    # lean below roughly follows those log-odds, 3.2 * exp(-d / 2.5).
    "team_weight": 1.0,
    "team_logodds": 3.2,
    "control_decay_m": 2.5,
    "margin_scale_m": 1.5,    # tanh scale of |d_away - d_home|: a contested ball says little
    "flight_speed": 12.0,     # m/s; faster than this the ball is in flight, not controlled
    "flight_factor": 0.3,
    # dead-ball evidence, a log-odds of dead vs alive
    "dead_bias": 0.0,
    "out_of_play": 4.0,
    "slow_speed": 1.5,        # m/s collective speed where dead and alive are equally likely
    "slow_gain": 2.5,
    "edge_m": 3.0,            # a ball lost or found this close to a line may have left play
    "min_gap": 25,            # frames; shorter gaps are ordinary occlusion
    "gap_returns_at_edge": 1.5,
    "gap_left_at_edge": 0.3,
    "before_set_piece_s": 5.0,
    "before_set_piece": 1.0,
    "in_set_piece": 0.3,
    "carry_weight": 1.0,      # last control episode's team, carried through passes and gaps
    "carry_decay_s": 6.0,
    "set_piece_team": 0.5,    # pull towards the tagged restarting team inside a set-piece tag
    "phase_team": 0.0,        # off: phases span several possessions and merged short ones
    # decoder costs
    "switch_cost": 20.0,     # H <-> A
    "to_dead_cost": 15.0,
    "from_dead_cost": 15.0,
}


# Smart Tagging phases whose Team is the side in possession. Their defending mirrors carry the same
# information, so they are not counted twice; the box / final-third tags disagree with the public
# ground truth about half the time and are left out.
ATTACKING_PHASES = ("build up", "progression", "creating chances", "long ball", "attacking transition")


def evidence(f, tags, params):
    """(N, 3) per-frame state scores."""
    p = params
    n = len(f["vis"])
    e = np.zeros((n, 3))

    d_home, d_away = f["d_home"], f["d_away"]
    usable = f["vis"] & ~f["out"] & np.isfinite(d_home) & np.isfinite(d_away)
    dmin = np.fmin(d_home, d_away)
    conf = p["team_logodds"] * np.exp(-dmin / p["control_decay_m"])
    conf = np.where(f["ball_speed"] > p["flight_speed"], conf * p["flight_factor"], conf)
    lean = np.tanh((d_away - d_home) / p["margin_scale_m"]) * conf * p["team_weight"]
    lean = np.where(usable, lean, 0.0)

    has = f["carried"] >= 0
    carry = p["carry_weight"] * np.exp(-f["carried_age"] / (p["carry_decay_s"] * 25.0)) * has
    lean = lean + np.where(f["carried"] == 0, 1.0, -1.0) * carry

    # Team evidence says which team, not whether the ball is alive. Adding it as +/- lean/2 would
    # raise the better alive state and quietly vote against D, so D gets the log-mean of the two
    # alive scores, log cosh(lean/2), which leaves alive-vs-dead exactly where the dead evidence
    # put it.
    e[:, H] += lean / 2
    e[:, A] -= lean / 2
    e[:, D] += np.logaddexp(lean / 2, -lean / 2) - np.log(2.0)

    speed = np.nan_to_num(f["team_speed"], nan=p["slow_speed"])
    dead = p["dead_bias"] + p["slow_gain"] * (p["slow_speed"] - speed)
    dead += p["out_of_play"] * f["out"]
    long_gap = ~f["vis"] & (f["gap_len"] > p["min_gap"])
    dead += p["gap_returns_at_edge"] * (long_gap & (np.nan_to_num(f["next_edge"], nan=99) < p["edge_m"]))
    dead += p["gap_left_at_edge"] * (long_gap & (np.nan_to_num(f["last_edge"], nan=99) < p["edge_m"]))
    e[:, D] += dead

    lead = int(round(p["before_set_piece_s"] * 25))
    for code, start, end, team, _ in tags:
        lo, hi = max(0, start), min(n, end)
        if code == "set pieces":
            e[max(0, start - lead):lo, D] += p["before_set_piece"]
            e[lo:hi, D] += p["in_set_piece"]
            pull = p["set_piece_team"]
        elif code in ATTACKING_PHASES:
            pull = p["phase_team"]
        else:
            continue
        if team is not None:
            e[lo:hi, team] += pull / 2
            e[lo:hi, 1 - team] -= pull / 2
    return e


def viterbi(e, params):
    """Best H/A/D path through e under the transition costs."""
    p = params
    cost = np.array([[0.0, p["switch_cost"], p["to_dead_cost"]],
                     [p["switch_cost"], 0.0, p["to_dead_cost"]],
                     [p["from_dead_cost"], p["from_dead_cost"], 0.0]])
    n = len(e)
    back = np.empty((n, 3), dtype=np.int8)
    score = e[0].copy()
    for t in range(1, n):
        cand = score[:, None] - cost              # [prev, cur]
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + e[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
