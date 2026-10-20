# Team Asare — ball possession from automatic tracking

More than a Hack 2027, Phase 1. Draft technical document.

## Approach

Per-frame evidence for three states — home in possession (`H`), away in possession (`A`), dead ball
(`D`) — decoded with a Viterbi pass that charges a fixed cost for every change of state.

That cost is the core idea. The ranking metric scores whole possessions, and per-frame guesses
flicker: the nearest player to the ball belongs to the possessing team only ~78% of the time.
With a switching cost, a change of team has to be backed by enough consistent evidence to pay for
it. A touch, a deflection or a pass in flight does not change possession, and a stretch with the
ball out of view keeps the current state, because it scores the same for every state. That is the
challenge's "loose ball stays with the last team" rule, falling out of the model rather than
being coded as a special case.

We chose this over a majority-vote window, as in the baseline, because a window delays every
transition by about half its width. It also can't tell a contested ball from a hand-over. The
decoder places each change where the evidence turns.

## Evidence

**Team (H vs A)**, only when the ball is visible and on the pitch:

- distance from the ball to the nearest home and nearest away player, in metres;
- lean = `3.2 · exp(−d_min / 2.5) · tanh((d_away − d_home) / 1.5)`. This roughly follows the
  measured log-odds that the nearest player's team owns the ball: 96% within 0.5 m, 91% within
  1 m, 63% beyond 4 m. The `tanh` term makes a ball both teams are equally close to count for
  little;
- scaled by 0.3 when the ball moves faster than 12 m/s (in flight, 60% reliable);
- **last team in control, carried forward.** A *control episode* is when the ball stays within
  1.2 m of a player, closer to that player's team than to the other, for at least 8 frames. There
  are about 8 a minute, and they name the right team 90% of the time. The team of the most recent
  episode is carried through passes in flight and gaps in the ball track: 83–87% correct on
  in-play frames outside episodes, many of which otherwise have no evidence or misleading
  evidence. It decays with a 6 s time constant;
- inside a Smart Tagging set piece with a team tag, a small pull towards the restarting team.

Team evidence only decides H vs A. It is added as ±lean/2, and D receives log cosh(lean/2). That
makes it neutral on alive vs dead, which is left entirely to the dead-ball evidence. Without this
correction, stronger team evidence quietly voted against dead balls.

**Dead ball (D vs alive)**, as a log-odds:

- collective player speed, the mean of the tracker's speed channel smoothed over ±1 s. Players walk
  during stoppages (median 1.2 m/s vs 2.0–2.4 m/s in play);
- ball outside the pitch by more than 1 m;
- the ball missing for over 1 s and reappearing within 3 m of a line: a throw-in, corner or goal
  kick. Strong. A weak version for a ball that vanished near a line;
- the 5 s before a Smart Tagging set piece starts, and weakly the set-piece interval itself.

**Fixed boundaries.** Ungraded frames are `X`. The second half is graded from frame 110,808, but
kickoff is at 113,742 (published in FAQ.md), so that stretch is `D`.

## Tried and dropped

- **Smart Tagging attacking phases as a team prior** (Build Up, Progression, Creating Chances,
  Long Ball, Attacking Transition). They agree with the public ground truth 73–100% of the time,
  but they span several possessions. As a prior they merged short possessions, and spellF1 fell
  from 0.48 to 0.43.
- **The box / final-third tags.** They agree with the possessing team only about 50% of the time.
- **A stationary visible ball as dead-ball evidence.** It was 14% precise.
- **Touches from changes in ball velocity.** The ball track is smooth. Its sharp velocity changes
  sit a median 4.6 m from the nearest player, so they are tracking artefacts, not kicks. Control
  episodes, which read control from dwell time, replaced them.
- **Snapping boundaries to events after decoding.** On the public window, many errors are the right
  change placed 1–2 s off, in both directions. Moving each H↔A change to the nearest start of a
  close contact (within 2 m) by the incoming team made spellF1 *worse* (0.491 → 0.457–0.480). From
  the decoded boundary, the nearest contact is often the wrong one. Snapping to the new team's
  *decisive* contact (the first after which the old team does not touch the ball again nearby) was
  neutral. Restart and stoppage anchors (ball starting to move, ball leaving play) were too
  inconsistent across the public window's four examples of each to try. Dropped.
- **Not used: the videos, the homography, the broadcast recording.** The tracking is already in
  pitch coordinates, and the broadcast feed is not synced to the tactical one.

## Choosing parameters

The public window holds 17 possessions, and its bootstrap noise band on spellF1 is about ±0.33.
So parameters were **not** chosen by maximising the public score. The evidence weights come from
measured rates (above). The switching costs were set in the middle of the range where the public
score is flat (switch cost 8–25 all score 0.46–0.49). Full-match outputs were checked for
plausibility: 309 possessions, mean 10.9 s; 126 stoppages; 47% of graded time dead, consistent
with the ~55 minutes of ball-in-play typical of a top-flight match.

| on the public 5 minutes | macroF1 | spellF1 | possessions (truth: 17) |
|---|---|---|---|
| team_asare v2 (control episodes) | 0.837 | 0.491 | 18 |
| team_asare v1 | 0.835 | 0.480 | 18 |
| baseline_smooth50 (the published floor) | 0.621 | 0.326 | 26 |

## Limitations and next steps

- Short possessions (under ~2 s) are mostly missed: the switching cost absorbs them.
- Team evidence comes from the nearest player and control episodes. Ideas: use the player's
  velocity relative to the ball, and track pressure from opponents.
- The ball is missing on 41% of frames. Interpolating short gaps and reasoning about which team
  surrounds the area where the ball was lost would help.
- Restart team: who takes a throw-in is unknown (tagged N/A). Which side of the pitch the ball left
  from, and who last touched it, could supply it.

## Reproduction

Python 3.14.0, `pip install -r requirements-solution.txt`, then put the tactical files
(`*_atd.txt`, `*_atd.xml`, `*_smart_tagging.xml`) in `data/tactical/`, then:

```bash
python -m solution.pipeline        # -> evaluation/results/team_asare.csv
python evaluation/validate_submission.py evaluation/results/team_asare.csv
```

Deterministic. Verified to give a byte-identical file, both from a warm cache and from the raw
files. The first run parses the tracking into `data/cache/` (gitignored).

## Tools

We used an AI coding assistant (Claude, by Anthropic) as a development tool, for drafting code and
documentation. As the competition terms require, no match data or extracts of it were given to
it. It saw only the published documentation, the public 5-minute ground truth and aggregate
results such as scores and rates. The pipeline itself uses no AI model: it is the deterministic
numpy code in this folder.
