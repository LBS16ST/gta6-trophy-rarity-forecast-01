# GTA VI Trophy Rarity Forecast — gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/LBS16ST/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/LBS16ST/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Original work for the make.fun competition
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for each of Grand Theft Auto VI's 55 PlayStation trophies, the
percent of tracked players who will have earned it in week one — as five
quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## Program

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` in the input directory (`$MAKEFUN_INPUT_DIR`, default `/input`)
and writes `<id>.out` to the output directory (`$MAKEFUN_OUTPUT_DIR`,
default `/output`) with exactly one key, `trophies`: one row per trophy
in the case, each `{"id": n, "q": [q05, q25, q50, q75, q95]}` with
quantiles in percent, finite, within [0.01, 99.99], and non-decreasing.

## Method

No number is guessed. Every forecast is computed from the case file
itself, in logit space — the same scale the scorer uses for its pinball
loss:

- **Recent-window trend.** Per trophy, a recency-weighted linear trend
  of logit(rate) against time, fitted only over the most recent third
  of the history. Launch week has regimes: an early flood of new
  players dilutes most rates, then earning catches up as player growth
  slows. A trend fitted over the whole history would extrapolate the
  flood phase long after it ended; the recent window follows the
  current regime.
- **Pooled drift.** The median per-trophy slope is the common drift
  (mostly new-player dilution). Each trophy's slope is blended with
  the pooled slope, trusting its own slope more the more reads it has.
- **Damping.** The further the target lies beyond the recent window,
  the more the forecast reverts toward persistence (the last read):
  damp factor = window span / (window span + horizon).
- **Honest spread.** Quantile width comes from the *local* wiggle
  around the recent trend (scatter of successive read-to-read changes
  after removing the fitted slope, so a curved history cannot inflate
  it), plus slope uncertainty, plus a term that grows with the horizon.
  Quantiles are Normal quantiles on the logit scale, mapped back to
  percents and clamped to [0.01, 99.99].
- A trophy with no usable history falls back to persistence with the
  floor spread, so the output is always valid.

## Validation

On the kit's two practice cases the program runs in ~0.07 s and both
outputs are valid (55 rows, correct ids, quantiles in range and
non-decreasing, 2,683 bytes each — far under the 65,536-byte limit).

Honest hold-out check (last read of each practice case as pseudo-target,
forecast from the earlier reads, mean pinball loss on the logit scale):

| case | this model | pure persistence |
| --- | --- | --- |
| p00 hold-out | 0.0260 | 0.0000 |
| p01 hold-out | 0.0260 | 0.0000 |

The practice histories are perfectly flat (pre-launch placeholder reads:
638 tracked players, identical rates), so any spread is pure cost there
and persistence cannot be beaten on them — the 0.026 is the price of the
spread, not a trend error. On a synthetic case with realistic moving
rates (smooth logistic player growth, per-trophy earning, a regime
change from dilution to recovery), the same model beats persistence
0.056 vs 0.119 at a 30-minute horizon and 0.234 vs 0.549 at 150 minutes.
The real test is the final case, cut at the deadline with a five-day
horizon, where rates will actually move.

## License

MIT. See LICENSE.
