#!/usr/bin/env python3
"""GTA VI trophy-rarity forecaster for make.fun `gta6-trophy-rarity-forecast-01`.

Original work, written from the case files themselves. Python 3.13,
standard library only, no network.

Method
------
Each case gives, per trophy, a history of Exophase rarity reads
(percent of tracked players who earned the trophy) plus the number of
players tracked at each read. The forecast target is one read far in the
future, so the model works in logit space -- the same scale the scorer
uses for its pinball loss -- and extrapolates each trophy's own trend,
shrunk toward the cross-trophy (pooled) trend:

1. Per trophy, fit a recency-weighted linear trend of logit(rate) against
   time, centred on the last read -- but only over the most recent third
   of the history. Launch week has regimes: an early flood of new players
   dilutes most rates, then earning catches up as player growth slows.
   A trend fitted over the whole history extrapolates the flood phase
   long after it ended; the recent window follows the current regime.
2. Pool the slopes: the median per-trophy slope is the common drift
   (mostly new-player dilution -- as tracked players flood in, most
   rates fall together). Each trophy's slope is blended with the pooled
   slope, trusting its own slope more the more reads it has.
3. Damp the extrapolation. The further the target lies beyond the
   recent window, the more the forecast reverts toward persistence
   (the last read). Damp factor = window span / (window span + horizon).
4. Spread comes from the *local* wiggle around the recent trend (the
   scatter of successive read-to-read changes after removing the fitted
   slope -- a curved history cannot inflate it) plus slope uncertainty
   plus a term that grows with the horizon, so far targets get honestly
   wider quantiles. Quantiles are Normal quantiles on the logit scale,
   mapped back to percents and clamped to [0.01, 99.99].

Nothing is guessed: every number is computed from the case file. A
trophy with no usable history falls back to persistence with the floor
spread, so the output is always valid.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from statistics import NormalDist

# Quantile z-scores, computed once from the standard normal.
_NORM = NormalDist()
_Z = {0.05: -1.6448536269514722, 0.25: -0.6744897501960817,
      0.5: 0.0, 0.75: 0.6744897501960817, 0.95: 1.6448536269514722}

P_MIN = 0.01   # percent; also the scorer's clamp
P_MAX = 99.99
SIGMA_FLOOR = 0.12   # logit scale; covers read quantisation + drift
SIGMA_CAP = 1.2
MAX_LOGIT_MOVE = 3.0  # cap on |median - last logit| over the horizon


def _parse_time(s):
    """ISO-8601 timestamp -> aware datetime (UTC)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _logit(p):
    p = min(max(p, P_MIN), P_MAX) / 100.0
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _z(tau):
    if tau in _Z:
        return _Z[tau]
    return _NORM.inv_cdf(tau)


def _wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_std, se_b)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, 0.0)
    b = sxy / sxx
    a = my - b * mx
    n = len(xs)
    if n > 2:
        sse = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        dof = n - 2
        resid_var = sse / sw * n / dof if dof > 0 else 0.0
        resid_std = math.sqrt(max(resid_var, 0.0))
        se_b = math.sqrt(resid_var / sxx) if sxx > 0 else 0.0
    else:
        resid_std = 0.0
        se_b = 0.0
    return (a, b, resid_std, se_b)


def _median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def forecast_case(case):
    """Return the output dict for one case."""
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]

    at = _parse_time(case["at"])
    horizon_min = float(case.get("horizon_minutes") or 0.0)
    if horizon_min <= 0 and case.get("target"):
        horizon_min = (_parse_time(case["target"]) - at).total_seconds() / 60.0
    horizon_min = max(horizon_min, 0.0)

    # History points: (minutes relative to `at`, rates list).
    reads = []
    for r in history:
        try:
            t = (_parse_time(r["at"]) - at).total_seconds() / 60.0
        except Exception:
            continue
        rates = r.get("rates")
        if not isinstance(rates, list):
            continue
        reads.append((t, rates))
    reads.sort(key=lambda x: x[0])

    if reads:
        t_last = reads[-1][0]
        t_first = reads[0][0]
        span_min = max(t_last - t_first, 0.0)
    else:
        t_last = 0.0
        span_min = 0.0

    # Recent window: the last third of the history (at least 90 minutes,
    # at least the last 3 reads). Regimes change during launch week, so
    # only the current regime informs the trend; older reads are ignored.
    win_start = t_last - max(span_min / 3.0, 90.0)
    win_span = max(t_last - win_start, 0.0)
    half_life = max(win_span / 2.0, 30.0)
    damp = win_span / (win_span + horizon_min) if (win_span + horizon_min) > 0 else 0.0

    # First pass: per-trophy fits over the recent window, to build the
    # pooled slope.
    fits = []  # per trophy: dict(last_p, a, b, wiggle, se_b, n)
    slopes = []
    for j, tr in enumerate(trophies):
        pts = []
        for (t, rates) in reads:
            if j < len(rates) and rates[j] is not None:
                try:
                    p = float(rates[j])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(p):
                    pts.append((t, p))
        if not pts:
            fits.append(None)
            continue
        last_p = pts[-1][1]
        win_pts = [(t, p) for (t, p) in pts if t >= win_start]
        if len(win_pts) < 3:
            win_pts = pts
        if len(win_pts) < 2:
            fits.append({"last_p": last_p, "a": _logit(last_p), "b": 0.0,
                         "wiggle": 0.0, "se_b": 0.0, "n": len(win_pts)})
            continue
        xs = [t - t_last for (t, _p) in win_pts]
        ys = [_logit(p) for (_t, p) in win_pts]
        ws = [0.5 ** ((t_last - t) / half_life) for (t, _p) in win_pts]
        fit = _wls_line(xs, ys, ws)
        if fit is None:
            fits.append({"last_p": last_p, "a": _logit(last_p), "b": 0.0,
                         "wiggle": 0.0, "se_b": 0.0, "n": len(win_pts)})
            continue
        a, b, _resid, se_b = fit
        # Local wiggle: scatter of successive changes after removing the
        # fitted slope. Local by construction, so a bent trajectory does
        # not inflate the spread.
        diffs = []
        for k in range(1, len(win_pts)):
            dt = win_pts[k][0] - win_pts[k - 1][0]
            if dt <= 0:
                continue
            diffs.append(_logit(win_pts[k][1]) - _logit(win_pts[k - 1][1])
                         - b * dt)
        if len(diffs) >= 2:
            mean_d = sum(diffs) / len(diffs)
            wiggle = math.sqrt(sum((d - mean_d) ** 2 for d in diffs)
                               / (len(diffs) - 1))
        else:
            wiggle = 0.0
        fits.append({"last_p": last_p, "a": a, "b": b, "wiggle": wiggle,
                     "se_b": se_b, "n": len(win_pts)})
        # Only trophies with real movement inform the pooled drift.
        if len(win_pts) >= 3 and wiggle > 0.0:
            slopes.append(b)
    pooled_slope = _median(slopes) if slopes else 0.0

    horizon_h = horizon_min / 60.0
    rows = []
    for j, tr in enumerate(trophies):
        fit = fits[j]
        if fit is None:
            # No history at all for this trophy: centre on 1% with a
            # wide spread rather than inventing a rate.
            med_logit = _logit(1.0)
            sigma = SIGMA_CAP
        else:
            last_logit = _logit(fit["last_p"])
            # Blend own slope with the pooled drift; more reads -> more
            # trust in the trophy's own slope.
            reliability = fit["n"] / (fit["n"] + 3.0)
            b_blend = reliability * fit["b"] + (1.0 - reliability) * pooled_slope
            move = b_blend * damp * horizon_min
            move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, move))
            med_logit = fit["a"] + move
            # Spread: local wiggle around the recent trend (a *local*
            # noise estimate, so a curved history cannot inflate it)
            # + slope uncertainty over the extrapolated distance + a
            # term that grows with the horizon + floor.
            wiggle = min(max(fit["wiggle"], 0.03), 0.35)
            var = (wiggle ** 2
                   + (fit["se_b"] * damp * horizon_min) ** 2
                   + (0.06 * math.sqrt(horizon_h)) ** 2
                   + SIGMA_FLOOR ** 2)
            sigma = min(math.sqrt(var), SIGMA_CAP)
            # A perfectly flat history still moves in the real world:
            # keep at least the floor spread (already inside `var`).
            _ = last_logit
        qs = []
        prev = -1.0
        for tau in levels:
            q = 100.0 * _sigmoid(med_logit + _z(float(tau)) * sigma)
            q = min(max(q, P_MIN), P_MAX)
            q = max(q, prev)  # enforce non-decreasing quantiles
            prev = q
            qs.append(round(q, 4))
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def process_file(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        case = json.load(f)
    out = forecast_case(case)
    data = json.dumps(out, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(data) > 65536:
        raise ValueError("output exceeds 65536 bytes: %d" % len(data))
    with open(out_path, "wb") as f:
        f.write(data)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        print("no .in files in %s" % in_dir, file=sys.stderr)
        return 1
    failed = 0
    for name in names:
        in_path = os.path.join(in_dir, name)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        try:
            process_file(in_path, out_path)
            print("wrote %s" % out_path)
        except Exception as exc:  # never crash the whole run on one case
            print("error on %s: %s" % (name, exc), file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
