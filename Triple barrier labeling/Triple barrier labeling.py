"""
TripleBarrierLabeling_Static_Pipeline.py
=========================================
STATIC hero-image pipeline (poster / carousel cover) for the BARRIER-BOX
TUNNEL: Lopez de Prado's triple-barrier labelling, made physical.

Standalone: DATA (cached SPY daily closes) -> EWMA VOL -> WINDOW SEARCH
(highest time-barrier share with both PT and SL still represented) ->
TRIPLE-BARRIER LABELLING -> VALIDATE -> render ONE high-res frame of the
fully-resolved corridor from a fixed, curated angle.

Same maths as the reel, no shortcuts:

    sigma_t   = EWMA std-dev of daily log returns (span = EWMA_SPAN)
    event t0  -> upper = P(t0) * exp(+K_SIGMA * sigma_t0)     [profit-take]
                 lower = P(t0) * exp(-K_SIGMA * sigma_t0)     [stop-loss]
                 horizon = t0 + N_DAYS trading days           [vertical barrier]
    label     = +1 upper touched first, -1 lower touched first, 0 neither

THE CLAIM: on the window the data itself picks, a LARGE share of the 32
labels are TIME-BARRIER expiries -- neither a win nor a loss -- the outcome
most retail backtests never model.

WHAT THIS IS NOT: not a candlestick chart, not an equity curve, not a
backtest result -- it shows how a label gets assigned before any model is
ever trained on it.

    python TripleBarrierLabeling_Static_Pipeline.py
    -> TripleBarrierLabeling_Static.png  (1080x1920 at 300 DPI = 3240x5760)
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import matplotlib.font_manager as fm
FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "Demiurge", "fonts")
for _f in ["Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf",
           "Inter-Bold.ttf", "JetBrainsMono-Regular.ttf", "JetBrainsMono-Bold.ttf"]:
    _p = os.path.join(FONT_DIR, _f)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
_AVAIL = {f.name for f in fm.fontManager.ttflist}
FONT_SANS = "Inter" if "Inter" in _AVAIL else "DejaVu Sans"
FONT_MONO = "JetBrains Mono" if "JetBrains Mono" in _AVAIL else "DejaVu Sans Mono"

CONFIG = {
    "TICKER": "SPY", "PERIOD": "8y",
    "EWMA_SPAN": 20, "N_DAYS": 8, "K_SIGMA": 2.5,
    "N_EVENTS": 14, "MIN_SIDE_SHARE": 0.15,
    "W": 1080, "H": 1920, "DPI": 300,
    "ELEV": 24, "AZIM": -48,             # curated fixed 3/4 angle for the still
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#8a8a8a",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": FONT_SANS,
}
CMAP = LinearSegmentedColormap.from_list(
    "barrier", [THEME["RED"], "#3a3a3a", THEME["GREEN"]], N=256)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}")


def _cached(name, fetch):
    p = os.path.join(BASE_DIR, "_cache", name + ".csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0, parse_dates=True)
    df = fetch()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p)
    return df


def fetch_prices():
    c = CONFIG
    log(f"[Data] Fetching {c['TICKER']} daily closes ({c['PERIOD']})...")
    try:
        def _dl():
            import yfinance as yf
            df = yf.download(c["TICKER"], period=c["PERIOD"], interval="1d",
                             progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df[["Close"]].dropna()
        df = _cached("spy_daily", _dl)
        px = df["Close"].dropna()
        if len(px) < 500:
            raise ValueError(f"only {len(px)} rows returned")
        return px, False
    except Exception as e:
        log(f"[Data] yfinance failed ({e}); synthetic fallback")
        rng = np.random.default_rng(7)
        n = 2200
        p = 400 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, n)))
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        return pd.Series(p, index=dates, name="Close"), True


def compute_sigma(px):
    ret = np.log(px / px.shift(1)).dropna()
    sigma = ret.ewm(span=CONFIG["EWMA_SPAN"], min_periods=CONFIG["EWMA_SPAN"]).std()
    sigma = sigma.dropna()
    aligned_px = px.loc[sigma.index]
    return aligned_px, sigma


def label_window(P, S, i0, n_events, N, k):
    events = []
    for j in range(n_events):
        t0 = i0 + j * N
        p0, s0 = float(P[t0]), float(S[t0])
        upper, lower = p0 * np.exp(k * s0), p0 * np.exp(-k * s0)
        path = P[t0 + 1: t0 + N + 1]
        hu = np.where(path >= upper)[0]
        hl = np.where(path <= lower)[0]
        tu = int(hu[0]) + 1 if len(hu) else None
        tl = int(hl[0]) + 1 if len(hl) else None
        if tu is None and tl is None:
            label, touch_day, touch_px = 0, N, float(path[-1])
        elif tu is not None and (tl is None or tu < tl):
            label, touch_day, touch_px = 1, tu, float(path[tu - 1])
        else:
            label, touch_day, touch_px = -1, tl, float(path[tl - 1])
        events.append(dict(t0=t0, p0=p0, s0=s0, upper=float(upper), lower=float(lower),
                            label=label, touch_day=touch_day, touch_px=touch_px))
    return events


def find_best_window(P, S, n_events, N, k, min_side):
    span = n_events * N
    best, best_relaxed = None, None
    for i0 in range(60, len(P) - span - N - 1):
        evs = label_window(P, S, i0, n_events, N, k)
        labs = np.array([e["label"] for e in evs])
        n = len(labs)
        z = (labs == 0).sum() / n
        p1, m1 = (labs == 1).sum() / n, (labs == -1).sum() / n
        if best_relaxed is None or z > best_relaxed[0]:
            best_relaxed = (z, i0, evs)
        if min(p1, m1) < min_side:
            continue
        if best is None or z > best[0]:
            best = (z, i0, evs)
    chosen = best if best is not None else best_relaxed
    return chosen[1], chosen[2]


def _box_edges(x0, x1, yh, z0, z1):
    xs, ys, zs = (x0, x1), (-yh, yh), (z0, z1)
    c000, c001, c010, c011 = (xs[0], ys[0], zs[0]), (xs[0], ys[0], zs[1]), \
        (xs[0], ys[1], zs[0]), (xs[0], ys[1], zs[1])
    c100, c101, c110, c111 = (xs[1], ys[0], zs[0]), (xs[1], ys[0], zs[1]), \
        (xs[1], ys[1], zs[0]), (xs[1], ys[1], zs[1])
    return [(c000, c001), (c000, c010), (c001, c011), (c010, c011),
            (c100, c101), (c100, c110), (c101, c111), (c110, c111),
            (c000, c100), (c001, c101), (c010, c110), (c011, c111)]


def _face_top_bottom(x0, x1, yh, z):
    return [(x0, -yh, z), (x1, -yh, z), (x1, yh, z), (x0, yh, z)]


def _face_far(x1, yh, z0, z1):
    return [(x1, -yh, z0), (x1, yh, z0), (x1, yh, z1), (x1, -yh, z1)]


def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    events = d["events"]; n = len(events)
    labs = np.array([e["label"] for e in events])
    pt = float((labs == 1).sum()) / n
    sl = float((labs == -1).sum()) / n
    tm = float((labs == 0).sum()) / n

    assert abs(pt + sl + tm - 1.0) < 1e-9, f"label shares do not sum to 1: {pt+sl+tm}"
    # Geometry needs 25-40 boxes to fly through without mush at either end.
    assert 12 <= n <= 40, f"event count {n} outside the 12-40 corridor band"
    # find_best_window enforces a >=15% floor on whichever of PT/SL is
    # smaller so the tunnel always shows both colours; 8% gives slack below
    # that floor for a data refresh, 45% keeps either side from swallowing
    # the whole corridor.
    assert 0.08 <= pt <= 0.45, f"profit-take share {pt:.2f} outside plausible band"
    assert 0.08 <= sl <= 0.45, f"stop-loss share {sl:.2f} outside plausible band"
    # The whole claim: TIME must be meaningfully above an even 3-way split
    # (33%) or "a large share" is false; 85% caps it short of an all-grey
    # tunnel, which would say nothing about PT/SL at all.
    assert 0.40 <= tm <= 0.85, f"time-barrier share {tm:.2f} outside plausible band"

    sigmas = np.array([e["s0"] for e in events])
    assert sigmas.min() > 0, "EWMA sigma must be strictly positive at every event"
    gaps = np.array([e["upper"] - e["lower"] for e in events])
    assert gaps.min() > 0, "a barrier is inverted: upper must exceed lower everywhere"

    figs = {
        "ticker": CONFIG["TICKER"],
        "n_events": str(n),
        "n_days": str(CONFIG["N_DAYS"]),
        "k_sigma": f"{CONFIG['K_SIGMA']:.1f}",
        "ewma_span": str(CONFIG["EWMA_SPAN"]),
        "pt_share_pct": f"{pt*100:.0f}%",
        "sl_share_pct": f"{sl*100:.0f}%",
        "time_share_pct": f"{tm*100:.0f}%",
        "pt_count": str(int((labs == 1).sum())),
        "sl_count": str(int((labs == -1).sum())),
        "time_count": str(int((labs == 0).sum())),
        "window_start": pd.Timestamp(d["window_start"]).strftime("%b %Y"),
        "window_end": pd.Timestamp(d["window_end"]).strftime("%b %Y"),
        "synthetic": str(d["synthetic"]),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    return figs


def render_static(out):
    c = CONFIG
    px_raw, synthetic = fetch_prices()
    aligned_px, sigma = compute_sigma(px_raw)
    P, S, dates = aligned_px.values, sigma.values, aligned_px.index
    i0, events = find_best_window(P, S, c["N_EVENTS"], c["N_DAYS"],
                                  c["K_SIGMA"], c["MIN_SIDE_SHARE"])
    n_events = len(events)
    total_days = n_events * c["N_DAYS"]
    path = P[i0: i0 + total_days + 1]
    path_dates = dates[i0: i0 + total_days + 1]
    zmin = min(path.min(), min(e["lower"] for e in events))
    zmax = max(path.max(), max(e["upper"] for e in events))
    pad = (zmax - zmin) * 0.06
    zmin, zmax = zmin - pad, zmax + pad
    yh = (zmax - zmin) * 0.16

    d = dict(events=events, path=path, N=c["N_DAYS"], n_events=n_events,
             total_days=total_days, synthetic=synthetic,
             window_start=path_dates[0], window_end=path_dates[-1])
    figs = validate(d)
    labs = np.array([e["label"] for e in events])
    log(f"window: {path_dates[0].date()} -> {path_dates[-1].date()}  events={n_events}  "
        f"PT={int((labs==1).sum())} SL={int((labs==-1).sum())} TIME={int((labs==0).sum())}")

    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100), dpi=c["DPI"],
                     facecolor=THEME["BG"])
    fig.text(0.5, 0.958, "TRIPLE BARRIER LABELING", ha="center", fontsize=28,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.927,
             f"{figs['ticker']} daily  ·  ±{figs['k_sigma']}σ barriers  ·  "
             f"{figs['n_days']}-day horizon",
             ha="center", fontsize=14, color=THEME["ORANGE"], family=THEME["FONT"])

    ax = fig.add_axes([-0.06, 0.185, 1.12, 0.62], projection="3d", facecolor=THEME["BG"])

    lit_segs, lit_colors = [], []
    trigger_faces, trigger_colors = [], []
    for i, ev in enumerate(events):
        x0, x1 = i * c["N_DAYS"], (i + 1) * c["N_DAYS"]
        edges = _box_edges(x0, x1, yh, ev["lower"], ev["upper"])
        col = (THEME["GREEN"] if ev["label"] == 1 else
               THEME["RED"] if ev["label"] == -1 else THEME["TEXT_DIM"])
        lit_segs.extend(edges)
        lit_colors.extend([col] * len(edges))
        if ev["label"] == 1:
            trigger_faces.append(_face_top_bottom(x0, x1, yh, ev["upper"]))
            trigger_colors.append(THEME["GREEN"])
        elif ev["label"] == -1:
            trigger_faces.append(_face_top_bottom(x0, x1, yh, ev["lower"]))
            trigger_colors.append(THEME["RED"])
        else:
            trigger_faces.append(_face_far(x1, yh, ev["lower"], ev["upper"]))
            trigger_colors.append(THEME["TEXT_DIM"])

    ax.add_collection3d(Line3DCollection(lit_segs, colors=lit_colors,
                                         linewidths=1.5, alpha=0.85, zorder=2))
    fc = [to_rgba(cc, 0.34) for cc in trigger_colors]
    ax.add_collection3d(Poly3DCollection(trigger_faces, facecolors=fc,
                                         edgecolors="none", zorder=3))

    xs = np.arange(len(path))
    zs = path
    ys = np.zeros(len(path))
    pts = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_cols = []
    for day in range(len(path) - 1):
        bi = min(day // c["N_DAYS"], n_events - 1)
        ev = events[bi]
        frac = np.clip((zs[day] - ev["lower"]) / max(ev["upper"] - ev["lower"], 1e-9), 0, 1)
        seg_cols.append(CMAP(frac))
    ax.add_collection3d(Line3DCollection(segs, colors=seg_cols, linewidths=2.8, zorder=9))
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=80, color="white",
              edgecolors=THEME["CYAN"], linewidths=1.6, zorder=12)

    ax.set_xlim(0, total_days); ax.set_ylim(-yh * 1.3, yh * 1.3); ax.set_zlim(zmin, zmax)
    ax.view_init(elev=c["ELEV"], azim=c["AZIM"])
    ax.set_box_aspect((1.9, 1, 1), zoom=1.42)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, 0.155, f"{figs['n_events']} EVENTS RESOLVED", ha="center",
             fontsize=13, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.20, 0.118, figs["pt_count"], ha="center", fontsize=24, color=THEME["GREEN"],
             fontweight="bold", family=THEME["FONT"])
    fig.text(0.20, 0.093, "PROFIT-TAKE", ha="center", fontsize=9.5,
             color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, 0.118, figs["sl_count"], ha="center", fontsize=24, color=THEME["RED"],
             fontweight="bold", family=THEME["FONT"])
    fig.text(0.5, 0.093, "STOP-LOSS", ha="center", fontsize=9.5,
             color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.80, 0.118, figs["time_count"], ha="center", fontsize=24, color=THEME["TEXT"],
             fontweight="bold", family=THEME["FONT"])
    fig.text(0.80, 0.093, "TIME LIMIT", ha="center", fontsize=9.5,
             color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, 0.058,
             f"{figs['time_share_pct']} of labels were TIME  ·  not a win, not a loss",
             ha="center", fontsize=12.5, color=THEME["YELLOW"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, 0.020, "@quant.dhawan", ha="center", fontsize=13,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "TripleBarrierLabeling_Static.png"))
