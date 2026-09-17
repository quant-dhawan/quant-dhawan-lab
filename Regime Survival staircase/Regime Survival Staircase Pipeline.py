"""
RegimeSurvivalStaircase_Static_Pipeline.py
============================================
STATIC hero-image pipeline (poster / carousel cover) for the Regime Survival
Staircase: three complete 3D STEP STAIRCASES -- Kaplan-Meier survival curves
of SPY volatility regimes (calm / normal / stressed spell length) extruded
into solid stair blocks, hard risers and flat treads, no smooth curve.

Same data source and same validate() as the reel: trailing 20-day realised
volatility of SPY log returns, binned at fixed 12%/20% annualised thresholds
into calm/normal/stressed; a spell is a maximal run of consecutive days in
one bin; the Kaplan-Meier estimator handles the still-open final spell as a
right-censored observation instead of a raw histogram. See the reel's
docstring for the full derivation -- this file only differs in camera and
resolution: one curated fixed 3/4 angle, all three flights fully built, at
300 DPI. This is a historical distribution, not a forecast of when today's
regime ends.

    python RegimeSurvivalStaircase_Static_Pipeline.py
    -> RegimeSurvivalStaircase_Static.png  (3240x5760)
"""
import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- fonts (Arial resolves to nothing on this box; ship Inter/JetBrains) ----
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
    "TICKER": "SPY",
    "VOL_WINDOW": 20,
    "VOL_LO": 0.12, "VOL_HI": 0.20,
    "T_MAX_DAYS": 40,
    "Y_SCALE": 0.30,
    "REGIMES": ["calm", "normal", "stressed"],
    "LANE_X": {"calm": -4.6, "normal": 0.0, "stressed": 4.6},
    "LANE_HALF_W": 1.5,
    "W": 1080, "H": 1920, "DPI": 300,
    "ELEV": 22, "AZIM": -18,            # curated fixed angle for the still
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#8a8a8a",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": FONT_SANS,
}
CMAP = LinearSegmentedColormap.from_list(
    "survival", [THEME["RED"], THEME["ORANGE"], THEME["YELLOW"],
                 THEME["GREEN"], THEME["CYAN"]], N=256)
LANE_COLOR = {"calm": THEME["GREEN"], "normal": THEME["YELLOW"], "stressed": THEME["RED"]}
LANE_LABEL = {"calm": "CALM", "normal": "NORMAL", "stressed": "STRESSED"}


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
    local = os.path.abspath(os.path.join(BASE_DIR, "..", "Demiurge", "data", "SPY.csv"))
    if os.path.exists(local):
        log(f"[Data] using shipped {local}")
        df = pd.read_csv(local, index_col=0, parse_dates=True)
        return df["SPY"].sort_index()

    def _fetch():
        try:
            import yfinance as yf
            raw = yf.download("SPY", period="max", interval="1d", progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            return raw[["Close"]].rename(columns={"Close": "SPY"})
        except Exception as e:
            log(f"[Data] yfinance failed ({e}); synthetic fallback")
            rng = np.random.default_rng(7)
            n = 5000
            rets = rng.normal(0.0003, 0.011, n)
            px = 100 * np.cumprod(1 + rets)
            dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
            return pd.DataFrame({"SPY": px}, index=dates)

    df = _cached("SPY_daily", _fetch)
    return df["SPY"].sort_index()


def compute_regimes(px):
    c = CONFIG
    ret = np.log(px / px.shift(1)).dropna()
    rvol = (ret.rolling(c["VOL_WINDOW"]).std() * np.sqrt(252)).dropna()

    def bucket(v):
        if v <= c["VOL_LO"]:
            return "calm"
        elif v <= c["VOL_HI"]:
            return "normal"
        return "stressed"

    return rvol.apply(bucket), rvol


def compute_spells(labels):
    vals = labels.values
    dates = labels.index
    spells = []
    cur, ln, sd = vals[0], 1, dates[0]
    for i in range(1, len(vals)):
        if vals[i] == cur:
            ln += 1
        else:
            spells.append({"regime": cur, "length": ln, "censored": 0, "start": sd})
            cur, ln, sd = vals[i], 1, dates[i]
    spells.append({"regime": cur, "length": ln, "censored": 1, "start": sd})
    return spells


def kaplan_meier(lengths, events):
    lengths = np.asarray(lengths); events = np.asarray(events)
    times, surv = [0], [1.0]
    s = 1.0
    for t in np.sort(np.unique(lengths)):
        n_t = np.sum(lengths >= t)
        d_t = np.sum((lengths == t) & (events == 1))
        if d_t > 0:
            s *= (1 - d_t / n_t)
            times.append(int(t)); surv.append(s)
    return np.array(times), np.array(surv)


def km_at(times, surv, t):
    idx = np.searchsorted(times, t, side="right") - 1
    return surv[max(idx, 0)]


def median_survival(times, surv):
    below = np.where(surv <= 0.5)[0]
    return int(times[below[0]]) if len(below) else None


def prepare():
    c = CONFIG
    px = fetch_prices()
    labels, rvol = compute_regimes(px)
    spells = compute_spells(labels)
    by_regime = {r: [s for s in spells if s["regime"] == r] for r in c["REGIMES"]}
    curves = {}
    for r in c["REGIMES"]:
        lens = np.array([s["length"] for s in by_regime[r]])
        cens = np.array([s["censored"] for s in by_regime[r]])
        events = 1 - cens
        times, surv = kaplan_meier(lens, events)
        naive_times, naive_surv = kaplan_meier(lens, np.ones_like(events))
        day_grid = np.arange(0, c["T_MAX_DAYS"])
        surv_grid = np.array([km_at(times, surv, t) for t in day_grid])
        curves[r] = dict(lengths=lens, censored=cens, times=times, surv=surv,
                          naive_times=naive_times, naive_surv=naive_surv,
                          surv_grid=surv_grid, median=median_survival(times, surv))
    return dict(px=px, labels=labels, rvol=rvol, spells=spells,
                by_regime=by_regime, curves=curves)


def validate(d):
    """Identical bands to the reel -- see RegimeSurvivalStaircase_Reel_Pipeline.py
    for the justification of each. Both pipelines must pass on the same data."""
    c = CONFIG
    curves = d["curves"]
    figs = {}

    for r in c["REGIMES"]:
        surv = curves[r]["surv"]; grid = curves[r]["surv_grid"]
        assert surv[0] == 1.0, f"{r}: KM curve must start at S(0)=1.0"
        assert np.all(np.diff(surv) <= 1e-12), f"{r}: KM curve increased somewhere"
        assert np.all(np.diff(grid) <= 1e-12), f"{r}: displayed step grid increased somewhere"

    for r in c["REGIMES"]:
        n = len(curves[r]["lengths"])
        assert n >= 30, f"{r}: only {n} spells, below the 30-spell floor for a stable KM tail"
        figs[f"n_spells_{r}"] = str(n)

    for r in c["REGIMES"]:
        med = curves[r]["median"]
        assert med is not None, f"{r}: KM curve never crosses 0.5, no median"
        assert isinstance(med, int) and 3 <= med <= 60, \
            f"{r}: median {med} outside plausible 3-60 trading day band"
        figs[f"median_{r}"] = str(med)

    censored_regime = None
    for r in c["REGIMES"]:
        cens = curves[r]["censored"]
        if cens.sum() > 0:
            censored_regime = r
            tc = int(curves[r]["lengths"][cens == 1][0])
            km_val = km_at(curves[r]["times"], curves[r]["surv"], tc)
            naive_val = km_at(curves[r]["naive_times"], curves[r]["naive_surv"], tc)
            assert km_val > naive_val, (
                f"{r}: censoring-aware KM ({km_val:.4f}) should exceed the naive "
                f"treat-as-death curve ({naive_val:.4f}) at the open spell's length {tc}")
            figs["censored_regime"] = r
            figs["censored_len_days"] = str(tc)
            figs["km_at_censor"] = f"{km_val:.3f}"
            figs["naive_at_censor"] = f"{naive_val:.3f}"
    assert censored_regime is not None, "no regime is currently open -- data has no live spell"

    figs["vol_window"] = str(c["VOL_WINDOW"])
    figs["vol_lo_pct"] = f"{c['VOL_LO']*100:.0f}"
    figs["vol_hi_pct"] = f"{c['VOL_HI']*100:.0f}"
    figs["n_days"] = str(len(d["px"]))
    figs["date_start"] = str(d["px"].index[0].date())
    figs["date_end"] = str(d["px"].index[-1].date())

    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    return figs


def render_static(out):
    c = CONFIG
    d = prepare()
    figs = validate(d)
    curves = d["curves"]
    t_max = c["T_MAX_DAYS"]
    ys_scale = c["Y_SCALE"]
    log(f"medians  calm={figs['median_calm']}d normal={figs['median_normal']}d stressed={figs['median_stressed']}d")

    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100), dpi=c["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, 0.965, "REGIME SURVIVAL STAIRCASE", ha="center", fontsize=25,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.935,
             f"Kaplan-Meier survival of SPY vol regimes  ·  {figs['vol_window']}d trailing realised vol",
             ha="center", fontsize=12.5, color=THEME["ORANGE"], family=THEME["FONT"])
    ax = fig.add_axes([-0.06, 0.02, 1.12, 0.86], projection="3d", facecolor=THEME["BG"])

    for r in c["REGIMES"]:
        lane_x = c["LANE_X"][r]
        hw = c["LANE_HALF_W"]
        heights = curves[r]["surv_grid"]
        days = np.arange(t_max)
        xs = np.full(t_max, lane_x - hw)
        ys = days.astype(float) * ys_scale
        zs = np.zeros(t_max)
        dxs = np.full(t_max, 2 * hw)
        dys = np.full(t_max, 0.82 * ys_scale)
        dzs = heights
        colors = CMAP(heights)
        ax.bar3d(xs, ys, zs, dxs, dys, dzs, color=colors,
                 edgecolor=LANE_COLOR[r], linewidth=0.3, alpha=0.95, shade=True, zorder=3)

        ax.text(lane_x, -2.6, 0.05, LANE_LABEL[r], color=LANE_COLOR[r],
                fontsize=12.5, fontweight="bold", family=FONT_MONO, ha="center", zorder=20)

        med = curves[r]["median"]
        ax.scatter([lane_x], [med * ys_scale], [0.5], s=70, color=THEME["TEXT"],
                  edgecolors=LANE_COLOR[r], linewidths=1.4, zorder=10)
        ax.text(lane_x, med * ys_scale + 0.7, 0.58, f"{med}d", color=THEME["TEXT"],
                fontsize=10, family=FONT_MONO, ha="left", zorder=11)

    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-3.0, t_max * ys_scale + 1.2)
    ax.set_zlim(0, 1.15)
    ax.view_init(elev=c["ELEV"], azim=c["AZIM"])
    ax.set_box_aspect((1.55, 1.0, 0.92), zoom=1.22)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, 0.058,
             f"median lifespan: calm {figs['median_calm']}d  ·  normal {figs['median_normal']}d  ·  stressed {figs['median_stressed']}d",
             ha="center", fontsize=11.5, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, 0.032, "history, not a countdown  ·  this is not a forecast",
             ha="center", fontsize=9.5, color=THEME["TEXT_DIM"], family=THEME["FONT"], alpha=0.8)
    fig.text(0.985, 0.010, "@quant.dhawan", ha="right", va="bottom",
             fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "RegimeSurvivalStaircase_Static.png"))
