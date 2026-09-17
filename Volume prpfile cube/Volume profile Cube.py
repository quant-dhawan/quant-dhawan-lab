"""
VolumeProfileCube_Static_Pipeline.py
====================================
STATIC hero image (poster / carousel cover) for "Where The Volume Actually Trades".

Standalone: DATA (cached SPY 1-minute bars) -> per-session VWAP -> basis-point
lattice -> VALUE AREA -> validate() -> render ONE high-res frame.

THE MATHS
For each session s, VWAP_s = sum_i v_i * tp_i / sum_i v_i with tp = (H+L+C)/3,
and every bar is re-expressed as z_i = 1e4 * (tp_i / VWAP_s - 1), basis points
from that session's own VWAP. Each session's own VWAP is the origin because
SPY drifts several dollars a week: a raw price grid would stack the sessions
into different slices of the axis and the lattice would show drift rather than
microstructure. The lattice L[session, half-hour bucket, bp bin] is the volume
traded in that cell, normalised by the session's total volume.

THE CLAIM
The value area -- the contiguous band holding 70% of a session's volume, grown
out from the point of control by the standard market-profile expansion -- holds
about three quarters of the day's volume inside about half of the day's
high-to-low range. Both numbers come out of validate(); the cyan wire slab
marks the band.

WHAT THIS IS NOT
Not a heatmap, not a 3D bar chart, not a price path. No surface, no
interpolation, no line through time: every mark is one discrete matplotlib
voxel standing for one bin of traded volume.

Unlike the reel, the camera is a fixed curated 3/4 angle, the fill threshold is
lower so more of the thin tail is drawn (a denser lattice, hence a larger
n_voxels than the reel writes), and the render is at 300 DPI. Every other
number in figures.json is identical to the reel's: same bars, same lattice,
same validate().

    python VolumeProfileCube_Static_Pipeline.py
    -> VolumeProfileCube_Static.png  (3240x5760)
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    "N_CHUNKS": 4,                # 4 x 7-day pulls; Yahoo caps 1m at 8 days/request
    "BP_LIM": 50.0,               # Z axis spans +/-50 bp around each session VWAP
    "NZ": 20,                     # price bins -> 5.0 bp per bin
    "BUCKET_MIN": 30,             # half-hour time-of-day buckets
    "NY": 13,                     # 09:30-16:00 in half hours
    "VA_TARGET": 0.70,            # value area definition: 70% of session volume
    "FILL_Q": 0.0004,             # lower than the reel: the still can carry the thin tail
    "W": 1080, "H": 1920, "DPI": 300,
    "ELEV": 14, "AZIM": -34,      # curated fixed angle for the still
    "BOX_ASPECT": (1.25, 0.95, 1.95), "ZOOM": 1.26,
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#8a8a8a",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": FONT_SANS,
}
CMAP = LinearSegmentedColormap.from_list(
    "vol", ["#123a8c", "#0066ff", "#00f2ff", "#00ff8c", "#ffd400", "#ff9500",
            "#ff3050"], N=256)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}")


def _cached(name, fetch):
    p = os.path.join(BASE_DIR, "_cache", name + ".csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0, parse_dates=True)
    df = fetch()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p)
    return df


def fetch_bars():
    """~19 sessions of SPY 1-minute bars. Fails loudly: no synthetic tape."""
    c = CONFIG

    def _pull():
        import yfinance as yf
        log(f"[Data] Fetching {c['TICKER']} 1m in {c['N_CHUNKS']} chunks...")
        frames = []
        end = pd.Timestamp.utcnow().normalize()
        for k in range(c["N_CHUNKS"]):
            e = end - pd.Timedelta(days=7 * k)
            s = e - pd.Timedelta(days=7)
            df = yf.download(c["TICKER"], start=s.date(), end=e.date(),
                             interval="1m", progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if len(df):
                frames.append(df)
        if not frames:
            raise RuntimeError("1-minute fetch returned nothing")
        bars = pd.concat(frames).sort_index()
        return bars[~bars.index.duplicated()]

    bars = _cached("spy_1m", _pull)
    if len(bars) < 5000:
        raise RuntimeError(f"only {len(bars)} 1m bars; intraday still refuses to "
                           "run on a thin tape and has no synthetic fallback")
    return bars


def build_lattice(bars):
    """(session, half-hour, bp-from-VWAP) volume lattice + per-session profiles."""
    c = CONFIG
    et = bars.index.tz_convert("America/New_York") if bars.index.tz is not None \
        else bars.index
    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    df = pd.DataFrame({"d": et.date, "m": et.hour * 60 + et.minute,
                       "tp": tp.values, "v": bars["Volume"].values.astype(float),
                       "hi": bars["High"].values, "lo": bars["Low"].values})
    sessions = sorted(df["d"].unique())

    nz, ny = c["NZ"], c["NY"]
    edges = np.linspace(-c["BP_LIM"], c["BP_LIM"], nz + 1)
    lat = np.zeros((len(sessions), ny, nz))
    prof = np.zeros((len(sessions), nz))
    rng_bp = np.zeros(len(sessions))
    for si, d in enumerate(sessions):
        g = df[df["d"] == d]
        v = g["v"].values
        vwap = (g["tp"].values * v).sum() / v.sum()
        bp = (g["tp"].values / vwap - 1.0) * 1e4
        yi = np.clip((g["m"].values - 570) // c["BUCKET_MIN"], 0, ny - 1).astype(int)
        zi = np.clip(np.digitize(bp, edges) - 1, 0, nz - 1)
        np.add.at(lat, (si, yi, zi), v)
        np.add.at(prof, (si, zi), v)
        lat[si] /= v.sum()
        prof[si] /= v.sum()
        rng_bp[si] = (g["hi"].max() / vwap - 1.0) * 1e4 - \
                     (g["lo"].min() / vwap - 1.0) * 1e4
    return {"lat": lat, "prof": prof, "rng_bp": rng_bp, "edges": edges,
            "sessions": sessions, "n_bars": len(df)}


def value_area(p, target):
    """Standard market-profile expansion out from the point of control."""
    tot = p.sum()
    poc = int(np.argmax(p))
    lo = hi = poc
    acc = p[poc]
    while acc < target * tot:
        up = p[hi + 1] if hi + 1 < len(p) else -1.0
        dn = p[lo - 1] if lo - 1 >= 0 else -1.0
        if up < 0 and dn < 0:
            break
        if up >= dn:
            hi += 1; acc += p[hi]
        else:
            lo -= 1; acc += p[lo]
    return lo, hi, acc / tot


def validate(d):
    """Every number that reaches the frame is asserted here, before any render."""
    c = CONFIG
    lat, prof, rng_bp = d["lat"], d["prof"], d["rng_bp"]
    bw = d["edges"][1] - d["edges"][0]
    n_sessions = lat.shape[0]

    los, his, shares, fracs = [], [], [], []
    for si in range(n_sessions):
        lo, hi, sh = value_area(prof[si], c["VA_TARGET"])
        los.append(lo); his.append(hi); shares.append(sh)
        fracs.append(((hi - lo + 1) * bw) / rng_bp[si])
    shares = np.array(shares); fracs = np.array(fracs)

    va_share = float(shares.mean())
    range_frac = float(fracs.mean())
    fill = lat >= c["FILL_Q"]
    n_vox = int(fill.sum())

    assert not np.isnan(lat).any(), "NaN in the volume lattice"          # every cell is a summed volume; a NaN means a bad bar slipped through the bp/bucket binning
    assert n_sessions >= 15, f"only {n_sessions} sessions"               # 4 x 7-day 1m chunks give 19-20 sessions; under 15 the per-session mean is too thin to average
    assert d["n_bars"] >= 5000, f"only {d['n_bars']} bars"               # 19 regular sessions x 390 minutes = 7,410; 5,000 tolerates two lost chunks, less means the tape is broken
    assert 0.70 <= va_share <= 0.78, f"value area share {va_share:.3f}"  # 0.70 is a hard floor of the expansion, which stops on first crossing; the overshoot is at most one edge bin, worth ~4% of a session at 5.0 bp resolution, so 0.78 catches a binning change that makes the profile too coarse to mean anything
    assert 0.38 <= range_frac <= 0.60, f"range fraction {range_frac:.3f}"  # mean of 19 per-session values, observed 0.485 with sd 0.096 so SE 0.022; +/-0.10 is ~4.5 SE, wide enough for normal month-to-month drift and still excluding a flat profile (~0.70) or a single-bin one (~0.15)
    assert va_share - range_frac >= 0.15, "value area no longer concentrated"  # this difference IS the claim: volume share must beat range share by a clear margin, observed 0.257, and 0.15 is the point below which the headline stops being worth saying
    assert n_vox >= 400, f"only {n_vox} lit voxels"                      # under ~400 cubes the lattice stops reading as a lattice at poster size

    figs = {
        "va_vol_pct": f"{va_share * 100:.0f}%",
        "va_range_pct": f"{range_frac * 100:.0f}%",
        "outside_vol_pct": f"{(1 - va_share) * 100:.0f}%",
        "outside_range_pct": f"{(1 - range_frac) * 100:.0f}%",
        "n_sessions": str(n_sessions),
        "n_bars": f"{d['n_bars']:,}",
        "n_voxels": str(n_vox),
        "bin_bp": f"{bw:.1f} bp",
        "bucket_min": f"{c['BUCKET_MIN']} min",
        "first_session": str(d["sessions"][0]),
        "last_session": str(d["sessions"][-1]),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log("[Validate] " + "  ".join(f"{k}={v}" for k, v in figs.items()))
    d["fill"] = fill
    d["va_lo"] = int(np.round(np.mean(los)))
    d["va_hi"] = int(np.round(np.mean(his)))
    d["va_bins"] = (np.array(los), np.array(his))
    return figs


def _wire_box(ax, x0, x1, y0, y1, z0, z1, color, alpha, lw):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for z in (z0, z1):
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            ax.plot([a[0], b[0]], [a[1], b[1]], [z, z],
                    color=color, alpha=alpha, linewidth=lw, zorder=20)
    for (px, py) in corners:
        ax.plot([px, px], [py, py], [z0, z1],
                color=color, alpha=alpha * 0.55, linewidth=lw * 0.7, zorder=20)


def render_static(out):
    c = CONFIG
    bars = fetch_bars()
    d = build_lattice(bars)
    f = validate(d)

    lat = d["lat"]
    hot = np.quantile(lat[lat > 0], 0.985)
    norm = np.clip(lat / hot, 0, 1) ** 0.55
    los, his = d["va_bins"]
    zi = np.arange(lat.shape[2])[None, :]
    in_va = ((zi >= los[:, None]) & (zi <= his[:, None]))[:, None, :] \
        .repeat(lat.shape[1], axis=1)

    cols = CMAP(norm)
    cols[..., 3] = np.where(in_va, 0.70 + 0.28 * norm, 0.16 + 0.14 * norm)
    nx, ny, nz = lat.shape

    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, 0.962, "WHERE THE VOLUME ACTUALLY TRADES", ha="center",
             fontsize=26, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, 0.936,
             f"SPY  ·  {f['n_bars']} one-minute bars  ·  {f['n_sessions']} sessions",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])

    ax = fig.add_axes([-0.17, 0.055, 1.34, 0.845], projection="3d",
                      facecolor=THEME["BG"])
    ax.voxels(d["fill"], facecolors=cols, edgecolors=(1, 1, 1, 0.13),
              linewidth=0.25, shade=False)
    _wire_box(ax, 0, nx, 0, ny, d["va_lo"], d["va_hi"] + 1,
              THEME["CYAN"], 0.85, 2.2)

    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
    ax.view_init(elev=c["ELEV"], azim=c["AZIM"])
    ax.set_box_aspect(c["BOX_ASPECT"], zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, 0.885,
             f"X  session     Y  {f['bucket_min']} of the day     "
             f"Z  {f['bin_bp']} from that day's VWAP     "
             "cyan box  typical value area",
             ha="center", fontsize=10.5, color=THEME["TEXT_DIM"], family=FONT_MONO)
    fig.text(0.5, 0.082,
             f"VALUE AREA  {f['va_vol_pct']} of volume  ·  "
             f"{f['va_range_pct']} of the range",
             ha="center", fontsize=17, color=THEME["CYAN"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, 0.052,
             f"the other {f['outside_range_pct']} of the range carries "
             f"{f['outside_vol_pct']} of the volume",
             ha="center", fontsize=12, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.075, 0.018, f"{f['n_voxels']} lit cells", ha="left", va="bottom",
             fontsize=10.5, color=THEME["TEXT_DIM"], alpha=0.75, family=FONT_MONO)
    fig.text(0.925, 0.018, "@quant.dhawan", ha="right", va="bottom",
             fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75,
             family=THEME["FONT"])

    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "VolumeProfileCube_Static.png"))
