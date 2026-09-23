"""
IndexConcentrationPacking_Static_Pipeline.py
============================================
STATIC hero-image pipeline (poster / carousel cover) for "The Index Is A Few
Spheres".

THE MATHS
---------
Same as the reel and computed here from scratch, so the still is standalone.
A pinned sample of 58 S&P 500 constituents, one sphere per name, radius the
CUBE ROOT of market cap:

    r_i = k * M_i^(1/3)        =>    sphere VOLUME  ~  market cap

Cube root, not radius-proportional: the eye reads volume, so volume is what
must carry the number. Radius-proportional would cube the exaggeration.
The pack is the same deterministic greedy algorithm under the same seed
(largest sphere at the centre, then the most central non-overlapping candidate
out of 6000 random draws), so this still is the reel's pack frozen, not a
second arrangement.

THE CLAIM
---------
The top 7 names are ~59% of the sampled index's market cap. The sample is a
PINNED SAMPLE of the index, not all 500 names, so the quoted share is the share
WITHIN THAT SAMPLE and the caption says so.

WHAT THIS IS NOT
----------------
Not a treemap, not a bubble chart on a plane, not a convex hull of anything.
No hull, no 2D projection, no rectangles.

Unlike the reel the camera is one curated fixed 3/4 angle, the sphere mesh is
denser and nothing is mid-drop, so the still is crisp at 300 DPI.

    python IndexConcentrationPacking_Static_Pipeline.py
    -> IndexConcentrationPacking_Static.png  (3240x5760)
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
    "TICKERS": [
        "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
        "BRK-B", "LLY", "JPM", "V", "XOM", "UNH",
        "WMT", "MA", "ORCL", "COST", "HD", "PG", "JNJ", "NFLX", "ABBV",
        "BAC", "CRM", "AMD", "CVX", "KO", "MRK", "PEP",
        "ADBE", "CSCO", "MCD", "ACN", "TMO", "LIN", "ABT", "QCOM", "TXN",
        "CAT", "GE", "DIS", "VZ", "PFE",
        "MOS", "NWSA", "MKTX", "HAS", "LW", "RL", "BEN", "DVA", "AOS",
        "TAP", "CPB", "FRT", "IVZ", "ALB",
    ],
    "TOP_N": 7, "MIN_CAP": 1e9, "MIN_RETURNED": 45,
    "PACK_PHI": 0.36, "PACK_GAP": 0.004, "PACK_CAND": 6000, "PACK_SEED": 7,
    "BOX_L": 1.0, "SOLID_N": 7, "LABEL_N": 5,
    "NU": 40, "NV": 28,        # denser than the reel's 16x12
    "NU_T": 22, "NV_T": 16,
    "ELEV": 14, "AZIM": -34,   # curated fixed angle for the still
    "W": 1080, "H": 1920, "DPI": 300,
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#8a8a8a",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": FONT_SANS,
}
CMAP = LinearSegmentedColormap.from_list(
    "cap", ["#0b2f7a", THEME["CYAN"], THEME["GREEN"], THEME["YELLOW"],
            THEME["ORANGE"], THEME["RED"]], N=256)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}")


def _cached(name, fetch):
    p = os.path.join(BASE_DIR, "_cache", name + ".csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0).iloc[:, 0].astype(float)
    s = fetch()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    s.to_frame("market_cap").to_csv(p)
    return s


def fetch_caps():
    """No synthetic fallback: a fabricated cap table would make the
    concentration share on the poster a lie. Fail loudly instead."""
    c = CONFIG

    def _pull():
        import yfinance as yf
        log(f"[Data] Fetching market caps for {len(c['TICKERS'])} tickers...")
        out = {}
        for t in c["TICKERS"]:
            mc = None
            try:
                info = yf.Ticker(t).info
                mc = info.get("marketCap")
                if not mc:
                    so = info.get("sharesOutstanding")
                    px = info.get("regularMarketPrice") or info.get("currentPrice")
                    mc = so * px if so and px else None
            except Exception as e:
                log(f"  {t}: {e}")
            out[t] = float(mc) if mc else np.nan
        return pd.Series(out, dtype="float64")

    s = _cached("sp500_sample_caps", _pull)
    s = s[np.isfinite(s.values) & (s.values > c["MIN_CAP"])]
    return s.sort_values(ascending=False)


def greedy_pack(caps):
    c = CONFIG
    L, gap = c["BOX_L"], c["PACK_GAP"]
    phi = c["PACK_PHI"]
    for _ in range(8):
        k = (phi * L ** 3 / ((4.0 / 3.0) * np.pi * caps.sum())) ** (1.0 / 3.0)
        r = k * caps ** (1.0 / 3.0)            # VOLUME ~ market cap
        rng = np.random.default_rng(c["PACK_SEED"])
        P = np.zeros((len(r), 3))
        ok_all = True
        for i in range(1, len(r)):
            h = L / 2.0 - r[i] - gap
            if h <= 0:
                ok_all = False
                break
            cand = rng.uniform(-h, h, (c["PACK_CAND"], 3))
            dd = np.linalg.norm(cand[:, None, :] - P[None, :i, :], axis=2)
            free = (dd >= (r[i] + r[:i] + gap)[None, :]).all(axis=1)
            if not free.any():
                ok_all = False
                break
            cc = cand[free]
            P[i] = cc[np.argmin(np.linalg.norm(cc, axis=1))]
        if ok_all:
            log(f"[Pack] phi={phi:.3f}  r_max={r[0]:.3f}  r_min={r[-1]:.3f}")
            return P, r, phi
        phi *= 0.9
    raise RuntimeError("greedy pack failed at every density")


def unit_sphere(nu, nv):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    return (np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)))

def lambert_facecolors(mx, my, mz, rgb, alpha, ambient=0.28):
    """Per-face Lambert shading for a unit-sphere mesh.

    plot_surface(shade=False) paints a sphere as one flat colour, which at
    reel scale reads as a 2D disc and turns the whole pack into a bubble
    chart. That is the one thing this visual must not be, so the curvature
    has to be lit explicitly. For a unit sphere the surface normal at a mesh
    point IS the point, so the diffuse term is just n . L.
    """
    L = np.array([-0.42, -0.66, 0.62])
    L = L / np.linalg.norm(L)
    ndotl = np.clip(mx * L[0] + my * L[1] + mz * L[2], 0.0, 1.0)
    shade = ambient + (1.0 - ambient) * ndotl
    spec = 0.38 * ndotl ** 10                      # tight highlight, reads as gloss
    out = np.empty(mx.shape + (4,))
    for k in range(3):
        out[..., k] = np.clip(rgb[k] * shade + spec, 0.0, 1.0)
    out[..., 3] = alpha
    return out


def validate(d):
    """Every number that reaches the poster is asserted here, before any render.
    Identical bands to the reel, on purpose: the two must not be able to drift
    apart."""
    c = CONFIG
    caps, P, r = d["caps"], d["P"], d["r"]
    n = len(caps)
    total = float(caps.sum())

    assert np.isfinite(caps).all(), "a fetched market cap is NaN or inf"
    assert (caps > 0).all(), "a fetched market cap is not positive"
    assert caps.min() > c["MIN_CAP"], \
        f"smallest cap ${caps.min()/1e9:.2f}B below the $1B floor"
    # No S&P 500 member trades under $1B; a row under that is a delisted or
    # renamed yfinance stub, not a constituent.

    assert n >= c["MIN_RETURNED"], f"only {n} of {len(c['TICKERS'])} tickers priced"
    # 45 of 58. Below that the denominator has lost enough tail that the quoted
    # share describes a different basket than the one named.

    top_n = c["TOP_N"]
    top_pct = float(caps[:top_n].sum() / total * 100.0)
    assert 45.0 <= top_pct <= 75.0, f"top-{top_n} share {top_pct:.1f}% outside band"
    # Measured 59.1% on 2026-09-17, bracketed by roughly +/-15pts: under 45% the
    # megacap block no longer dominates and the premise is gone; over 75% the
    # tail has been eaten and the sample is broken.

    top1_pct = float(caps[0] / total * 100.0)
    assert 6.0 <= top1_pct <= 25.0, f"largest name {top1_pct:.1f}% of sample"
    # Measured 12.8%. Under 6% there is no giant to pack around; over 25% one
    # name is half the box and the pack will not close.

    top10_pct = float(caps[:10].sum() / total * 100.0)
    assert top10_pct >= top_pct, "top-10 share below top-7, sort is broken"

    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    S = r[:, None] + r[None, :]
    np.fill_diagonal(D, np.inf)
    clear = float((D - S).min())
    assert clear >= 0.0, f"two spheres overlap by {-clear:.5f} box units"
    # Hard zero. Overlapping spheres would show volume that is not there, the
    # exact exaggeration the cube-root mapping exists to avoid.

    out = float((np.abs(P) + r[:, None]).max())
    assert out <= c["BOX_L"] / 2.0 + 1e-9, f"a sphere pokes {out:.4f} outside the box"
    # The box is the sampled index; a sphere outside it claims a member that is
    # not in the sample.

    fill = float((4.0 / 3.0 * np.pi * (r ** 3).sum()) / c["BOX_L"] ** 3 * 100.0)

    figs = {
        "top_n": str(top_n),
        "top_n_pct": f"{top_pct:.0f}%",
        "top_n_pct_exact": f"{top_pct:.1f}%",
        "top1_pct": f"{top1_pct:.0f}%",
        "top10_pct": f"{top10_pct:.0f}%",
        "n_names": str(n),
        "n_pinned": str(len(c["TICKERS"])),
        "total_cap": f"${total/1e12:.1f}T",
        "largest": d["names"][0],
        "largest_cap": f"${caps[0]/1e12:.2f}T",
        "smallest": d["names"][-1],
        "radius_ratio": f"{r[0]/r[-1]:.1f}x",
        "cap_ratio": f"{caps[0]/caps[-1]:.0f}x",
        "fill_pct": f"{fill:.0f}%",
        "top_names": ", ".join(d["names"][:top_n]),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[Validate] top-{top_n}={figs['top_n_pct_exact']}  n={n}  "
        f"clearance={clear:.5f}  fill={figs['fill_pct']}")
    return figs


def render_static(out):
    c = CONFIG
    s = fetch_caps()
    names, caps = list(s.index), s.values.astype(float)
    P, r, phi = greedy_pack(caps)
    d = dict(caps=caps, names=names, P=P, r=r)
    figs = validate(d)

    v = r / r.max()                    # same cube root the radius uses
    col = [CMAP(x) for x in v]
    L = c["BOX_L"]; h = L / 2.0
    n = len(r)

    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100), dpi=c["DPI"],
                     facecolor=THEME["BG"])
    fig.text(0.5, 0.945, "THE INDEX IS A FEW SPHERES", ha="center", fontsize=30,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.917,
             f"{figs['n_names']} S&P 500 names  ·  sphere VOLUME = market cap",
             ha="center", fontsize=14, color=THEME["ORANGE"], family=THEME["FONT"])

    ax = fig.add_axes([-0.17, 0.120, 1.34, 0.80], projection="3d",
                      facecolor=THEME["BG"])

    corners = np.array([[sx * h, sy * h, sz * h]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), color=THEME["CYAN"],
                alpha=0.40, linewidth=1.0, zorder=0)

    # hand-rolled painter's sort: matplotlib ranks 3d collections against each
    # other by zorder only, so far spheres must be drawn first or the tail
    # paints over the giants.
    er, ar = np.radians(c["ELEV"]), np.radians(c["AZIM"])
    cam = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])
    zrank = {i: k for k, i in enumerate(np.argsort(P @ cam))}

    sx, sy, sz = unit_sphere(c["NU"], c["NV"])
    tx, ty, tz = unit_sphere(c["NU_T"], c["NV_T"])
    for i in range(n):
        solid = i < c["SOLID_N"]
        mx, my, mz = (sx, sy, sz) if solid else (tx, ty, tz)
        X = P[i, 0] + r[i] * mx
        Y = P[i, 1] + r[i] * my
        Z = P[i, 2] + r[i] * mz
        zo = 2 + 2 * zrank[i]
        if solid:
            fc = lambert_facecolors(mx, my, mz, col[i][:3], 0.88)
            ax.plot_surface(X, Y, Z, facecolors=fc, linewidth=0,
                            antialiased=True, shade=False, zorder=zo)
            ax.plot_wireframe(X, Y, Z, color="#ffffff", alpha=0.16,
                              linewidth=0.45, rstride=2, cstride=2, zorder=zo + 1)
        else:
            ax.plot_wireframe(X, Y, Z, color=col[i], alpha=0.60, linewidth=0.35,
                              rstride=1, cstride=1, zorder=zo)

    for i in range(min(c["LABEL_N"], n)):
        ax.text(P[i, 0], P[i, 1], P[i, 2] + r[i] * 0.05, names[i],
                color=THEME["TEXT"], fontsize=11, fontweight="bold",
                ha="center", va="center", family=THEME["FONT"],
                alpha=0.92, zorder=400)

    ax.set_xlim(-h, h); ax.set_ylim(-h, h); ax.set_zlim(-h, h)
    ax.view_init(elev=c["ELEV"], azim=c["AZIM"])
    ax.set_box_aspect((1, 1, 1))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, 0.108, f"TOP {figs['top_n']} = {figs['top_n_pct']} OF THE SAMPLE",
             ha="center", fontsize=26, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, 0.079,
             f"{figs['n_names']} names  ·  {figs['total_cap']} packed  ·  "
             f"top 10 = {figs['top10_pct']}",
             ha="center", fontsize=13, color=THEME["TEXT_DIM"], family=FONT_MONO)
    fig.text(0.5, 0.050,
             f"{figs['largest']} alone is {figs['top1_pct']}  ·  "
             f"{figs['cap_ratio']} the cap of the smallest name here",
             ha="center", fontsize=12, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, 0.026, "pinned sample of the index, not all 500 names",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"], alpha=0.65,
             family=THEME["FONT"])
    fig.text(0.985, 0.010, "@quant.dhawan", ha="right", va="bottom",
             fontsize=11, color=THEME["TEXT_DIM"], alpha=0.75,
             family=THEME["FONT"])

    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "IndexConcentrationPacking_Static.png"))
