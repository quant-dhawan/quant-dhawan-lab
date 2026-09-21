"""
AnalogForecasting_Static_Pipeline.py
=====================================
STATIC hero image (poster / carousel cover) for "The Market Has Been Here Before".

THE MATHS
Identical to the reel and recomputed here, standalone. Every SPY session becomes
a standardised 3-vector ( 20-day log momentum , log 20-day realised vol , 60-day
volume z ). k-means reduces 15 years of them to 48 seeds; the seeds are mirrored
across the six faces of [-3, 3]^3 so `scipy.spatial.Voronoi` returns every cell
CLOSED and clipped to the box. Each day lands in the cell of its nearest seed,
and a cell is coloured by the MEAN NEXT-DAY LOG RETURN of the days inside it.
The camera window around today's seed is carved exactly, with
`scipy.spatial.HalfspaceIntersection`, so no cell can leave the frame.

THE CLAIM
Today's state vector lands in one cell. That cell has been visited N times
before; those N analogs averaged X% the next session. N and X come from
validate(), which is the same function the reel runs.

AND THE HONEST PART
48 cells over ~3,700 observations is ~77 days per cell, so the per-cell means are
noise. validate() asserts that today's analog mean is inside two standard errors
of zero and that a 2000-draw label permutation spreads the cell means at least as
wide as the real data does. The build fails if this ever looks like an edge.

WHAT THIS VISUAL IS NOT
NOT a point cloud, NOT a manifold embedding, NOT a clustering result. The CELLS
are the subject; the observations are never drawn as dust. Market State Cloud,
Diffusion Maps and Manifold Learning already exist in this repo and this must not
drift into any of them.

Unlike the reel the camera is a fixed curated 3/4 angle, the window holds more
cells, and the whole thing is rendered at 300 DPI so the still stays crisp.

    python AnalogForecasting_Static_Pipeline.py
    -> AnalogForecasting_Static.png  (3240x5760)
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import Voronoi, ConvexHull, HalfspaceIntersection

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
    "TICKER": "SPY", "PERIOD": "15y",
    "MOM_WIN": 20, "VOL_WIN": 20, "VZ_WIN": 60,
    "N_SEEDS": 48, "BOX": 3.0, "SEED_CLIP": 0.93,
    "KM_SEED": 7, "N_PERM": 2000, "PERM_SEED": 3,
    "W": 1080, "H": 1920, "DPI": 300,
    "ELEV": 20, "AZIM": 24,      # curated fixed angle for the still
    "VIEW_R": 1.70,              # wider window than the reel: denser complex
    "ZOOM": 1.20,
}
THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#8a8a8a",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "FONT": FONT_SANS,
}
CMAP = LinearSegmentedColormap.from_list(
    "analog", ["#ff3050", "#c9566a", "#8e9ab5", "#3fb98a", "#00ff8c"], N=256)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}")


def _cached(name, fetch):
    p = os.path.join(BASE_DIR, "_cache", name + ".csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0, parse_dates=True)
    df = fetch()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p)
    return df


def fetch_spy():
    """SPY daily OHLCV. Daily reel, so a synthetic fallback is permitted."""
    c = CONFIG

    def _dl():
        import yfinance as yf
        df = yf.download(c["TICKER"], period=c["PERIOD"], interval="1d",
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if not len(df):
            raise RuntimeError("empty yfinance frame")
        return df

    try:
        df = _cached("spy_daily", _dl)
        log(f"[Data] SPY {df.index.min().date()} -> {df.index.max().date()}  rows {len(df)}")
        return df
    except Exception as e:
        log(f"[Data] yfinance failed ({e}); synthetic fallback")
        rng = np.random.default_rng(19)
        n = 3800
        vol = 0.008 * np.exp(0.45 * np.cumsum(rng.normal(0, 0.06, n)))
        r = rng.normal(0.0004, 1, n) * vol
        px = 120 * np.exp(np.cumsum(r))
        vv = 7e7 * np.exp(rng.normal(0, 0.35, n) + 6 * np.abs(r))
        idx = pd.bdate_range(end=pd.Timestamp("2026-09-16"), periods=n)
        return pd.DataFrame({"Close": px, "Volume": vv}, index=idx)


def state_vectors(df):
    c = CONFIG
    px = df["Close"].astype(float)
    vv = df["Volume"].astype(float)
    r = np.log(px / px.shift(1))
    mom = np.log(px / px.shift(c["MOM_WIN"]))
    rv = np.log(r.rolling(c["VOL_WIN"]).std() * np.sqrt(252))
    vz = (vv - vv.rolling(c["VZ_WIN"]).mean()) / vv.rolling(c["VZ_WIN"]).std()
    F = pd.DataFrame({"mom": mom, "rv": rv, "vz": vz,
                      "nxt": r.shift(-1)}).dropna()
    X = F[["mom", "rv", "vz"]].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, F["nxt"].values, F.index


def tessellate(X):
    from sklearn.cluster import KMeans
    c = CONFIG
    b = c["BOX"] * c["SEED_CLIP"]
    km = KMeans(n_clusters=c["N_SEEDS"], n_init=10,
                random_state=c["KM_SEED"]).fit(np.clip(X, -b, b))
    seeds = np.clip(km.cluster_centers_, -b, b)
    pts = [seeds]
    for ax in range(3):
        for s in (-1.0, 1.0):
            m = seeds.copy()
            m[:, ax] = 2.0 * s * c["BOX"] - m[:, ax]
            pts.append(m)
    P = np.vstack(pts)
    return seeds, Voronoi(P), P


def cell_geometry(v):
    """Convex polyhedron -> (fill triangles, TRUE polyhedron edges)."""
    h = ConvexHull(v)
    tris = v[h.simplices]
    eq = h.equations[:, :3]
    emap = {}
    for si, simp in enumerate(h.simplices):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            emap.setdefault(tuple(sorted((int(simp[a]), int(simp[b])))), []).append(si)
    segs = []
    for k, sis in emap.items():
        if len(sis) == 2 and abs(float(np.dot(eq[sis[0]], eq[sis[1]]))) > 0.9999:
            continue
        segs.append(v[list(k)])
    return tris, np.asarray(segs)


def window_cells(P, ctr, R, n_cells):
    """Cells around today's state, each exactly half-space-clipped to the view."""
    out = {}
    for i in range(n_cells):
        if float(np.abs(P[i] - ctr).max()) >= R - 0.08:
            continue
        dvec = P - P[i]
        mid = 0.5 * (P + P[i])
        keep = np.arange(len(P)) != i
        A = dvec[keep]
        b = -np.einsum("ij,ij->i", dvec[keep], mid[keep])
        for ax in range(3):
            for sgn in (-1.0, 1.0):
                n = np.zeros(3); n[ax] = sgn
                A = np.vstack([A, n]); b = np.append(b, -(sgn * ctr[ax] + R))
        hs = HalfspaceIntersection(np.column_stack([A, b]), P[i])
        out[i] = np.unique(np.round(hs.intersections, 9), axis=0)
    return out


def permutation_p(lab, y, cnt, n_cells):
    c = CONFIG
    w = cnt / cnt.sum()

    def disp(vals):
        m = np.array([vals[lab == k].mean() for k in range(n_cells)])
        return float(np.sqrt(np.sum(w * (m - vals.mean()) ** 2)))

    obs = disp(y)
    rng = np.random.default_rng(c["PERM_SEED"])
    null = np.array([disp(rng.permutation(y)) for _ in range(c["N_PERM"])])
    return obs, float(np.median(null)), float((null >= obs).mean())


# ----------------------------------------------------------------------------
def validate(d):
    """Every number that reaches the image is asserted here, before any render."""
    c = CONFIG
    vor = d["vor"]
    n_cells = c["N_SEEDS"]

    nvts = []
    for i in range(n_cells):
        reg = vor.regions[vor.point_region[i]]
        assert len(reg) > 0 and -1 not in reg, f"cell {i} region is unbounded"
        nvts.append(len(reg))
    assert min(nvts) >= 4, f"a cell has {min(nvts)} vertices, not a polyhedron"
    # 4 = tetrahedron, the minimum closed 3D cell; fewer means the mirror trick failed.
    vmax = max(float(np.abs(vor.vertices[vor.regions[vor.point_region[i]]]).max())
               for i in range(n_cells))
    assert vmax <= c["BOX"] + 1e-6, f"cell vertex at {vmax:.4f} outside the box"
    # 1e-6 is float slack only: the mirror planes put vertices exactly on +-BOX.

    verts, ctr, R = d["verts"], d["ctr"], c["VIEW_R"]
    assert d["today"] in verts, "today's cell is not in the drawn window"
    assert 10 <= len(verts) <= 34, f"{len(verts)} cells in the window, legibility band"
    # The still can carry a denser complex than the reel, hence the wider band.
    for k, v in verts.items():
        assert len(v) >= 4, f"drawn cell {k} has {len(v)} vertices, not a polyhedron"
        off = float(np.abs(v - ctr).max())
        assert off <= R + 1e-6, f"drawn cell {k} reaches {off:.4f} > half-width {R}"
    # Exact, not cosmetic: the half-space clip means nothing can leave the frame.

    n_obs = int(d["n_obs"])
    assert 3000 <= n_obs <= 6000, f"{n_obs} sessions outside the 15y band"
    # 15y of SPY is ~3,770 sessions minus the 60-bar warmup.

    means, cnt = d["means"], d["cnt"]
    assert np.isfinite(means).all(), "a per-cell mean next-day return is not finite"
    assert cnt.min() >= 5, f"a cell holds only {cnt.min()} days, its mean is meaningless"
    # Expected occupancy is n_obs/48 ~ 77, so 5 only trips on a degenerate tiling.

    n_an, mu, se = int(d["n_analogs"]), float(d["mu"]), float(d["se"])
    assert n_an >= 25, f"today's cell has only {n_an} analogs"
    # A third of expected occupancy: "this has happened before" must be literally true.
    assert abs(mu) <= 0.005, f"cell mean {mu*100:.2f}% is implausible for a daily mean"
    # 0.5%/day is >7x the unconditional SPY mean; only a data error reaches it.

    tstat = abs(mu) / se
    assert tstat < 2.0, f"analog mean is {tstat:.2f} SE from zero, the caption is wrong"
    # The honesty gate: if the analog mean ever became significant, the "no edge"
    # story would be false and this build must fail instead of shipping it.
    pv = float(d["p_shuffle"])
    assert pv >= 0.20, f"permutation p={pv:.3f}: cell means beat shuffled labels"
    # p >= 0.20 = the tessellation explains no more next-day variation than a
    # random relabelling. Currently ~0.96; 0.20 leaves room for drift.

    figs = {
        "n_analogs": str(n_an),
        "analog_mean": f"{mu*100:+.2f}%",
        "analog_se": f"{se*100:.2f}%",
        "t_stat": f"{tstat:.1f}",
        "base_mean": f"{d['base_mean']*100:+.2f}%",
        "n_obs": f"{n_obs:,}",
        "n_cells": str(n_cells),
        "n_neighbours": str(int(d["n_neighbours"])),
        "years": "15",
        "p_shuffle": f"{pv:.2f}",
        "shuffle_pct": f"{pv*100:.0f}%",
        "ticker": c["TICKER"],
        "last_date": d["last_date"],
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log("[validate] " + "  ".join(f"{k}={v}" for k, v in figs.items()))
    return figs


# ----------------------------------------------------------------------------
def prepare():
    c = CONFIG
    df = fetch_spy()
    X, y, dates = state_vectors(df)
    log(f"state vectors: {X.shape}  {dates[0].date()} -> {dates[-1].date()}")

    seeds, vor, P = tessellate(X)
    n_cells = c["N_SEEDS"]
    lab = ((X[:, None, :] - seeds[None, :, :]) ** 2).sum(-1).argmin(1)
    cnt = np.bincount(lab, minlength=n_cells)
    means = np.array([y[lab == k].mean() for k in range(n_cells)])

    today = int(lab[-1])
    sel = np.where(lab == today)[0][:-1]
    mu = float(y[sel].mean())
    se = float(y[sel].std(ddof=1) / np.sqrt(len(sel)))
    obs, nullmed, pv = permutation_p(lab, y, cnt, n_cells)
    log(f"dispersion obs {obs*100:.4f}%  shuffled median {nullmed*100:.4f}%  p={pv:.3f}")

    nbrs = {int(a if b == today else b) for a, b in vor.ridge_points
            if (a == today and b < n_cells) or (b == today and a < n_cells)}

    ctr = seeds[today].copy()
    verts = window_cells(P, ctr, c["VIEW_R"], n_cells)
    geom = {k: cell_geometry(v) for k, v in verts.items()}
    log(f"window cells drawn: {len(geom)}  facet-neighbours of today: "
        f"{len(set(geom) & nbrs)}")

    scale = float(np.percentile(np.abs(means), 85)) * 1.15
    norm = np.clip(0.5 + means / (2.0 * max(scale, 1e-9)), 0.0, 1.0)
    cols = [CMAP(v) for v in norm]
    dists = np.linalg.norm(seeds - ctr, axis=1)

    return dict(seeds=seeds, vor=vor, verts=verts, geom=geom, cols=cols,
                dists=dists, today=today, nbrs=nbrs, ctr=ctr, today_pt=X[-1],
                means=means, cnt=cnt, n_obs=len(X), n_analogs=len(sel),
                mu=mu, se=se, base_mean=float(y.mean()), p_shuffle=pv,
                n_neighbours=len(nbrs), last_date=str(dates[-1].date()))


def render(d, figs):
    c = CONFIG
    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, 0.955, "THE MARKET HAS BEEN HERE BEFORE", ha="center",
             fontsize=30, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, 0.929,
             f"{figs['n_cells']} Voronoi cells of SPY state space  ·  "
             "momentum × volatility × volume",
             ha="center", fontsize=14, color=THEME["ORANGE"], family=THEME["FONT"])

    ax = fig.add_axes([-0.08, 0.215, 1.16, 0.675], projection="3d",
                      facecolor=THEME["BG"])
    cx, cy, cz = d["ctr"]; R = c["VIEW_R"]
    ax.set_xlim(cx - R, cx + R); ax.set_ylim(cy - R, cy + R)
    ax.set_zlim(cz - R, cz + R)
    ax.view_init(elev=c["ELEV"], azim=c["AZIM"])
    ax.set_box_aspect((1, 1, 1), zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    today, nbrs, cols, dists = d["today"], d["nbrs"], d["cols"], d["dists"]
    order = sorted(d["geom"], key=lambda k: -dists[k])
    dnear = dists[order[-2]]
    for k in order:
        tris, segs = d["geom"][k]
        if k == today:
            ea, fa, ec, lw = 1.0, 0.44, THEME["CYAN"], 2.6
        elif k in nbrs:
            ea, fa, ec, lw = 0.26, 0.0, cols[k], 1.0
        else:
            ea = 0.16 * float(np.exp(-(dists[k] - dnear) / 0.9))
            fa, ec, lw = 0.0, cols[k], 0.8
        if fa > 0.01:
            pc = Poly3DCollection(tris, facecolors=cols[k], linewidths=0,
                                  alpha=fa, zorder=12)
            pc.set_edgecolor((0, 0, 0, 0))
            ax.add_collection3d(pc)
        if ea > 0.01:
            ax.add_collection3d(Line3DCollection(
                segs, colors=ec, linewidths=lw, alpha=ea,
                zorder=14 if k == today else 4))

    p = d["today_pt"]
    ax.scatter([p[0]], [p[1]], [p[2]], s=150, color=THEME["YELLOW"],
               edgecolors=THEME["TEXT"], linewidths=1.6, depthshade=False,
               zorder=30)

    fig.text(0.5, 0.176, f"{figs['n_analogs']} ANALOGS IN TODAY'S CELL",
             ha="center", fontsize=27, fontweight="bold", color=THEME["CYAN"],
             family=THEME["FONT"])
    fig.text(0.5, 0.134, f"mean next session  {figs['analog_mean']}",
             ha="center", fontsize=19, color=THEME["TEXT"], family=FONT_MONO)
    fig.text(0.5, 0.092,
             f"standard error {figs['analog_se']}   ·   t = {figs['t_stat']}"
             f"   ·   base rate {figs['base_mean']}",
             ha="center", fontsize=13, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, 0.061,
             f"{figs['shuffle_pct']} of shuffled labels spread the cells wider",
             ha="center", fontsize=13, color=THEME["RED"], family=THEME["FONT"])
    fig.text(0.985, 0.018, "@quant.dhawan", ha="right", va="bottom",
             fontsize=12, color=THEME["TEXT_DIM"], alpha=0.75,
             family=THEME["FONT"])

    out = os.path.join(BASE_DIR, "AnalogForecasting_Static.png")
    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    return out


def main():
    t0 = time.time(); log("=== ANALOG FORECASTING STATIC ===")
    d = prepare()
    figs = validate(d)
    out = render(d, figs)
    log(f"OK {out}  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
