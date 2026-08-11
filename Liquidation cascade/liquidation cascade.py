"""
Liquidation_Cascade_Static_Pipeline.py
=======================================
STATIC hero-image pipeline for "Liquidation Cascade".

Standalone: real BTC hourly -> worst 24h window -> modelled leverage ladder ->
ONE high-res frame at the CLIMAX (price at the low, every breached shelf lit)
from a fixed near-side-on angle, so it reads as a price chart first.

    python Liquidation_Cascade_Static_Pipeline.py
    -> Liquidation_Cascade_Static.png  (3240x5760)
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "TICKER": "BTC-USD", "PERIOD": "730d", "INTERVAL": "1h", "WINDOW_H": 24,
    "LEVERAGE": [100, 50, 25, 10, 5], "OI_LOOKBACK_H": 24 * 21,
    "OI_TOTAL_USD": 2.5e9, "N_ENTRY_BINS": 90, "SHELVES_PER_TIER": 13,
    "W": 1080, "H": 1920, "DPI": 300, "ELEV": 12, "AZIM": -78,
}
THEME = {"BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#8a8a8a",
         "ORANGE": "#ff9500", "CYAN": "#00f2ff", "RED": "#ff3050",
         "FONT": "Arial"}
CMAP = LinearSegmentedColormap.from_list(
    "liq", ["#123a6b", "#0066ff", "#00c2ff", "#ffd400", "#ff9500", "#ff3050"],
    N=256)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}")


def fetch_crash():
    c = CONFIG
    log(f"[Data] Fetching {c['TICKER']} {c['INTERVAL']}...")
    try:
        import yfinance as yf
        d = yf.download(c["TICKER"], period=c["PERIOD"], interval=c["INTERVAL"],
                        progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d = d.xs(c["TICKER"], axis=1, level=1)
        px = d["Close"].dropna()
    except Exception as e:
        log(f"[Data] yfinance failed ({e}); synthetic fallback")
        rng = np.random.default_rng(3)
        n = 17000
        p = 60000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
        p[12000:12024] *= np.linspace(1, 0.86, 24)
        px = pd.Series(p, index=pd.date_range(end=pd.Timestamp.utcnow(),
                                              periods=n, freq="h"))
    W = c["WINDOW_H"]
    roll = px / px.shift(W) - 1
    i_end = px.index.get_loc(roll.idxmin())
    crash = px.iloc[i_end - W:i_end + 1]
    prior = px.iloc[max(0, i_end - W - c["OI_LOOKBACK_H"]):i_end - W]
    log(f"worst {W}h: {roll.min()*100:.2f}%  ${crash.iloc[0]:,.0f} -> ${crash.iloc[-1]:,.0f}")
    return crash, prior


def build_ladder(crash, prior):
    c = CONFIG
    entries = prior.values if len(prior) > 50 else crash.values
    hist, edges = np.histogram(entries, bins=c["N_ENTRY_BINS"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist / max(hist.sum(), 1)
    p0, p_low = float(crash.iloc[0]), float(crash.min())
    tiers = []
    for L in c["LEVERAGE"]:
        liq = centers * (1.0 - 1.0 / L)
        usd = w * c["OI_TOTAL_USD"] / len(c["LEVERAGE"])
        keep = (liq < p0 * 1.001) & (liq > p_low * 0.90) & (usd > 0)
        u, lq = usd[keep], liq[keep]
        if u.sum() > 0:
            u = u / u.sum() * c["OI_TOTAL_USD"] / len(c["LEVERAGE"])
        if len(lq) > c["SHELVES_PER_TIER"]:
            e = np.linspace(lq.min(), lq.max(), c["SHELVES_PER_TIER"] + 1)
            b = np.clip(np.digitize(lq, e) - 1, 0, c["SHELVES_PER_TIER"] - 1)
            u = np.array([u[b == k].sum() for k in range(c["SHELVES_PER_TIER"])])
            lq = 0.5 * (e[:-1] + e[1:])
            m = u > 0
            lq, u = lq[m], u[m]
        tiers.append({"L": L, "liq": lq, "usd": u})
    return tiers


def render_static(out):
    c = CONFIG
    crash, prior = fetch_crash()
    tiers = build_ladder(crash, prior)
    p0, plo, phi = float(crash.iloc[0]), float(crash.min()), float(crash.max())
    price_now = float(crash.iloc[-1])
    usd_max = max((t["usd"].max() if len(t["usd"]) else 0) for t in tiers)
    liquidated = sum(float(t["usd"][t["liq"] >= price_now].sum()) for t in tiers)
    log(f"climax: BTC ${price_now:,.0f}  ${liquidated/1e9:.2f}B liquidated")

    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100), dpi=c["DPI"],
                     facecolor=THEME["BG"])
    fig.text(0.5, 0.955, "LIQUIDATION CASCADE", ha="center", fontsize=31,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.926,
             f"BTC {crash.iloc[-1]/crash.iloc[0]*100-100:.1f}% in {c['WINDOW_H']}h"
             f"  ·  {crash.index[0]:%b %Y}",
             ha="center", fontsize=15, color=THEME["ORANGE"], family=THEME["FONT"])

    ax = fig.add_axes([-0.20, 0.195, 1.40, 0.715], projection="3d",
                      facecolor=THEME["BG"])
    ny = len(tiers)
    for yi, t in enumerate(tiers):
        y = ny - 1 - yi
        for liq, usd in zip(t["liq"], t["usd"]):
            if usd <= 0:
                continue
            wq = (usd / usd_max) ** 0.55
            hit = liq >= price_now
            col = THEME["RED"] if hit else CMAP(0.12 + 0.30 * wq)
            alpha = (0.16 + 0.42 * wq) if hit else (0.10 + 0.22 * wq)
            verts = [[(0.0, y - 0.42, liq), (6.0, y - 0.42, liq),
                      (6.0, y + 0.42, liq), (0.0, y + 0.42, liq)]]
            ax.add_collection3d(Poly3DCollection(
                verts, facecolors=col, edgecolors="none", alpha=alpha, zorder=2))

    n_h = len(crash)
    xs = np.linspace(0.0, 6.0, n_h)
    ys = np.full(n_h, -1.15)
    zs = crash.values
    pts = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    dr = np.clip((p0 - zs[:-1]) / max(p0 - plo, 1e-9), 0, 1)
    ax.add_collection3d(Line3DCollection(segs, colors=CMAP(0.25 + 0.75 * dr),
                                         linewidths=3.2, zorder=9))
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=95, color="white",
               edgecolors=THEME["RED"], linewidths=1.9, zorder=12)

    ax.set_xlim(0, 6.0); ax.set_ylim(-1.7, ny - 0.3)
    ax.set_zlim(plo * 0.985, phi * 1.004)
    ax.view_init(elev=c["ELEV"], azim=c["AZIM"])
    ax.set_box_aspect((1.5, 0.85, 1.25))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    for yi, t in enumerate(tiers):
        tot = float(t["usd"].sum())
        frac = float(t["usd"][t["liq"] >= price_now].sum()) / tot if tot else 0
        col = (THEME["RED"] if frac > 0.85 else
               THEME["ORANGE"] if frac > 0.45 else THEME["CYAN"])
        fig.text(0.13 + yi * 0.185, 0.172, f"{t['L']}x", ha="center",
                 fontsize=19, color=col, fontweight="bold", family=THEME["FONT"])
        fig.text(0.13 + yi * 0.185, 0.152, f"{frac*100:.0f}% gone", ha="center",
                 fontsize=11, color=col, alpha=0.8, family=THEME["FONT"])

    fig.text(0.5, 0.113, f"${liquidated/1e9:.2f}B liquidated", ha="center",
             fontsize=28, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, 0.086, f"BTC  ${price_now:,.0f}", ha="center", fontsize=17,
             color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.050,
             "price: real BTC data  ·  ladder modelled, equal notional per tier",
             ha="center", fontsize=10, color="#5a5a5a", family=THEME["FONT"])
    fig.text(0.5, 0.020, "@quant.dhawan", ha="center", fontsize=13,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "Liquidation_Cascade_Static.png"))
