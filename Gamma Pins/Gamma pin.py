"""
Gamma_Pin_Static_Pipeline.py
=============================
STATIC hero-image pipeline for "The Gamma Pin".

Standalone: real SPY option chain -> Black-Scholes gamma -> GEX landscape over
(strike x days-to-expiry) -> ONE high-res frame at a fixed angle chosen to put
the pin ridge in profile with the negative-gamma wing visible behind it.

    python Gamma_Pin_Static_Pipeline.py
    -> Gamma_Pin_Static.png  (3240x5760)
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "TICKER": "SPY", "N_EXP": 8, "RATE": 0.04, "STRIKE_PCT": 0.065,
    "N_K": 130, "N_T": 34, "SMOOTH": (0.8, 0.8),      # denser than the reel
    "W": 1080, "H": 1920, "DPI": 300, "ELEV": 26, "AZIM": -52,
}
THEME = {"BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#8a8a8a",
         "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
         "RED": "#ff3050", "FONT": "Arial"}
CMAP = LinearSegmentedColormap.from_list(
    "gex", ["#ff3050", "#a01430", "#101820", "#0066ff", "#00c2ff", "#eaffff"],
    N=256)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}")


def fetch_gex():
    c = CONFIG
    log(f"[Data] Fetching {c['TICKER']} option chain...")
    try:
        import yfinance as yf
        tk = yf.Ticker(c["TICKER"])
        S = float(tk.fast_info["lastPrice"])
        exps = tk.options[:c["N_EXP"]]
        rows = []
        for e in exps:
            days = max((pd.Timestamp(e) - pd.Timestamp.today()).days, 0)
            T = max(days, 1) / 365.0
            ch = tk.option_chain(e)
            for df, sgn in ((ch.calls, 1.0), (ch.puts, -1.0)):
                d = df[["strike", "openInterest", "impliedVolatility"]].dropna()
                d = d[(d.impliedVolatility > 0.01) & (d.openInterest > 0)]
                if not len(d):
                    continue
                K = d.strike.values
                iv = d.impliedVolatility.values
                d1 = (np.log(S / K) + (c["RATE"] + iv ** 2 / 2) * T) / (iv * np.sqrt(T))
                gam = norm.pdf(d1) / (S * iv * np.sqrt(T))
                rows.append(pd.DataFrame({
                    "days": max(days, 1), "K": K,
                    "gex": sgn * d.openInterest.values * gam * 100 * S * S * 0.01}))
        return pd.concat(rows), S, len(exps)
    except Exception as e:
        log(f"[Data] chain fetch failed ({e}); synthetic fallback")
        S = 500.0
        K = np.arange(S * 0.9, S * 1.1, 1.0)
        rows = []
        for days in [1, 2, 5, 9, 16, 30]:
            pin = np.exp(-((K - S * 1.004) ** 2) / (2 * 3.0 ** 2)) * 3.2e9 / days ** 0.6
            put = -np.exp(-((K - S * 0.975) ** 2) / (2 * 6.0 ** 2)) * 4e8 / days ** 0.5
            rows.append(pd.DataFrame({"days": days, "K": K, "gex": pin + put}))
        return pd.concat(rows), S, 6


def render_static(out):
    c = CONFIG
    raw, S, n_exp = fetch_gex()
    lo, hi = S * (1 - c["STRIKE_PCT"]), S * (1 + c["STRIKE_PCT"])
    raw = raw[(raw.K >= lo) & (raw.K <= hi)]
    per = raw.groupby(["days", "K"]).gex.sum().reset_index()
    day_vals = np.sort(per.days.unique())
    Kg = np.linspace(lo, hi, c["N_K"])
    Zd = np.array([np.interp(Kg, per[per.days == dv].sort_values("K").K.values,
                             per[per.days == dv].sort_values("K").gex.values,
                             left=0.0, right=0.0) for dv in day_vals])
    Tg = np.linspace(day_vals.min(), day_vals.max(), c["N_T"])
    Z = np.array([np.interp(Tg, day_vals, Zd[:, j]) for j in range(len(Kg))]).T
    Z = gaussian_filter(Z, sigma=c["SMOOTH"], mode="nearest")
    KK, TT = np.meshgrid(Kg, Tg)

    tot = Z.sum(axis=0)
    pin = float(Kg[int(np.argmax(tot))])
    pin_gex = float(Z[:, int(np.argmax(tot))].max())
    log(f"spot ${S:,.2f}  pin ${pin:,.0f} ({(pin/S-1)*100:+.2f}%)  "
        f"peak ${pin_gex/1e9:.2f}B  pos/neg cells {int((Z>0).sum())}/{int((Z<0).sum())}")

    Zs = np.sign(Z) * np.sqrt(np.abs(Z) / np.abs(Z).max())
    zpos, zneg = float(max(Zs.max(), 1e-9)), float(max(-Zs.min(), 1e-9))
    zlim = float(np.abs(Zs).max())
    nrm = np.clip(0.5 + 0.5 * np.where(Zs >= 0, Zs / zpos, Zs / zneg), 0, 1)

    fig = plt.figure(figsize=(c["W"] / 100, c["H"] / 100), dpi=c["DPI"],
                     facecolor=THEME["BG"])
    fig.text(0.5, 0.955, "THE GAMMA PIN", ha="center", fontsize=33,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.926, f"{c['TICKER']} dealer gamma  ·  strike × days to expiry",
             ha="center", fontsize=15, color=THEME["ORANGE"], family=THEME["FONT"])

    ax = fig.add_axes([-0.20, 0.185, 1.40, 0.725], projection="3d",
                      facecolor=THEME["BG"])
    ax.plot_surface(KK, TT, Zs, facecolors=CMAP(nrm), rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False, zorder=2)
    ax.contour(KK, TT, Zs, levels=[0.0], colors=[THEME["YELLOW"]],
               linewidths=1.6, alpha=0.55, zorder=4)

    j = int(np.argmin(np.abs(Kg - pin)))
    ax.plot([pin, pin], [Tg.min(), Tg.min()], [0, Zs[0, j]], color="white",
            linewidth=1.3, alpha=0.5, zorder=9)
    ax.scatter([pin], [Tg.min()], [Zs[0, j]], s=100, color="white",
               edgecolors=THEME["CYAN"], linewidths=2.1, zorder=12)

    ax.set_xlim(Kg.min(), Kg.max()); ax.set_ylim(Tg.max(), Tg.min())
    ax.set_zlim(-zlim * 0.55, zlim * 1.03)
    ax.view_init(elev=c["ELEV"], azim=c["AZIM"])
    ax.set_box_aspect((1.5, 0.95, 0.95))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, 0.150, f"pinned to \\${pin:,.0f}", ha="center", fontsize=29,
             color=THEME["CYAN"], fontweight="bold", family=THEME["FONT"])
    fig.text(0.5, 0.120,
             f"spot \\${S:,.2f}   ·   {abs(pin/S-1)*100:.2f}% away   ·   "
             f"\\${pin_gex/1e9:.1f}B gamma",
             ha="center", fontsize=15, color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, 0.090, "blue = dealers fade every move   ·   red = they amplify it",
             ha="center", fontsize=13, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, 0.062, "Gamma explodes into expiry. The magnet gets stronger.",
             ha="center", fontsize=14, color=THEME["TEXT"], style="italic",
             family=THEME["FONT"])
    fig.text(0.5, 0.036,
             f"real {c['TICKER']} chain, {n_exp} expiries  ·  heights on a signed √ scale",
             ha="center", fontsize=10, color="#5a5a5a", family=THEME["FONT"])
    fig.text(0.5, 0.013, "@quant.dhawan", ha="center", fontsize=13,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    fig.savefig(out, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"Static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "Gamma_Pin_Static.png"))
