# Quant Dhawan Lab

The Python behind the quant animations on [@quant.dhawan](https://www.instagram.com/quant.dhawan).

Every reel, hero image and carousel starts as a script in this repository. Each one takes a single idea from quantitative finance, econophysics or stochastic calculus and renders it: a Marchenko-Pastur eigenvalue spectrum separating signal from noise, a Hawkes process clustering its own aftershocks, an implied volatility surface breathing across strikes. The maths is the point. The animation is how it gets explained.

**72 topics. 80 scripts. One idea each.**

```
├── Marchenko/
│   └── MarchenkoPastur_Reel_Pipeline.py     -> 1080x1350 reel
├── IV surface/
│   └── ...                                   -> 1920x1080 hero image
└── ...
```

## What is actually in here

These are render pipelines, not a trading library. A typical script pulls prices with `yfinance`, computes one quantity, and writes frames with `matplotlib`, stitched by `moviepy` into a portrait reel or a landscape still. There is no execution layer, no broker adapter and no strategy you can deploy. Read them the way you would read a worked example: the interesting part is the twenty lines in the middle that do the maths.

## Quickstart

```bash
git clone https://github.com/quant-dhawan/quant-dhawan-lab.git
cd quant-dhawan-lab
pip install numpy scipy pandas matplotlib yfinance moviepy
python "Beginner Quant Projects/Monte Carlo GBM.py"
```

Most scripts run with nothing but that stack. Three folders (`Wavelet Transform`, `RMT_Correlation_Filter`, `Fisher Transfrom`) carry their own `requirements.txt` for extras. Output lands next to the script.

If you are starting from zero, `Beginner Quant Projects/` is the door: efficient frontier, EWMA, mean reversion, Monte Carlo GBM, Sharpe ratio. Five scripts, plainly written.

## The 72 topics

**Start here**
`Beginner Quant Projects` · `Quant Research Projects`

**Stochastic calculus and option pricing**
`Bachlier` · `Black Scholes` · `Brownian Bridge` · `Cox process` · `GBM` · `Gamma Pins` · `Girsanov` · `Heston Model` · `IV surface` · `Ito Lemma` · `longstaff schwartz` · `Optimal Stopping` · `Ornstein-Uhlenbeck` · `OU Converge` · `Rough Volatility` · `SABR`

**Tails, distributions and simulation**
`Fat Tails` · `fat tail isosurface` · `Mandelbrot` · `Monte Carlo` · `Multifractal` · `Return Distribution` · `Sequential Monte Carlo` · `Wasserstein`

**Correlation, covariance and portfolios**
`Cointegration` · `Copulas` · `Hierarchical risk parity` · `Kelly Criterion` · `Marchenko` · `MST` · `RMT_Correlation_Filter` · `Sharpe Ratio` · `Statistical Arbitrage`

**Signal processing and time series**
`FFT` · `Fisher Transfrom` · `GARCH` · `Hidden Markov` · `Hilbert Transfrom` · `Hurst Exponent` · `Kalman` · `Lyapunov Exponent` · `Poincare` · `Recurrence` · `SSA` · `Wavelet Transform`

**Information theory and geometry**
`Diffusion Maps` · `Lempel-Ziv` · `Mutual Info` · `Path Signatures` · `Shannon Entropy`

**Econophysics**
`Cusp Catastrophe` · `Ergo` · `Hawkes Process` · `Ising Model` · `Kuramoto Model` · `Liquidation cascade` · `Lorenz BTC` · `Omori Law` · `Potts Model` · `Sandpile Model` · `Stoch Resonance` · `Wave Function Collapse`

**Machine learning**
`GAN` · `Neural Network` · `Overfitting Landscape` · `Reinforcement Learning`

**Microstructure and rates**
`Avellaneda Stoikov` · `Market State Cloud` · `Market Vector Field` · `Yield Curve`

## Read the theory

The written explanations live at [learn.thuztra.com](https://learn.thuztra.com), a free reference library of glossary terms and long-form articles. Several map directly onto folders here:

| Folder | Written up at |
|---|---|
| `Marchenko`, `RMT_Correlation_Filter` | [Fat tails](https://learn.thuztra.com/topics/fat-tails) |
| `Monte Carlo`, `Sequential Monte Carlo` | [Monte Carlo](https://learn.thuztra.com/topics/monte-carlo) |
| `IV surface`, `SABR`, `Heston Model` | [Implied volatility](https://learn.thuztra.com/topics/implied-volatility) |
| `Optimal Stopping`, `longstaff schwartz` | [Optimal stopping](https://learn.thuztra.com/topics/optimal-stopping) |
| `FFT` | [Fourier pricing](https://learn.thuztra.com/topics/fourier-pricing) |
| `Kelly Criterion` | [Kelly criterion](https://learn.thuztra.com/reference/kelly-criterion) |
| `Black Scholes` | [Black-Scholes as a heat equation](https://learn.thuztra.com/reference/black-scholes-as-a-heat-equation) |

Browse the whole thing: [glossary](https://learn.thuztra.com/glossary) · [reference articles](https://learn.thuztra.com/reference)

## Known rough edges

Kept honest rather than quietly cleaned up.

- Three folder names are misspelled in the tree: `Bachlier` (Bachelier), `Fisher Transfrom` and `Hilbert Transfrom` (Transform). Renaming them breaks every existing link, so they stay for now.
- 56 scripts still carry the retired `@quant.traderr` handle in headers or captions, and one draws it into rendered images as a watermark. The account is now `@quant.dhawan`.
- Scripts fetch live prices at run time, so a rerun will not reproduce an old render byte for byte.

## Licence and use

MIT. Take the code, take the maths, publish what you make with it.

Nothing here is investment advice, a recommendation, or a claim about returns. These are teaching artefacts.

## Elsewhere

- Animations and daily posts: [@quant.dhawan](https://www.instagram.com/quant.dhawan)
- Free reference library: [learn.thuztra.com](https://learn.thuztra.com)
- The backtesting engine: [thuztra.com](https://thuztra.com)
- Mail: `Team@thuztra.com`

Built by Vansh Dhawan.
