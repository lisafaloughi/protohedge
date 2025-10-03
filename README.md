# ProtoHedge: Interpretable Hedging with Market Prototypes

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This repository extends the original **Deep Hedging** framework by introducing a **prototype-based interpretable agent** that makes hedging decisions transparent and auditable.

> **Note**
>
> This repo builds on the original Deep Hedging implementation by **Hans Buehler and contributors** (MIT license): https://github.com/hansbuehler/deephedging  
> Our contribution is the integration of **prototype-based interpretability** (ProtoHedge). All credit for the base framework belongs to the upstream authors. This fork is released for **academic and research** purposes.

---

## 🔍 What’s new in ProtoHedge

- **Prototype layer (`ClusteredProtoLayer`)** that learns a fixed set of market **prototypes** (via KMeans/medoids) from features such as `price`, `delta`, and `time_left`.
- Each prototype has a **trainable action vector**; the final action is a **similarity-weighted combination** of bounded prototype actions.
- Full **traceability**: every action can be explained by which prototypes influenced it and how strongly.
- Modular additions; compatible with the original Deep Hedging training loop.

---

## Repository structure
- `notebooks/` — training and testing notebooks (prototype trainer, Black–Scholes experiments, etc.)
- `pictures/` — plots used in README
- Core modules: `agents.py`, `layers.py`, `trainer.py`, `world.py`
- Utilities: `plot_training.py`, `plot_bs_hedge.py`, `softclip.py`


---
# Deep Hedging
## Reinforcement Learning for Hedging Derivatives under Market Frictions

This repository builds on the [Deep Hedging framework](http://deep-hedging.com), which trains hedging strategies under market frictions by optimizing a **risk-adjusted utility** of terminal P&L.

The purpose of this section is to provide ProtoHedge users with the essentials of Deep Hedging.  
For full details, please see the [original project](https://github.com/hansbuehler/deephedging).

---

### Core Idea

The Deep Hedging problem for a horizon $T$ hedged over $M$ time steps with $N$ instruments is to find an optimal *action* function $a(s_t)$ (hedge adjustment at time $t$ given state $s_t$) that maximizes expected utility:

$$
\sup_a\; U\Bigg[ Z_T + \sum_{t=0}^{T-1} a(s_t)\cdot DH_t - \gamma_t \cdot |a(s_t)| \Bigg],
$$

where:
- $s_t$ = observable state (features like price, delta, time left),
- $H_t$ = instrument mid-prices,
- $DH_t$ = forward returns of hedges,
- $\gamma_t$ = proportional transaction costs,
- $Z_T$ = payoff at maturity,
- $U$ = monetary utility (e.g., CVaR, entropic utility).

---

### Key Components

- **World (`world.py`)**  
  Provides simulated market data (e.g., Black–Scholes, stochastic volatility).  
  Exposes:
  - `features`: per-step and per-path state variables (spot, vol, time left, etc.).
  - `payoff`: terminal payoff of the product.
  - `hedges`: returns of hedging instruments.
  - `cost`: transaction cost of trading.

- **Gym (`gym.py`)**  
  Wraps the agent and world into a training environment.  
  Given simulated paths, evaluates actions → computes P&L → applies utility → computes loss.

- **Agent (`agents.py`)**  
  Defines the mapping from features to actions.  
  - In vanilla Deep Hedging: a feed-forward or recurrent neural net.  
  - In ProtoHedge: a prototype-based layer (`ClusteredProtoLayer`) replaces the black-box net.

- **Trainer (`trainer.py`)**  
  Wrapper around `keras.fit()`. Handles training, validation, monitoring, and caching.

---

### Installation

See `requirements.txt` for exact dependencies. At a minimum:

- Python 3.9+
- TensorFlow ≥ 2.10
- tensorflow-probability ≥ 0.15
- cvxpy
- cdxbasics


## Key Objects and Functions

* **world.SimpleWorld_Spot_ATM** class  

    Simple world with either plain Black & Scholes dynamics, or with one asset and one floating ATM option.  
    The asset has stochastic volatility and a mean-reverting drift.  
    The implied volatility of the asset is not equal to the realized volatility, allowing to reproduce some results from [this paper](https://arxiv.org/abs/2103.11948).  

    * Set the `black_scholes` boolean config flag to `True` to turn the world into a simple Black & Scholes world with no traded option.
    * Use `no_stoch_vol` to turn off stochastic volatility, and `no_stoch_drift` to turn off stochastic drift.  
    If both are `True`, the market is Black & Scholes, but the option can still be traded for hedging.  

    See `notebooks/simpleWorld_Spot_ATM.ipynb`.  

---

* **gym.VanillaDeepHedgingGym** class  

    The main Deep Hedging training gym (a Monte Carlo loop). It creates the agent network internally and computes the monetary utility $U$ along with unhedged baseline results.  

    To run the model on all samples of a given `world`, use:  

    ```python
    r = gym(world.tf_data)
    ```

    The returned dictionary includes:
    * `utility (:,)` – objective to maximize (per path).
    * `utility0 (:,)` – objective without hedging (per path).
    * `loss (:,)` – the loss `-utility-utility0` per path.
    * `payoff (:,)` – terminal payoff per path.
    * `pnl (:,)` – mid-price PnL from trading (ex cost) per path.
    * `cost (:,)` – aggregate cost of trading per path.
    * `gains (:,)` – total gains (`payoff + pnl - cost`) per path.
    * `actions (:,M,N)` – actions taken per step and instrument.
    * `deltas (:,M,N)` – deltas per step and path (`np.cumsum(actions, axis=1)`).

---

* **trainer.train()** function  

    The main Deep Hedging training engine (stochastic gradient descent).  
    Trains the model using Keras. Any Keras optimizer can be used.  

    * Supports live visualization in Jupyter notebooks.  
    * When training outside Jupyter, set `config.train.monitor_type = "none"`.  
    * See `notebooks/trainer.ipynb` as an example.  

---

### ProtoHedge Extensions

In addition to the original Deep Hedging components, **ProtoHedge** introduces new modules:

* **ClusteredProtoLayer**  
    - A custom layer that stores a fixed set of prototypes (learned via KMeans or medoids).  
    - Each prototype has an associated **trainable action vector**.  
    - During inference, actions are chosen as a **similarity-weighted combination** of prototype actions.  
    - Supports **soft clipping** of actions via a differentiable bounding function.

* **ProtoAgent**  
    - A new agent class built on top of `ClusteredProtoLayer`.  
    - Encodes input market states (`price`, `delta`, `time_left`, etc.), compares them to prototypes, and outputs interpretable hedging actions.  
    - Every action can be **explained** in terms of its closest prototype(s).  
    - Fully integrated into the `VanillaDeepHedgingGym` and training pipeline.

* **Interpretability utilities**  
    - Functions for tracing decisions back to prototypes.  
    - Tools for visualizing **prototype usage frequency** and **per-step prototype assignments**.  

These extensions preserve compatibility with the original Deep Hedging training loop while enabling **interpretable, prototype-based hedging strategies**.
    
       
    


## Interpreting Progress Graphs

Here is an example of progress information printed by  `NotebookMonitor`:

![progess example](pictures/progress.png)

The graphs show:

* (1): visualizing convergence
    
    * (1a): last 100 epochs loss view on convergence: initial loss, full training set loss with std error, batch loss, validation loss, and the running best fit.
    
    * (1b): loss across all epochs, same metrics as above.
    
    * (1c): learned utility for the hedged payoff, last 100 epochs.
    
    * (1d): learned utility for the unhedged payoff, last 100 epochs.

    * (1e): memory usage
    
* (2) visualizing the result on the training set:
    
    * (2a) for the training set shows the payoff as function of terminal spot, the hedge, and the overall gains i.e. payoff plus hedge less cost. Each of these is computed less their monetary utility, hence you are comparing terminal payoffs which have the same initial monetary utility.

    * (2b) visualizing terminal payoffs makes sense for European payoffs, but is a lot less intuitive for more exotic payoffs. Hence, (2b) shows the same data is (2a) but
    it adds one stddev for the value of the payoff for a given terminal spot.

    * (2c) shows the utility for payoff and gains by percentile for the training set. Higher is better; the algorithm
    is attempting to beat the total expected utility which is the left most percentile. In the example we see that the gains process dominates the initial payoff in all percentiles.
    
    * (2a', 2b', 2c') are the same graphs but for the validation set 
    
* (3, 4) visualize actions:
    
    * (3a) shows  actions per time step: blue the spot, and orange the option.
    * (3b) shows the aggregated action as deltas accross time steps. Note that the concept of "delta" only makes sense if the instrument is actually the same per time step, e.g. spot of an stock price. For floating options this is not a particularly meaningful concept.

    * (4a) shows the average action for the first asset (spot) for a few time steps over spot bins.
    * (4b) shows the avergae delta for the first asset for a a few time steps over spot bins.
    * (4a*) shows the same as (4a), but also showing a one stddev bar around each bin mid-point.
    * (4b*) the same for (4b)

Text information:

* (A) provides information on the number of weights used, whether an initial delta or recurrent weights were used, features used and available, and the utility.

  It also shows the caching path.

* (B) Provides information during training on progress, and an estimate of total time required.
    

    
## Running ProtoHedge

The easiest way to get started is to run the provided Jupyter notebook:

Open **`notebooks/proto-trainer.ipynb`**

This notebook will:  
1. Build the **simulation world** (`SimpleWorld_Spot_ATM`, Black–Scholes or stochastic vol).  
2. Initialize the **ProtoAgent** (using the `ClusteredProtoLayer` with prototypes).  
3. Train the model using the standard Deep Hedging training loop.  
4. Plot convergence graphs (loss, utilities, actions, deltas).  
5. Show **prototype attribution results** so you can see which market prototypes drove each hedging action.  

The output includes:  
- Hedging performance (comparable to the black-box baseline).  
- Plots of training progress.  
- Tables/figures showing prototype activations and weights (for interpretability).  

---

## Misc Code Overview

**Core training files:**
- `world.py` — market simulator (`SimpleWorld_Spot_ATM`).
- `gym.py` — Monte Carlo training loop (`VanillaDeepHedgingGym`).
- `trainer.py` — wrapper around `keras.fit()`, with caching and visualization support.
- `objectives.py` — monetary utilities (CVaR, entropic utility, etc.).
- `base.py`, `fd.py`, `model_base.py` — supporting math and utility functions.
- `layers.py` — contains standard dense layers and the new prototype layer.
- `agents.py` — defines agents and how they connect to layers.
- `softclip.py` — smooth bounding for hedging actions.

**Notebooks:**
- `notebooks/proto-trainer.ipynb` — main ProtoHedge training and evaluation run.
- `notebooks/test-proto-trainer.ipynb` — additional experiments on black-scholes world.
- `notebooks/test-stoch-proto-trainer.ipynb` — stochastic volatility experiments.
- `notebooks/trainer.ipynb` — original Deep Hedging baseline for comparison.

**Utilities & figures:**
- `plot_training.py` — live training visualizations.
- `plot_bs_hedge.py` — compare hedging strategies vs Black–Scholes.
- `pictures/progress.png` — example of training progress output.
- `Network.md` — notes on network and recurrent agent options.

    
## Acknowledgments
This work builds on the [Deep Hedging](https://github.com/hansbuehler/deephedging) framework by Hans Buehler and contributors.  

## License
Released under the MIT License. See [LICENSE](LICENSE) for details.