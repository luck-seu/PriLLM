# PriLLM: data-calibrated Stackelberg routing for LLM pricing (calibration code and 2026 data)

Calibration part of **Pricing Online LLM Services with Data-Calibrated Stackelberg Routing Game**
(Guo, Bai, Jin; AAAI 2026; [arXiv:2511.09062](https://arxiv.org/abs/2511.09062)).

## Paper versions

- AAAI 2026 version: *Proceedings of the AAAI Conference on Artificial Intelligence* 40(20), 17005-17013
  ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38748)).
- Extended version with the full technical appendices: [arXiv:2511.09062](https://arxiv.org/abs/2511.09062)
  (the arXiv listing is marked "Extended version"). It adds, among other things, the segmentation-based exact
  pricing algorithm (SegPrice). That solver and the game-abstraction network are not part of this repository, which
  contains the calibration part only.

> **Status: not yet re-verified.** Files were reorganised and renamed; they compile, but nothing has been executed end to end.
> This repository contains the calibration code only (paper §3 and §4), the crawler, and one month of new market data.
> The paper's own 2025 data is not included, so the paper's numbers cannot be reproduced from this repository.

## Layout

| Paper | Claim | Code |
|---|---|---|
| §3 Stackelberg routing game | user-side equilibrium (unique, Thm 1) as a convex potential minimisation | `priLLM/game/equilibrium.py` |
| §4 Data-driven calibration | learn `w_q, w_d, {b_j}` through the differentiable equilibrium (Thm 2); LP initialisation | `priLLM/calibration/cost_learner.py` (+ `real_data.py`, `build_master_df.py`) |
| §6 baseline (XGBoost) | non-game ML market simulation | `priLLM/calibration/xgboost_baseline.py` |
| Experiments | Fig. 3 flow prediction; Table 2 parameter-learning methods | `experiments/calibration/` |

## Data

`data/release/market_inputs_top20_month.csv`: 28 days of app-level token demand with TPS, delay, price and Arena score
for 20 models (apps anonymised). Definitions and sources: `data/release/README.md`. It is not the paper's data.
`priLLM/calibration/real_data.py` reads this file directly (apps are summed per model and day).
Example: `python -m priLLM.calibration.cost_learner --start_date 2026-10-19 --num_days 14`.

Some experiment inputs (e.g. Table 2, Fig. 3) are generated with seeded random numbers.

### Get the data yourself

`crawler/` collects the public OpenRouter ranking and app-usage pages (daily, standard library + `curl`), and
`data_collection/` joins them with endpoint speed, llm-bench-data probes and LMArena scores. See `crawler/README.md`
and `data_collection/README.md`.

## Setup

```bash
pip install -r requirements.txt
python -m experiments.calibration.fig3_flow_prediction --help
```
Run everything from the repository root.
