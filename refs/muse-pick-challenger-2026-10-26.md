# Muse's Pick Challenger — Lab Study (2026-10-26)

Kass's order (2026-10-26 ~18:31 ET): "I have a document in my Drive
called trading bot. RLML combinations I want you to go through
that and I want you to pick something and create your own next
challenger create a stacker and the brain for the Challenger. Make
sure to label it that it's just your pic."

## Provenance — the source doc

No Drive file carries Kass's stated title; the only RL+ML catalog
in his Drive is the spreadsheet **"Machine Learning and
Reinforcement Learning Kinds, Definitions, and Use Cases"**
(Drive id `1r5x4Ze25CFWYBQGPCvqSOp4f-UwY_99jyefhbD54Ld0`; tabs
"Table 1" + "Source References"). **Kass confirmed 2026-10-26
~18:35 ET it is the correct doc** ("the one that tells you all the
combinations").

### The four rows the pick is drawn from (Table 1, verbatim)

| Type | Kind/Framework | Definition | Key Features | Common Use Cases | Target Users | Source |
|---|---|---|---|---|---|---|
| ML Paradigm | Supervised Learning | A learning method where models are trained using labeled datasets to map inputs to known outputs. | Static pattern recognition, high fidelity to labeled ground truth, requires large labeled datasets. | Spam detection, image recognition, predicting housing prices, stable classification problems. | Teams with large historical datasets and stable prediction requirements. | 1-3 |
| RL Paradigm | Reinforcement Learning (RL) | A paradigm where an intelligent agent learns to make decisions by interacting with a dynamic environment. | Feedback via rewards/penalties, maximization of cumulative reward, exploration of new strategies, Markov Decision Process (MDP) modeling. | Game AI, robotics control, autonomous vehicle navigation, supply chain optimization. | Teams building autonomous systems capable of multi-step reasoning and self-optimization. | 1-4 |
| RL Paradigm/Technique | Reward Shaping | A technique that augments sparse rewards with additional intermediate signals to guide the learning process. | Provides intermediate feedback; utilizes distance-based metrics (e.g., Manhattan distance); balances shaping signals against main goal rewards. | Solving sparse reward problems; accelerating learning in grid worlds or robotics navigation. | RL practitioners; AI researchers seeking faster convergence. | 5 |
| RL Library | Stable Baselines3 | A set of reliable implementations of reinforcement learning algorithms in PyTorch based on OpenAI Gym. | High-quality implementations of PPO, DQN, and SAC; active maintenance; usability-focused API. | Prototyping RL agents, benchmarking standard algorithms on environments like MountainCar. | Researchers and developers who want to avoid reinventing the wheel with standardized implementations. | 1, 8 |

### Table 1 — full contents (all 15 rows, saved for provenance)

| Type | Kind/Framework | Definition | Key Features | Common Use Cases | Target Users | Source |
|---|---|---|---|---|---|---|
| ML Paradigm | Supervised Learning | A learning method where models are trained using labeled datasets to map inputs to known outputs. | Static pattern recognition, high fidelity to labeled ground truth, requires large labeled datasets. | Spam detection, image recognition, predicting housing prices, stable classification problems. | Teams with large historical datasets and stable prediction requirements. | 1-3 |
| RL Paradigm | Reinforcement Learning (RL) | A paradigm where an intelligent agent learns to make decisions by interacting with a dynamic environment. | Feedback via rewards/penalties, maximization of cumulative reward, exploration of new strategies, Markov Decision Process (MDP) modeling. | Game AI, robotics control, autonomous vehicle navigation, supply chain optimization. | Teams building autonomous systems capable of multi-step reasoning and self-optimization. | 1-4 |
| RL Paradigm/Technique | Reward Shaping | A technique that augments sparse rewards with additional intermediate signals to guide the learning process. | Provides intermediate feedback; utilizes distance-based metrics (e.g., Manhattan distance); balances shaping signals against main goal rewards. | Solving sparse reward problems; accelerating learning in grid worlds or robotics navigation. | RL practitioners; AI researchers seeking faster convergence. | 5 |
| ML Serving | Ray Serve | A scalable model-serving library built on top of Ray for building end-to-end AI applications and complex multi-model pipelines. | Model composition, independent auto-scaling, hardware utilization optimization, Python-native business logic integration, horizontal scalability. | Serving deep learning neural networks, recommendation systems, real-time ML inference, financial ML, and complex multi-step workflows. | Python-first teams, startups, and organizations with existing Ray investments or complex ML platforms. | 6, 7 |
| ML Serving | NVIDIA Triton Inference Server | An open-source software platform for deploying and serving AI models in production across various frameworks on GPU or CPU. | Supports TensorFlow, PyTorch, and ONNX; GPU/CPU acceleration; Model Analyzer; dynamic batching; concurrent model execution; TensorRT optimization. | Deploying efficient deep learning models at scale, real-time image classification, LLM serving, and GPU-intensive video processing. | Enterprise engineering teams requiring high reliability ( $99.999\%$ uptime), maximum GPU throughput, and low tail latency. | 1, 6, 7 |
| ML Serving | BentoML | An open-source platform for building, shipping, and scaling AI model serving infrastructure in production. | Multi-framework support (PyTorch, TensorFlow, scikit-learn, XGBoost), adaptive batching, automatic Docker container generation, model registry, REST/gRPC API. | Recommendation engines, dynamic pricing models, visual search capabilities in e-commerce. | Python-first ML teams needing flexible model serving and MLOps integration. | 7 |
| ML Serving | RayLLM | An LLM-serving solution built on Ray Serve designed to deploy and manage open-source Large Language Models. | OpenAI-compatible API, auto-scaling, multi-GPU support, multi-node inference, integration with LangChain. | Deploying pre-configured open-source LLMs or custom models in production. | Developers using LLM tooling like LangChain and LlamaIndex. | 6 |
| RL Library | Ray RLlib | An industry-grade reinforcement learning library built on the Ray framework for scalable RL development. | Distributed cluster scaling, multi-agent support, hyperparameter optimization, Kubernetes integration, Apache-2.0 license, DeveloperAPI for custom algorithms. | Robotics simulations, financial trading models, complex multi-agent warehouse simulations, large-scale training. | Enterprise-level RL projects and developers requiring distributed computing or custom RL algorithm development. | 1, 8, 9 |
| RL Library | Stable Baselines3 | A set of reliable implementations of reinforcement learning algorithms in PyTorch based on OpenAI Gym. | High-quality implementations of PPO, DQN, and SAC; active maintenance; usability-focused API. | Prototyping RL agents, benchmarking standard algorithms on environments like MountainCar. | Researchers and developers who want to avoid reinventing the wheel with standardized implementations. | 1, 8 |
| RL Library | OpenAI Gym | A toolkit for developing and comparing reinforcement learning algorithms by providing standard environments. | Standardized interface for continuous control tasks, support for benchmarks like HalfCheetah and Humanoid. | Evaluating RL agents on various physics-based simulation tasks. | RL researchers and benchmarking teams. | 10 |
| RL Library | OpenAI Baselines | A set of baseline implementations of reinforcement learning algorithms developed by OpenAI. | Published under MIT license; implements ordinary and prioritized experience replay; uses ring buffer structure with Python list storage. | Standard benchmarking for RL algorithms; transition storage and sampling. | Researchers and developers using baseline RL implementations. | 9 |
| RL Library | Chainer ChainerRL | A deep reinforcement learning library based on the Chainer framework. | MIT license; supports Nstep configuration; unique (non-duplicated) transition sampling; utilizes two-list storage for  $O(1)$  popping of oldest elements. | Trajectory tracking with Nstep; trajectory-based RL training. | Legacy Chainer users; developers needing unique transition sampling. | 9 |
| RL Component | Reverb | An efficient data storage and transport system (by DeepMind) designed for large-scale distributed reinforcement learning research. | Distributed server-client architecture, sharding, rate limiting, supports multiple selectors (Uniform, Prioritized, FIFO, LIFO), TensorFlow integration. | Experience replay for DQN and SAC, distributing model weights, production-level replay serving. | Machine learning researchers and enterprise engineering team building distributed actor-learner systems. | 9, 11, 12 |
| RL Component | Prioritized Experience Replay (PER) | A mechanism that samples important transitions from a replay buffer more frequently based on Temporal Difference (TD) error. | Stochastic prioritization, importance-sampling (IS) weights for bias correction, binary segment tree/SumTree data structures. | Stabilizing training and improving sample efficiency in Deep Q-Networks (DQN) and Rainbow DQN. | RL practitioners implementing deep off-policy algorithms. | 13-16 |
| RL Component | Hindsight Experience Replay (HER) | A replay-buffer strategy for goal-conditioned RL that relabels past experiences with alternative goals. | Goal relabeling, conversion of failures into pseudo-successes, independence from reward shaping. | Robotic manipulation (grasping, pushing), navigation tasks with sparse success signals. | Teams with limited data collection budgets or those working in sparse-reward environments. | 17 |

## The pick

- **Eyes — Muse's Pick Stacker** (Supervised Learning row): the
  production stacker recipe (StandardScaler → SMOTE → RF + XGBoost
  + LightGBM → LogisticRegression) with ONE added base learner,
  **ExtraTrees** (mirroring the RandomForest's 200 trees / depth 8),
  trained on Champion #2's scored-102 frame, graded against
  Champion #2 (run 64) re-scored on identical holdout timestamps.
  The single varied lever is ensemble diversity.
- **Brain — Muse's Pick PPO** (RL + Reward Shaping rows, SB3
  library row): SB3 PPO trained exactly like the scored-eyes
  study's 60k arm — FRESH brain (no warm start exists in that
  recipe across the study's construction; the 60k arm used
  `warm_start=False` with a nonexistent champion path, and this
  study states the same plainly), 60,000 steps, seed 42,
  `ent_coef=0.05`, 98-wide state (94 scored features mapped
  s/50 − 1 + the stacker vote + 3 portfolio readings) — consuming
  the PICK stacker's votes (cross-fitted with the pick recipe in
  training; the fitted pick stacker on the holdout) — plus the
  signature **reward shaping**:

      shaped step reward = standard dense reward
          − 0.5 × max(0, dd_t − dd_{t−1})

  where the standard dense reward is TradingEnv's per-step
  mark-to-market log return in percentage points clipped to
  [−1, 1], dd_t is the current episode's drawdown in percentage
  points against its running peak equity, and the standard
  terminal +1/−1 on episode P&L sign (and the env's standard
  activity penalty) ride along unchanged. Shaping is TRAINING
  ONLY, implemented in `ShapedTradingEnv`
  (src/models/muse_pick.py), a TradingEnv subclass swapped into
  ppo_train for this study's train call only; the duel grades both
  sides in the standard production env.

  **Rationale:** the promotion gate already grades
  score = P&L − 0.5 × max drawdown, yet all 51 prior PPO runs
  trained on raw mark-to-market alone. This challenger trains on
  a per-step decomposition of the objective it is graded by —
  every step that deepens the episode's drawdown pays half that
  deepening immediately, in the same units and at the same 0.5
  weight the gate applies to the episode's worst drawdown.

## Pre-registered design (one shot each — no tuning sweeps)

- Data: fresh BTC/USD 1h frame (Coinbase, 730d); holdout = last
  792 production-supervised rows (33 days), identical timestamps
  for every grade; train/holdout disjointness asserted.
- Stacker grading: holdout accuracy (strict > wins), precision,
  up-call count, walk-forward (5-fold TimeSeriesSplit) vs Champion
  #2 re-scored on the same timestamps; degenerate-win guard
  (a win with ≤7 up-calls or precision ≤0.05 is flagged DEGENERATE,
  run 63's treatment: recorded as a strict win, pending for Kass,
  anatomy on the record).
- PPO grading: the standard duel — 3 chronological windows × 10
  episodes, next-candle-open fills, score = P&L − 0.5 × maxDD;
  challenger pair (pick stacker + pick PPO) vs champion pair
  (run 44 PPO on prod21 windows + run 64 votes, as the scored-eyes
  study graded its champion side); gate formula (aggregate beats
  champion AND ≥2/3 windows) + Kass's two-part floor via
  `gate_decision()`, stated as the WOULD-BE verdict; under the
  promotion lock a pass records `pending_approval`, never an
  install.
- Telemetry: SB3 capture via ppo_train's `_TelemetryCallback`
  (logging-only), lifted into the PPO run's record.
- Governance: LAB ONLY. Guard hashes (lab champions, live seat,
  shadow seat) verified bit-identical before/after; both books
  never opened by the study (read-hash info only). Builder:
  `scripts/build_muse_pick.py`; machine record:
  `data/muse_pick/results.json`.

## Results

**Status: COMPLETE — split verdict. The pick STACKER won genuinely
and sits PENDING KASS APPROVAL (run 69); the pick PPO lost its
duel (run 70).** Registered as runs **69** (stacker) and **70**
(PPO), family `muse_pick` (dashboard family code 14), both
promoted=0. Nothing installed; guard hashes verified below.

### Construction (as built)

- Fresh BTC/USD 1h frame (Coinbase, 730d): 17,510 bars,
  2024-10-26 23:00 UTC → 2026-10-26 22:00 UTC; supervised 17,054
  rows; holdout = last 792 rows from 2026-09-23 23:00 UTC — the
  nightly split, disjointness asserted. Scored train frame 12,086
  rows (the same count run 64's study trained on; 4,176 warmup
  rows dropped — the 180-day percentile window's honest cost).
- Causality (v5.1 standard, re-run on this build): truncation
  rebuilds at bars 9,000 and 17,018 — sampled scored columns exact
  on every shared row and run 64's votes identical (max diff 0.0)
  on 200 shared rows at each cut. PASS.
- Votes: pick-recipe cross-fit on the train frame — mean 0.498,
  std 0.013, leading-segment neutral share 0.200 (the expected
  1/5); holdout votes mean 0.500 (pick) / 0.499 (run 64).

### Stacker grading (identical holdout timestamps; brains named)

| model | holdout acc | precision | walk-forward acc | pred "up" |
|---|---|---|---|---|
| Champion #2 stacker (run 64, re-scored) | 0.4975 | 0.4867 | 0.5121 (recorded) | 339/792 |
| **Muse's Pick Stacker (run 69)** | **0.5265** | **0.5167** | 0.5022 | **418/792** |

- **Strict-win rule: WIN, +0.0290.** The re-score is genuine, not
  a copy of the record: on these fresher timestamps run 64's
  accuracy lands on the same 394/792 but its precision (0.4867 vs
  recorded 0.4854) and up-calls (339 vs 342) moved.
- **Degenerate guard: NOT degenerate.** 418 up-calls and the best
  precision in the study — a real two-sided classifier. (Its
  P(up) max is 0.524: the whole stack family votes in a narrow
  band around 0.50; the pick's edge is in the calls it makes, not
  in conviction.)
- Anatomy: all four base learners beat the champion stacker on
  the holdout (XGB 0.5341, ExtraTrees 0.5265, LGBM 0.5240, RF
  0.5177); the meta-learner leans on the new learner (ET +0.97)
  and RF (+0.78), against XGB (−0.15) and LGBM (−0.36). Top
  features by mean importance: Stochastic RSI, Distance to
  $5,000 Round Number, Force Index, Williams %R, Choppiness —
  the same flat, oscillator-led profile as runs 63/64.
- **The honest caveat:** the walk-forward ordering disagrees with
  the holdout. The pick's WF (0.5022, on this study's train frame)
  sits below Champion #2's recorded WF mark (0.5121, on its own
  study's frame — a different, overlapping window set). The
  strict rule grades on the shared holdout and the pick won it by
  the largest genuine stacker margin of the clean era; the WF
  conflict is on the record for Kass's decision, and the
  full-indicator study's own warning applies: single-holdout
  margins are single-holdout sized.

### PPO duel (brains named; standard duel, both sides graded in
the production env — shaping was training-only)

Buy-and-hold over the duel windows: **−$5,139.39** (a falling
holdout). Champion side recomputed fresh (its scored-eyes record
was +39.76 / +$461.81 on that study's holdout; the shared window
of the two studies reproduces exactly: window 2 −114.46 both
times).

| Side | Agg score | P&L | MaxDD | Windows (score) | Mix (last window) |
|---|---|---|---|---|---|
| Champion pair (run 44 PPO, prod21 + run 64 votes) | **+14.30** | **+$385.43** | $350.63 | +238.68 / −114.46 / −81.32 | hold .58 / buy 0 / sell .41 / short .01 |
| Muse's Pick pair (run 70 PPO + pick stacker votes) | −1,325.94 | −$2,025.21 | $1,863.52 | +250.84 / −535.42 / −3,693.24 | hold .41 / buy .11 / sell .43 / short .04 |

- **Formula: FAIL — 1/3 windows.** The two-part floor never
  engaged (it is only evaluated on a formula pass). Verdict
  recorded: champion kept; `pending_approval=false`.
- The shaped brain actually won window 1 (+250.84 vs +238.68) and
  made money in window 2 (+$122.17 P&L) — then window 3 destroyed
  it (−$2,761.48 P&L, $1,863.52 drawdown): it traded far more
  than the champion (0.70 vs 0.10 round trips/day) into the
  falling window and the drawdown it was trained to avoid is
  exactly what it produced. Read plainly: penalizing drawdown
  *increases* during 24-step training episodes did not teach
  drawdown control at duel scale — episode peaks reset daily in
  training, while the duel's damage compounds across a 264-candle
  window the reward never sees. The shaping hypothesis is tested
  and closed, the same closure pattern as the scored-eyes study:
  the varied lever was not the binding constraint.

### Telemetry (run 70, captured by the logging-only callback)

118 rollouts over 60,000 steps. Policy entropy 1.3839 → 1.0516
(min 0.971) — no entropy collapse; the shaped brain kept
exploring throughout, so its loss is not an exploration failure.
Clip fraction (end) 0.0646, explained variance 0.2194, value loss
0.4847. Approx KL was not in SB3's logged values for this run and
is recorded null, not reconstructed. The run surfaces as the
first row of the Training Lab's telemetry panel (`from: 70`).

### Two-way verification

- Duel arithmetic reconciles two ways (asserted in the builder):
  each side's aggregate = mean of its window scores, totals = sum
  of window P&Ls, and per-window buy-and-hold recomputed straight
  from the closes matches both sides to 1e-6.
- Stacker grades read back from training.db after registration
  (run 69: acc 0.5265 / pending true; run 70: duel −1325.9 /
  pending false / telemetry present) and match results.json;
  the dashboard feed generated from the registry reproduces both
  rows in All Models — "Muse's Pick Stacker" PASS, "Muse's Pick
  PPO" FAIL, family code 14 — and the gate read back from both
  dashboards still names run 68 (the newest NIGHTLY run) as its
  challenger, with `pending` = run 69.

## Perimeter verification

Guard hashes (sha256, first 8) — builder's guard_before ==
guard_after == live files at completion:

| File | Hash |
|---|---|
| models/ppo_champion.zip (lab) | e610ddf3 |
| models/stacker_champion.pkl (lab, run 64) | c0373d15 |
| models/stacker_champion.json | 52a93161 |
| models/live_seat/stacker.pkl (run 40) | 0dd62fc4 |
| models/live_seat/stacker.json | 4cc2147e |
| models/live_seat/ppo.zip | e610ddf3 |
| models/shadow_seat/* (stacker/meta/ppo) | c0373d15 / 52a93161 / e610ddf3 |

The paper books' hashes changed during the study window only
because the live loops write them on candle closes; the study
never opened either book (read-hash, info only).

## Execution incidents (recorded for the paper trail)

- A VM restart (~23:00 UTC, uptime reset) killed the first
  builder process mid-way through the PPO train phase. The
  checkpointed phases (data/frames/stacker/votes) survived on
  disk; one relaunch resumed at the train phase and ran to
  completion. No phase was duplicated (registration is
  idempotence-guarded and the registry holds exactly one
  muse_pick stacker run and one muse_pick PPO run).
- The training feed grew to 11,630B at 70 runs, tripping the
  11,400B size sentinel; re-baselined to 11,950B with the basis
  recorded in tests/test_compact_feeds.py, per that test's
  documented pattern (hard budget 16,384B untouched). Two
  incorporation tests that had frozen the pre-study registry
  state as literals (nothing pending; telemetry block empty)
  were updated to the new recorded facts (run 69 pending;
  run 70 the first telemetry row) — the audit/telemetry builders
  themselves needed no change: both blocks read the registry
  verbatim, which is what the updated tests now assert.
- Final suite after the study: **274 passed** (266 baseline + 8
  new tests in tests/test_muse_pick.py, which pin the
  hand-computed shaping math, the lockstep shaped-vs-production
  env identity, the 4-learner pipeline, and the family-14
  dashboard rendering). One baseline run before the study showed
  a transient failure in
  test_shadow_desk.py::test_shadow_vote_matches_run64_computed_the_study_way;
  it passed standalone and in both full runs after the study —
  recorded here so the flake is on the paper trail, not hidden.
