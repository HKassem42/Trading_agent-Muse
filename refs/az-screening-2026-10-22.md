# A-Z Indicator Screening — 2026-10-22

Kass's directive (2026-10-22): "include all these features to test" — the
A-Z indicator encyclopedia (`refs/az-indicator-encyclopedia.md`, ~300 entries)
gets its own screening program. Every OHLCV-computable indicator was
implemented causally and screened solo + marginal-contribution on clean data
under multiple-comparison discipline. **Screening only**: trading is PARKED;
nothing here was promoted or registered as a challenger. Machine-readable
results: `data/az_screen/results.json` (script: `scripts/az_screen.py`,
library: `src/features/az_indicators.py`, tests: `tests/test_az_indicators.py`).

## Verdict

**Graduates: NONE (0 of 132).**

- 19 of 132 indicators reach raw p < 0.05 on the solo test;
  6.6 would be expected by chance alone at that
  threshold, so a raw-significant list is mostly noise by construction.
- Exactly **1** indicator survives Bonferroni (alpha = 0.05/132 = 3.79e-04):
  **az_aroon_down** (Aroon Down(14)) — solo walk-forward accuracy 0.5181, binomial
  p = 1.35e-04, BH q = 0.0179, Spearman IC +0.0202 ± 0.0078, non-duplicate
  (max |r| vs the production 21 = 0.583). It is nevertheless **not a graduate**:
  its marginal delta is **−0.0042** — adding it to the full-21 stacker made
  walk-forward accuracy slightly *worse*. It carries a detectable solo trace and
  negative marginal value in combination. Both facts are in the table; the
  graduation rule requires both.
- 16 of the 132 are near-duplicates of production features already in the
  model (max |r| > 0.95 vs the 21), including the two deliberate positive
  controls (az_rsi_14 ≈ rsi_c at 0.999, az_atr_pct ≈ atr_rel at 0.997) — the
  redundancy detector flags known ground truth correctly.
- The remaining raw-significant names fail Bonferroni, and several are
  "significant" only in the two-sided sense with walk-forward accuracy below
  0.50 (az_cg_10 0.488, az_roc_12 0.491) — anti-signal at best, noise at worst.

Bottom line: after screening the full OHLCV-computable encyclopedia, no
indicator demonstrates signal that (a) survives multiplicity correction,
(b) is not already in the model under another name, and (c) adds anything
on top of the existing 21 features. This is the third independent null on
1h BTC (per-candle TA stacker, HA-sequence network, now the encyclopedia).

## Methods

- **Data**: same cached supervised frame as the diagnosis battery
  (`data/diagnosis/sup.pkl`, 17049 rows, 2024-11-09 23:00:00 → 2026-10-21 17:00:00,
  Coinbase BTC-USD 1h). Production split: train = first 16257 rows,
  holdout = last 792 (screening grades on the train frame via
  walk-forward, exactly as production stacker grading does).
- **Implementations**: pandas_ta 0.4.71b0 wrappers with explicit canonical
  parameters + hand implementations where pandas_ta lacks the indicator or
  its default is non-causal (TA-Lib not installed). Every builder is causal
  (value at row t is a function of rows ≤ t only); scaling to the [−1, 1]
  contract uses statistics fitted on the TRAIN split only (natural bounds /
  unit / robust train-median-and-percentile scaling), pinned by
  `tests/test_az_indicators.py`. Full suite: 41/41 passing.
- **Per indicator**: (i) solo walk-forward single-feature stacker
  (`stacker.build_pipeline`, seed 42, TimeSeriesSplit 5 folds) — per-fold
  accuracies, pooled out-of-fold accuracy, and a two-sided binomial p-value
  against the OOF majority-class rate; (ii) Spearman IC vs next-candle
  return per fold (mean ± SE) plus pooled, as a model-free second angle;
  (iii) marginal paired delta: full-21 + indicator vs full-21 on identical
  rows/folds (mean ± SE); (iv) redundancy: max |Pearson r| of the scaled
  series vs each production feature and vs every other screened series
  (train rows); > 0.95 = duplicate-class.
- **Multiple comparisons**: N = 132 tests. Bonferroni alpha = 3.79e-04;
  Benjamini-Hochberg q-values reported alongside. **Graduation rule**
  (fixed in the script before any results were seen): binomial p <
  Bonferroni alpha AND solo WF accuracy ≥ 0.51 AND pooled OOF accuracy
  above the majority rate AND max |r| vs the 21 ≤ 0.95 AND marginal
  delta ≥ 0.

## The full table (sorted by binomial p, best first)

| Indicator | Family | Definition / source | Solo WF acc | Pooled OOF acc (majority) | Binom p | BH q | IC mean ± SE | Marginal Δ ± SE | max \|r\| vs 21 (nearest) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| az_aroon_down | trend | Aroon Down(14), pandas_ta | 0.5181 | 0.5181 (0.5017) | 1.35e-04 | 0.018 | +0.0202 ± 0.0078 | -0.0042 ± 0.0039 | 0.583 (rsi_c) | significant but fails effect/marginal rule |
| az_bear_power | statistical | Elder Bear Power = (low - EMA13)/close, hand | 0.5160 | 0.5160 (0.5017) | 8.56e-04 | 0.057 | -0.0361 ± 0.0023 | -0.0134 ± 0.0036 | 0.858 (dist_vwap) | not significant after Bonferroni |
| az_cg_10 | momentum | Center of Gravity(10), Ehlers, pandas_ta | 0.4880 | 0.4880 (0.5017) | 1.52e-03 | 0.067 | +0.0272 ± 0.0035 | +0.0016 ± 0.0013 | 0.865 (macd_hist_rel) | not significant after Bonferroni |
| az_kdj_j | momentum | KDJ J line = 3K - 2D of stochastic(14,3,3) | 0.5144 | 0.5144 (0.5016) | 2.77e-03 | 0.092 | -0.0351 ± 0.0025 | +0.0009 ± 0.0020 | 0.740 (ha_dir) | not significant after Bonferroni |
| az_donchian_pos | structure | Position of close in Donchian channel(20), pandas_ta | 0.5140 | 0.5140 (0.5016) | 3.87e-03 | 0.102 | -0.0406 ± 0.0061 | +0.0023 ± 0.0046 | 0.859 (rsi_c) | not significant after Bonferroni |
| az_kama_10_dist | ma_distance | (close - KAMA(10))/close, pandas_ta | 0.5138 | 0.5138 (0.5017) | 4.82e-03 | 0.106 | -0.0426 ± 0.0032 | +0.0004 ± 0.0042 | 0.853 (dist_vwap) | not significant after Bonferroni |
| az_qstick_10 | statistical | QStick(10) = SMA of (close-open), pandas_ta | 0.5129 | 0.5129 (0.5017) | 8.99e-03 | 0.169 | -0.0340 ± 0.0044 | +0.0004 ± 0.0041 | 0.845 (dist_vwap) | not significant after Bonferroni |
| az_alligator_lips | structure | (close - SMMA5 of median price)/close, unshifted, pandas_ta | 0.5125 | 0.5125 (0.5017) | 1.21e-02 | 0.176 | -0.0451 ± 0.0028 | +0.0052 ± 0.0063 | 0.864 (dist_vwap) | not significant after Bonferroni |
| az_roc_12 | momentum | ROC(12) %, pandas_ta | 0.4909 | 0.4909 (0.5017) | 1.24e-02 | 0.176 | -0.0295 ± 0.0023 | +0.0022 ± 0.0043 | 0.881 (dist_vwap) | not significant after Bonferroni |
| az_wclose_dev | statistical | (close - SMA14 of weighted close)/close, hand | 0.5123 | 0.5123 (0.5017) | 1.33e-02 | 0.176 | -0.0374 ± 0.0029 | -0.0045 ± 0.0018 | 0.907 (dist_vwap) | not significant after Bonferroni |
| az_obv_mom_10 | volume | OBV 10-bar change / 10-bar volume, pandas_ta OBV | 0.5118 | 0.5118 (0.5017) | 1.85e-02 | 0.222 | -0.0384 ± 0.0092 | -0.0041 ± 0.0063 | 0.670 (rsi_c) | not significant after Bonferroni |
| az_squeeze_mom | statistical | TTM Squeeze momentum (BB/KC), pandas_ta | 0.5114 | 0.5114 (0.5016) | 2.22e-02 | 0.244 | -0.0227 ± 0.0057 | -0.0010 ± 0.0029 | 0.836 (ewo) | not significant after Bonferroni |
| az_vol_osc | volume | Volume Oscillator = (EMA5-EMA20)/EMA20 of volume, hand | 0.4924 | 0.4924 (0.5017) | 3.24e-02 | 0.310 | +0.0153 ± 0.0063 | -0.0064 ± 0.0042 | 0.518 (range_pct) | not significant after Bonferroni |
| az_adosc | volume | Chaikin Oscillator / A-D Oscillator(3,10), pandas_ta | 0.5109 | 0.5109 (0.5017) | 3.31e-02 | 0.310 | -0.0419 ± 0.0052 | +0.0002 ± 0.0033 | 0.638 (adl_slope) | not significant after Bonferroni |
| az_pvt_mom_10 | volume | PVT 10-bar change / 10-bar volume, pandas_ta PVT | 0.4926 | 0.4926 (0.5017) | 3.52e-02 | 0.310 | -0.0284 ± 0.0068 | -0.0073 ± 0.0065 | 0.780 (dist_vwap) | not significant after Bonferroni |
| az_crsi | momentum | Connors RSI(3,2,100), pandas_ta | 0.5100 | 0.5100 (0.5013) | 4.37e-02 | 0.340 | -0.0373 ± 0.0052 | -0.0046 ± 0.0041 | 0.799 (price_change) | not significant after Bonferroni |
| az_pivot_s1_dist | levels | (close - prior-day S1)/close, hand | 0.4932 | 0.4932 (0.5017) | 4.91e-02 | 0.340 | -0.0299 ± 0.0038 | +0.0012 ± 0.0049 | 0.696 (dist_vwap) | not significant after Bonferroni |
| az_klinger_diff | volume | Klinger Oscillator(34,55) minus signal EMA(13), hand | 0.5100 | 0.5100 (0.5016) | 4.96e-02 | 0.340 | -0.0281 ± 0.0085 | +0.0060 ± 0.0054 | 0.691 (macd_hist_rel) | not significant after Bonferroni |
| az_cci_20 | momentum | CCI(20), pandas_ta | 0.5100 | 0.5100 (0.5016) | 5.00e-02 | 0.340 | -0.0217 ± 0.0092 | -0.0020 ± 0.0034 | 0.383 (volatility_21) | not significant after Bonferroni |
| az_zscore_50 | statistical | Price z-score(50), hand | 0.5100 | 0.5100 (0.5017) | 5.18e-02 | 0.340 | -0.0312 ± 0.0055 | -0.0000 ± 0.0048 | 0.934 (rsi_c) | not significant after Bonferroni |
| az_ema_20_dist | ma_distance | (close - EMA(20))/close, pandas_ta | 0.5098 | 0.5098 (0.5016) | 5.41e-02 | 0.340 | -0.0376 ± 0.0021 | -0.0061 ± 0.0053 | 0.946 (dist_vwap) | not significant after Bonferroni |
| az_smi | momentum | Stochastic Momentum Index, pandas_ta | 0.5095 | 0.5095 (0.5016) | 6.33e-02 | 0.375 | -0.0310 ± 0.0027 | -0.0018 ± 0.0053 | 0.964 (rsi_c) | duplicate of production feature |
| az_disparity_14 | momentum | Disparity Index(14) = (close-SMA14)/SMA14 | 0.5095 | 0.5095 (0.5017) | 6.84e-02 | 0.375 | -0.0373 ± 0.0030 | -0.0118 ± 0.0037 | 0.906 (dist_vwap) | not significant after Bonferroni |
| az_keltner_pos | structure | Position of close in Keltner channel(20,2), pandas_ta kc | 0.5093 | 0.5093 (0.5016) | 7.10e-02 | 0.375 | -0.0357 ± 0.0020 | -0.0036 ± 0.0038 | 0.970 (rsi_c) | duplicate of production feature |
| az_ibs | statistical | Internal Bar Strength = (close-low)/(high-low), hand | 0.5094 | 0.5094 (0.5017) | 7.12e-02 | 0.375 | -0.0412 ± 0.0057 | -0.0030 ± 0.0034 | 0.666 (price_change) | not significant after Bonferroni |
| az_rel_vol_index | momentum | Relative Volatility Index (Beyers), rolling-std form, hand | 0.5093 | 0.5093 (0.5017) | 7.38e-02 | 0.375 | -0.0161 ± 0.0045 | -0.0040 ± 0.0060 | 0.769 (rsi_c) | not significant after Bonferroni |
| az_bwmfi | volume | Market Facilitation Index = (high-low)/volume (Bill Williams), hand | 0.4941 | 0.4941 (0.5017) | 8.11e-02 | 0.396 | -0.0115 ± 0.0044 | -0.0087 ± 0.0046 | 0.266 (atr_rel) | not significant after Bonferroni |
| az_hma_20_dist | ma_distance | (close - HMA(20))/close, pandas_ta | 0.5089 | 0.5089 (0.5015) | 8.71e-02 | 0.396 | -0.0341 ± 0.0089 | +0.0061 ± 0.0049 | 0.800 (ha_dir) | not significant after Bonferroni |
| az_ichimoku_kijun | structure | (close - Kijun-sen 26)/close, hand | 0.5089 | 0.5089 (0.5015) | 8.71e-02 | 0.396 | -0.0370 ± 0.0072 | +0.0067 ± 0.0029 | 0.936 (dist_vwap) | not significant after Bonferroni |
| az_alma_20_dist | ma_distance | (close - ALMA(20))/close, pandas_ta | 0.5088 | 0.5088 (0.5016) | 9.36e-02 | 0.412 | -0.0438 ± 0.0045 | -0.0033 ± 0.0025 | 0.852 (ha_dir) | not significant after Bonferroni |
| az_acf_1_50 | statistical | Autocorrelation of returns, lag 1, window 50, hand | 0.4946 | 0.4946 (0.5017) | 1.02e-01 | 0.435 | -0.0032 ± 0.0101 | +0.0029 ± 0.0048 | 0.226 (volatility_21) | not significant after Bonferroni |
| az_kst | momentum | Know Sure Thing line, pandas_ta | 0.4950 | 0.4950 (0.5016) | 1.26e-01 | 0.513 | -0.0196 ± 0.0093 | -0.0079 ± 0.0063 | 0.887 (dist_sma50) | not significant after Bonferroni |
| az_alligator_teeth | structure | (close - SMMA8 of median price)/close, unshifted, pandas_ta | 0.5083 | 0.5083 (0.5017) | 1.28e-01 | 0.513 | -0.0411 ± 0.0012 | +0.0044 ± 0.0041 | 0.937 (dist_vwap) | not significant after Bonferroni |
| az_er_10 | trend | Kaufman Efficiency Ratio(10), hand | 0.5082 | 0.5082 (0.5017) | 1.33e-01 | 0.514 | -0.0057 ± 0.0103 | -0.0034 ± 0.0045 | 0.276 (volatility_21) | not significant after Bonferroni |
| az_wma_20_dist | ma_distance | (close - WMA(20))/close, pandas_ta | 0.5079 | 0.5079 (0.5016) | 1.42e-01 | 0.534 | -0.0382 ± 0.0022 | -0.0025 ± 0.0042 | 0.923 (dist_vwap) | not significant after Bonferroni |
| az_pvi_ret_20 | volume | Positive Volume Index 20-bar % change, pandas_ta | 0.4959 | 0.4959 (0.5016) | 1.86e-01 | 0.672 | -0.0079 ± 0.0127 | -0.0057 ± 0.0033 | 0.586 (body_pct) | not significant after Bonferroni |
| az_vwma_dist | volume | (close - VWMA20)/close, pandas_ta | 0.5072 | 0.5072 (0.5016) | 1.88e-01 | 0.672 | -0.0347 ± 0.0031 | -0.0098 ± 0.0014 | 0.980 (dist_vwap) | duplicate of production feature |
| az_alligator_jaw | structure | (close - SMMA13 of median price)/close, unshifted, pandas_ta | 0.5073 | 0.5073 (0.5017) | 1.94e-01 | 0.675 | -0.0363 ± 0.0029 | +0.0062 ± 0.0026 | 0.932 (dist_vwap) | not significant after Bonferroni |
| az_ema_50_dist | ma_distance | (close - EMA(50))/close, pandas_ta | 0.5072 | 0.5072 (0.5017) | 2.00e-01 | 0.676 | -0.0347 ± 0.0060 | +0.0001 ± 0.0041 | 0.982 (dist_sma50) | duplicate of production feature |
| az_bb_bw | volatility | Bollinger BandWidth(20,2), pandas_ta | 0.4962 | 0.4962 (0.5016) | 2.16e-01 | 0.712 | -0.0009 ± 0.0107 | -0.0053 ± 0.0040 | 0.981 (volatility_21) | duplicate of production feature |
| az_ichimoku_tenkan | structure | (close - Tenkan-sen 9)/close, hand | 0.5069 | 0.5069 (0.5017) | 2.32e-01 | 0.729 | -0.0433 ± 0.0048 | +0.0039 ± 0.0036 | 0.786 (dist_vwap) | not significant after Bonferroni |
| az_roc_6 | momentum | ROC(6) %, pandas_ta | 0.5068 | 0.5068 (0.5017) | 2.39e-01 | 0.729 | -0.0392 ± 0.0066 | -0.0004 ± 0.0067 | 0.763 (dist_vwap) | not significant after Bonferroni |
| az_plus_di | trend | +DI(14), pandas_ta | 0.4966 | 0.4966 (0.5017) | 2.42e-01 | 0.729 | -0.0246 ± 0.0049 | -0.0045 ± 0.0058 | 0.875 (rsi_c) | not significant after Bonferroni |
| az_aroon_up | trend | Aroon Up(14), pandas_ta | 0.5067 | 0.5067 (0.5017) | 2.46e-01 | 0.729 | -0.0090 ± 0.0051 | +0.0029 ± 0.0047 | 0.591 (rsi_c) | not significant after Bonferroni |
| az_dss_10 | momentum | Double-Smoothed Stochastic (Bressert), hand | 0.4967 | 0.4967 (0.5016) | 2.56e-01 | 0.729 | -0.0282 ± 0.0086 | +0.0017 ± 0.0027 | 0.527 (ha_dir) | not significant after Bonferroni |
| az_willr_14 | momentum | Williams %R(14), pandas_ta | 0.5065 | 0.5065 (0.5017) | 2.60e-01 | 0.729 | -0.0424 ± 0.0034 | -0.0021 ± 0.0029 | 0.783 (rsi_c) | not significant after Bonferroni |
| az_hist_vol_20 | volatility | Historical volatility: rolling std(20) of log returns, hand | 0.4967 | 0.4967 (0.5016) | 2.64e-01 | 0.729 | +0.0059 ± 0.0107 | -0.0006 ± 0.0031 | 0.885 (atr_rel) | not significant after Bonferroni |
| az_starc_pos | structure | (close - SMA20) / (2*ATR10), STARC position, hand | 0.5064 | 0.5064 (0.5016) | 2.67e-01 | 0.729 | -0.0340 ± 0.0018 | -0.0055 ± 0.0026 | 0.946 (rsi_c) | not significant after Bonferroni |
| az_dow_sin | calendar | sin(2*pi*dayofweek/7) | 0.4969 | 0.4969 (0.5017) | 2.71e-01 | 0.729 | -0.0023 ± 0.0108 | -0.0028 ± 0.0025 | 0.265 (atr_rel) | not significant after Bonferroni |
| az_sma_200_dist | ma_distance | (close - SMA(200))/close, pandas_ta | 0.4968 | 0.4968 (0.5014) | 2.84e-01 | 0.729 | -0.0211 ± 0.0096 | +0.0022 ± 0.0052 | 0.882 (h4_rsi_c) | not significant after Bonferroni |
| az_apo_rel | momentum | APO = (EMA12-EMA26)/close (MACD line, price-relative) | 0.5062 | 0.5062 (0.5017) | 2.91e-01 | 0.729 | -0.0267 ± 0.0067 | -0.0036 ± 0.0015 | 0.985 (ewo) | duplicate of production feature |
| az_pv_corr_20 | statistical | Rolling corr(20) of price returns vs volume returns, hand | 0.4970 | 0.4970 (0.5016) | 2.94e-01 | 0.729 | -0.0127 ± 0.0033 | -0.0072 ± 0.0053 | 0.619 (ewo) | not significant after Bonferroni |
| az_psar_dist | trend | Signed (close - Parabolic SAR)/close, pandas_ta | 0.4972 | 0.4972 (0.5017) | 3.03e-01 | 0.729 | -0.0314 ± 0.0051 | +0.0000 ± 0.0018 | 0.744 (macd_hist_rel) | not significant after Bonferroni |
| az_donchian_width | structure | Donchian channel width / close, pandas_ta | 0.4972 | 0.4972 (0.5016) | 3.10e-01 | 0.729 | +0.0001 ± 0.0099 | -0.0057 ± 0.0023 | 0.935 (volatility_21) | not significant after Bonferroni |
| az_ret_kurt_20 | statistical | Rolling kurtosis(20) of returns, hand | 0.4972 | 0.4972 (0.5016) | 3.10e-01 | 0.729 | +0.0087 ± 0.0106 | -0.0055 ± 0.0017 | 0.198 (volatility_21) | not significant after Bonferroni |
| az_dow_cos | calendar | cos(2*pi*dayofweek/7) | 0.4973 | 0.4973 (0.5017) | 3.11e-01 | 0.729 | +0.0032 ± 0.0104 | -0.0060 ± 0.0060 | 0.164 (h4_atr_rel) | not significant after Bonferroni |
| az_hour_cos | calendar | cos(2*pi*hour/24) | 0.5060 | 0.5060 (0.5017) | 3.15e-01 | 0.729 | -0.0005 ± 0.0083 | -0.0114 ± 0.0037 | 0.186 (volume_change) | not significant after Bonferroni |
| az_linreg_slope_20 | statistical | Linear regression slope(20)/close, pandas_ta | 0.4973 | 0.4973 (0.5016) | 3.27e-01 | 0.744 | -0.0277 ± 0.0086 | +0.0017 ± 0.0026 | 0.898 (ewo) | not significant after Bonferroni |
| az_typical_dev | statistical | (close - SMA20 of typical price)/close, hand | 0.5056 | 0.5056 (0.5016) | 3.49e-01 | 0.773 | -0.0356 ± 0.0016 | +0.0015 ± 0.0035 | 0.955 (dist_vwap) | duplicate of production feature |
| az_uo | momentum | Ultimate Oscillator(7,14,28), pandas_ta | 0.5055 | 0.5055 (0.5015) | 3.58e-01 | 0.773 | -0.0401 ± 0.0053 | +0.0030 ± 0.0093 | 0.653 (rsi_c) | not significant after Bonferroni |
| az_pivot_dist | levels | (close - classic Pivot Point of prior completed day)/close, hand | 0.4977 | 0.4977 (0.5017) | 3.62e-01 | 0.773 | -0.0310 ± 0.0062 | -0.0007 ± 0.0022 | 0.902 (dist_sma50) | not significant after Bonferroni |
| az_vroc_14 | volume | Volume Rate of Change(14), hand | 0.4978 | 0.4978 (0.5017) | 3.71e-01 | 0.773 | +0.0104 ± 0.0070 | +0.0007 ± 0.0025 | 0.483 (range_pct) | not significant after Bonferroni |
| az_garman_klass_20 | volatility | Garman-Klass OHLC volatility estimator(20), hand | 0.4978 | 0.4978 (0.5016) | 3.81e-01 | 0.773 | +0.0017 ± 0.0078 | -0.0074 ± 0.0011 | 0.957 (atr_rel) | duplicate of production feature |
| az_supertrend_dir | trend | SuperTrend(10,3) direction, pandas_ta | 0.4979 | 0.4979 (0.5017) | 3.81e-01 | 0.773 | -0.0231 ± 0.0067 | +0.0058 ± 0.0038 | 0.657 (ewo) | not significant after Bonferroni |
| az_tsi | momentum | True Strength Index (25,13) double-smoothed, hand | 0.4979 | 0.4979 (0.5017) | 3.81e-01 | 0.773 | -0.0251 ± 0.0046 | +0.0026 ± 0.0021 | 0.911 (ewo) | not significant after Bonferroni |
| az_medprice_dev | statistical | (close - SMA14 of median price)/close, hand | 0.4980 | 0.4980 (0.5017) | 3.90e-01 | 0.780 | -0.0375 ± 0.0028 | -0.0061 ± 0.0043 | 0.909 (dist_vwap) | not significant after Bonferroni |
| az_atr_pct | volatility | ATR(14)/close, pandas_ta [positive control: production atr_rel] | 0.5053 | 0.5053 (0.5017) | 4.04e-01 | 0.797 | +0.0065 ± 0.0072 | -0.0006 ± 0.0040 | 0.997 (atr_rel) | duplicate of production feature |
| az_vidya_14_dist | ma_distance | (close - VIDYA(14))/close, pandas_ta | 0.5051 | 0.5051 (0.5017) | 4.34e-01 | 0.797 | -0.0397 ± 0.0044 | -0.0053 ± 0.0030 | 0.938 (dist_sma50) | not significant after Bonferroni |
| az_hour_sin | calendar | sin(2*pi*hour/24) | 0.4983 | 0.4983 (0.5017) | 4.39e-01 | 0.797 | -0.0075 ± 0.0049 | -0.0080 ± 0.0041 | 0.212 (range_pct) | not significant after Bonferroni |
| az_pat_doji | pattern | Doji flag: body <= 10% of range | 0.4983 | 0.4983 (0.5017) | 4.39e-01 | 0.797 | -0.0088 ± 0.0072 | -0.0068 ± 0.0031 | 0.152 (lower_wick) | not significant after Bonferroni |
| az_thermo | volatility | Elder Thermometer, pandas_ta | 0.4983 | 0.4983 (0.5017) | 4.39e-01 | 0.797 | +0.0116 ± 0.0067 | +0.0041 ± 0.0008 | 0.721 (range_pct) | not significant after Bonferroni |
| az_linreg_dev | statistical | (close - linear regression value(20))/close, pandas_ta | 0.4983 | 0.4983 (0.5016) | 4.49e-01 | 0.797 | -0.0313 ± 0.0097 | +0.0013 ± 0.0015 | 0.730 (ha_dir) | not significant after Bonferroni |
| az_ma_cross_spread | ma_distance | (SMA50 - SMA200)/close, pandas_ta | 0.5047 | 0.5047 (0.5014) | 4.52e-01 | 0.797 | -0.0037 ± 0.0094 | +0.0019 ± 0.0040 | 0.807 (d1_dist_sma) | not significant after Bonferroni |
| az_pat_shooting_star | pattern | Shooting star flag (encoded -1): upper wick >= 2x body | 0.4985 | 0.4985 (0.5017) | 4.60e-01 | 0.797 | -0.0096 ± 0.0060 | -0.0024 ± 0.0048 | 0.278 (upper_wick) | not significant after Bonferroni |
| az_pat_three | pattern | Three white soldiers (+1) / three black crows (-1), strong-body form | 0.4985 | 0.4985 (0.5017) | 4.60e-01 | 0.797 | -0.0159 ± 0.0067 | -0.0041 ± 0.0048 | 0.254 (ha_dir) | not significant after Bonferroni |
| az_sma_20_dist | ma_distance | (close - SMA(20))/close, pandas_ta | 0.5047 | 0.5047 (0.5016) | 4.65e-01 | 0.797 | -0.0354 ± 0.0016 | -0.0045 ± 0.0039 | 0.955 (dist_vwap) | duplicate of production feature |
| az_t3_10_dist | ma_distance | (close - T3(10))/close, pandas_ta | 0.5049 | 0.5049 (0.5017) | 4.65e-01 | 0.797 | -0.0350 ± 0.0061 | -0.0058 ± 0.0037 | 0.875 (macd_hist_rel) | not significant after Bonferroni |
| az_pat_hammer | pattern | Hammer flag: lower wick >= 2x body, upper <= body | 0.4986 | 0.4986 (0.5017) | 4.81e-01 | 0.814 | -0.0011 ± 0.0083 | -0.0029 ± 0.0044 | 0.278 (lower_wick) | not significant after Bonferroni |
| az_chaikin_vol | volatility | Chaikin Volatility(10,10), hand | 0.4987 | 0.4987 (0.5017) | 4.92e-01 | 0.822 | +0.0163 ± 0.0072 | -0.0058 ± 0.0042 | 0.429 (range_pct) | not significant after Bonferroni |
| az_ichimoku_span_b | structure | (close - Senkou Span B 52, unshifted)/close, hand | 0.5046 | 0.5046 (0.5017) | 5.08e-01 | 0.836 | -0.0348 ± 0.0058 | -0.0052 ± 0.0052 | 0.941 (dist_sma50) | not significant after Bonferroni |
| az_ret_skew_20 | statistical | Rolling skew(20) of returns, hand | 0.5044 | 0.5044 (0.5016) | 5.19e-01 | 0.836 | -0.0025 ± 0.0082 | -0.0057 ± 0.0024 | 0.507 (rsi_c) | not significant after Bonferroni |
| az_smma_20_dist | ma_distance | (close - SMMA(20))/close, pandas_ta | 0.5044 | 0.5044 (0.5016) | 5.19e-01 | 0.836 | -0.0346 ± 0.0045 | -0.0069 ± 0.0029 | 0.978 (dist_sma50) | duplicate of production feature |
| az_force_index | volume | Elder Force Index(13), pandas_ta efi | 0.5044 | 0.5044 (0.5017) | 5.30e-01 | 0.843 | -0.0368 ± 0.0040 | +0.0015 ± 0.0042 | 0.727 (dist_vwap) | not significant after Bonferroni |
| az_supertrend_dist | trend | (close - SuperTrend line)/close, pandas_ta | 0.5044 | 0.5044 (0.5017) | 5.42e-01 | 0.851 | -0.0391 ± 0.0074 | +0.0010 ± 0.0022 | 0.821 (dist_sma50) | not significant after Bonferroni |
| az_mfi_14 | volume | Money Flow Index(14), pandas_ta | 0.4992 | 0.4992 (0.5017) | 5.59e-01 | 0.868 | -0.0256 ± 0.0059 | -0.0041 ± 0.0030 | 0.780 (rsi_c) | not significant after Bonferroni |
| az_inertia | momentum | Inertia(20, RVI-based), pandas_ta | 0.4991 | 0.4991 (0.5016) | 5.70e-01 | 0.875 | -0.0191 ± 0.0052 | -0.0054 ± 0.0028 | 0.560 (netvol_rel) | not significant after Bonferroni |
| az_laguerre_rsi | momentum | Laguerre RSI (gamma 0.8), Ehlers, hand | 0.5039 | 0.5039 (0.5017) | 6.00e-01 | 0.900 | -0.0246 ± 0.0091 | -0.0058 ± 0.0045 | 0.776 (dist_sma50) | not significant after Bonferroni |
| az_mcg_20_dist | ma_distance | McGinley Dynamic(20) distance, hand | 0.5039 | 0.5039 (0.5017) | 6.00e-01 | 0.900 | -0.0338 ± 0.0037 | -0.0035 ± 0.0024 | 0.975 (dist_sma50) | duplicate of production feature |
| az_eom_14 | volume | Ease of Movement(14), pandas_ta | 0.5037 | 0.5037 (0.5017) | 6.36e-01 | 0.944 | -0.0211 ± 0.0044 | -0.0010 ± 0.0045 | 0.759 (dist_vwap) | not significant after Bonferroni |
| az_dema_20_dist | ma_distance | (close - DEMA(20))/close, pandas_ta | 0.5035 | 0.5035 (0.5016) | 6.61e-01 | 0.961 | -0.0335 ± 0.0081 | -0.0067 ± 0.0019 | 0.820 (macd_hist_rel) | not significant after Bonferroni |
| az_coppock | momentum | Coppock Curve(11,14,10), pandas_ta | 0.4997 | 0.4997 (0.5015) | 6.80e-01 | 0.961 | -0.0246 ± 0.0056 | -0.0025 ± 0.0027 | 0.881 (ewo) | not significant after Bonferroni |
| az_pfe_10 | momentum | Polarized Fractal Efficiency(10), hand | 0.4999 | 0.4999 (0.5017) | 6.80e-01 | 0.961 | -0.0342 ± 0.0054 | -0.0058 ± 0.0026 | 0.810 (rsi_c) | not significant after Bonferroni |
| az_fisher_9 | momentum | Fisher Transform(9), pandas_ta | 0.5035 | 0.5035 (0.5017) | 6.86e-01 | 0.961 | -0.0290 ± 0.0038 | +0.0066 ± 0.0035 | 0.753 (rsi_c) | not significant after Bonferroni |
| az_bop | statistical | Balance of Power = (close-open)/(high-low), pandas_ta | 0.5000 | 0.5000 (0.5017) | 6.93e-01 | 0.961 | -0.0295 ± 0.0051 | -0.0059 ± 0.0035 | 0.809 (price_change) | not significant after Bonferroni |
| az_stochrsi_k | momentum | StochRSI %K(14,14,3,3), pandas_ta | 0.5033 | 0.5033 (0.5016) | 6.99e-01 | 0.961 | -0.0317 ± 0.0058 | -0.0002 ± 0.0033 | 0.747 (macd_hist_rel) | not significant after Bonferroni |
| az_minus_di | trend | -DI(14), pandas_ta | 0.5034 | 0.5034 (0.5017) | 6.99e-01 | 0.961 | +0.0243 ± 0.0042 | -0.0050 ± 0.0028 | 0.880 (rsi_c) | not significant after Bonferroni |
| az_stc | momentum | Schaff Trend Cycle, pandas_ta | 0.5000 | 0.5000 (0.5015) | 7.18e-01 | 0.974 | -0.0086 ± 0.0076 | -0.0000 ± 0.0050 | 0.502 (rsi_c) | not significant after Bonferroni |
| az_rsi_sma_14 | momentum | SMA(14) of RSI(14) | 0.5031 | 0.5031 (0.5017) | 7.37e-01 | 0.974 | -0.0186 ± 0.0069 | -0.0044 ± 0.0028 | 0.856 (ewo) | not significant after Bonferroni |
| az_ao | momentum | Awesome Oscillator(5,34), pandas_ta | 0.5001 | 0.5001 (0.5016) | 7.44e-01 | 0.974 | -0.0249 ± 0.0079 | +0.0016 ± 0.0057 | 0.946 (ewo) | not significant after Bonferroni |
| az_ichimoku_span_a | structure | (close - Senkou Span A, unshifted)/close, hand | 0.5028 | 0.5028 (0.5015) | 7.63e-01 | 0.974 | -0.0425 ± 0.0035 | +0.0053 ± 0.0016 | 0.949 (dist_vwap) | not significant after Bonferroni |
| az_stoch_d | momentum | Stochastic %D(14,3,3), pandas_ta | 0.5029 | 0.5029 (0.5016) | 7.63e-01 | 0.974 | -0.0282 ± 0.0048 | -0.0054 ± 0.0029 | 0.749 (dist_vwap) | not significant after Bonferroni |
| az_trima_20_dist | ma_distance | (close - TRIMA(20))/close, pandas_ta | 0.5029 | 0.5029 (0.5016) | 7.63e-01 | 0.974 | -0.0349 ± 0.0020 | +0.0012 ± 0.0020 | 0.932 (dist_vwap) | not significant after Bonferroni |
| az_pat_engulfing | pattern | Engulfing pattern, signed +1 bull / -1 bear | 0.5029 | 0.5029 (0.5017) | 7.77e-01 | 0.974 | +0.0097 ± 0.0050 | -0.0047 ± 0.0043 | 0.384 (body_pct) | not significant after Bonferroni |
| az_adr_rel | levels | Mean daily range/close over prior 14 completed days, hand | 0.5027 | 0.5027 (0.5016) | 8.01e-01 | 0.974 | -0.0009 ± 0.0053 | -0.0009 ± 0.0062 | 0.746 (h4_atr_rel) | not significant after Bonferroni |
| az_ac_osc | momentum | Accelerator Oscillator = AO - SMA5(AO) (Bill Williams) | 0.5005 | 0.5005 (0.5015) | 8.10e-01 | 0.974 | -0.0181 ± 0.0094 | -0.0041 ± 0.0019 | 0.819 (macd_hist_rel) | not significant after Bonferroni |
| az_nvi_ret_20 | volume | Negative Volume Index 20-bar % change, pandas_ta | 0.5005 | 0.5005 (0.5016) | 8.10e-01 | 0.974 | -0.0073 ± 0.0091 | -0.0035 ± 0.0038 | 0.293 (dist_vwap) | not significant after Bonferroni |
| az_adx_14 | trend | ADX(14), pandas_ta | 0.5027 | 0.5027 (0.5017) | 8.16e-01 | 0.974 | -0.0029 ± 0.0047 | -0.0012 ± 0.0041 | 0.500 (volatility_21) | not significant after Bonferroni |
| az_pat_marubozu | pattern | Marubozu: body >= 90% of range, signed by direction | 0.5025 | 0.5025 (0.5017) | 8.43e-01 | 0.974 | -0.0194 ± 0.0145 | -0.0048 ± 0.0018 | 0.304 (body_pct) | not significant after Bonferroni |
| az_tsf_dev | statistical | (close - Time Series Forecast(20))/close, pandas_ta linreg+slope | 0.5007 | 0.5007 (0.5016) | 8.50e-01 | 0.974 | -0.0251 ± 0.0097 | -0.0051 ± 0.0027 | 0.663 (ha_dir) | not significant after Bonferroni |
| az_keltner_width | structure | Keltner channel width / close, pandas_ta kc | 0.5024 | 0.5024 (0.5016) | 8.57e-01 | 0.974 | +0.0075 ± 0.0082 | -0.0049 ± 0.0029 | 0.992 (atr_rel) | duplicate of production feature |
| az_ema_200_dist | ma_distance | (close - EMA(200))/close, pandas_ta | 0.5022 | 0.5022 (0.5014) | 8.70e-01 | 0.974 | -0.0245 ± 0.0084 | +0.0022 ± 0.0035 | 0.885 (h4_rsi_c) | not significant after Bonferroni |
| az_chop_14 | volatility | Choppiness Index(14), pandas_ta | 0.5024 | 0.5024 (0.5017) | 8.70e-01 | 0.974 | +0.0087 ± 0.0094 | -0.0012 ± 0.0045 | 0.486 (volatility_21) | not significant after Bonferroni |
| az_ulcer_14 | volatility | Ulcer Index(14), hand | 0.5009 | 0.5009 (0.5015) | 8.77e-01 | 0.974 | +0.0148 ± 0.0122 | +0.0024 ± 0.0057 | 0.669 (ewo) | not significant after Bonferroni |
| az_cmf_20 | volume | Chaikin Money Flow(20), pandas_ta | 0.5009 | 0.5009 (0.5016) | 8.77e-01 | 0.974 | -0.0337 ± 0.0050 | -0.0067 ± 0.0025 | 0.883 (adl_slope) | not significant after Bonferroni |
| az_parkinson_20 | volatility | Parkinson OHLC volatility estimator(20), hand | 0.5009 | 0.5009 (0.5016) | 8.77e-01 | 0.974 | +0.0032 ± 0.0088 | -0.0095 ± 0.0029 | 0.956 (atr_rel) | duplicate of production feature |
| az_bull_power | statistical | Elder Bull Power = (high - EMA13)/close, hand | 0.5010 | 0.5010 (0.5017) | 8.77e-01 | 0.974 | -0.0248 ± 0.0031 | -0.0018 ± 0.0038 | 0.861 (dist_vwap) | not significant after Bonferroni |
| az_vortex_diff | trend | Vortex VI+ - VI- (14), pandas_ta | 0.5023 | 0.5023 (0.5017) | 8.84e-01 | 0.974 | -0.0236 ± 0.0026 | -0.0037 ± 0.0057 | 0.852 (rsi_c) | not significant after Bonferroni |
| az_ppo | momentum | Percentage Price Oscillator(12,26,9), pandas_ta | 0.5009 | 0.5009 (0.5015) | 8.91e-01 | 0.974 | -0.0203 ± 0.0098 | +0.0056 ± 0.0021 | 0.867 (ewo) | not significant after Bonferroni |
| az_trix_15 | momentum | TRIX(15) %, pandas_ta | 0.5011 | 0.5011 (0.5017) | 8.91e-01 | 0.974 | -0.0186 ± 0.0084 | -0.0004 ± 0.0066 | 0.905 (h4_dist_sma) | not significant after Bonferroni |
| az_dpo_20 | momentum | Detrended Price Oscillator(20), causal form, hand | 0.5021 | 0.5021 (0.5016) | 8.97e-01 | 0.974 | -0.0031 ± 0.0154 | -0.0044 ± 0.0047 | 0.577 (dist_sma50) | not significant after Bonferroni |
| az_vhf_28 | trend | Vertical Horizontal Filter(28), hand | 0.5021 | 0.5021 (0.5016) | 8.97e-01 | 0.974 | +0.0031 ± 0.0115 | +0.0044 ± 0.0027 | 0.445 (volatility_21) | not significant after Bonferroni |
| az_roc_24 | momentum | ROC(24) %, pandas_ta | 0.5010 | 0.5010 (0.5015) | 9.04e-01 | 0.974 | -0.0328 ± 0.0100 | +0.0002 ± 0.0032 | 0.913 (ewo) | not significant after Bonferroni |
| az_zscore_20 | statistical | Price z-score(20) = (close-SMA)/SD, hand | 0.5011 | 0.5011 (0.5016) | 9.18e-01 | 0.974 | -0.0351 ± 0.0018 | -0.0002 ± 0.0036 | 0.876 (rsi_c) | not significant after Bonferroni |
| az_acf_5_50 | statistical | Autocorrelation of returns, lag 5, window 50, hand | 0.5012 | 0.5012 (0.5016) | 9.31e-01 | 0.974 | +0.0035 ± 0.0081 | +0.0012 ± 0.0061 | 0.037 (adl_slope) | not significant after Bonferroni |
| az_pivot_r1_dist | levels | (close - prior-day R1)/close, hand | 0.5020 | 0.5020 (0.5017) | 9.38e-01 | 0.974 | -0.0253 ± 0.0075 | +0.0017 ± 0.0040 | 0.733 (ewo) | not significant after Bonferroni |
| az_zlema_20_dist | ma_distance | (close - ZLMA(20))/close, pandas_ta | 0.5013 | 0.5013 (0.5016) | 9.45e-01 | 0.974 | -0.0361 ± 0.0092 | +0.0015 ± 0.0045 | 0.817 (ha_dir) | not significant after Bonferroni |
| az_bb_pctb | structure | Bollinger %B(20,2), pandas_ta (clipped outside the bands) | 0.5018 | 0.5018 (0.5016) | 9.52e-01 | 0.974 | -0.0353 ± 0.0018 | +0.0027 ± 0.0046 | 0.876 (rsi_c) | not significant after Bonferroni |
| az_cmo_14 | momentum | Chande Momentum Oscillator(14), pandas_ta | 0.5020 | 0.5020 (0.5017) | 9.52e-01 | 0.974 | -0.0359 ± 0.0024 | -0.0013 ± 0.0020 | 0.999 (rsi_c) | duplicate of production feature |
| az_rsi_14 | momentum | RSI(14), pandas_ta [positive control: production rsi_c] | 0.5020 | 0.5020 (0.5017) | 9.52e-01 | 0.974 | -0.0359 ± 0.0024 | -0.0013 ± 0.0020 | 0.999 (rsi_c) | duplicate of production feature |
| az_stoch_k | momentum | Stochastic %K(14,3,3), pandas_ta | 0.5015 | 0.5015 (0.5017) | 9.73e-01 | 0.986 | -0.0343 ± 0.0035 | -0.0061 ± 0.0067 | 0.781 (dist_vwap) | not significant after Bonferroni |
| az_mass_index | volatility | Mass Index(9,9,25), hand | 0.5017 | 0.5017 (0.5015) | 9.79e-01 | 0.986 | +0.0067 ± 0.0137 | -0.0013 ± 0.0026 | 0.408 (volatility_21) | not significant after Bonferroni |
| az_tema_20_dist | ma_distance | (close - TEMA(20))/close, pandas_ta | 0.5015 | 0.5015 (0.5016) | 9.86e-01 | 0.986 | -0.0299 ± 0.0100 | -0.0065 ± 0.0025 | 0.748 (ha_dir) | not significant after Bonferroni |

## Redundancy structure

Duplicates of production features (max |r| vs the 21 > 0.95) — 16:

- az_cmo_14 ≈ rsi_c (|r| = 0.999)
- az_rsi_14 ≈ rsi_c (|r| = 0.999)
- az_atr_pct ≈ atr_rel (|r| = 0.997)
- az_keltner_width ≈ atr_rel (|r| = 0.992)
- az_apo_rel ≈ ewo (|r| = 0.985)
- az_ema_50_dist ≈ dist_sma50 (|r| = 0.982)
- az_bb_bw ≈ volatility_21 (|r| = 0.981)
- az_vwma_dist ≈ dist_vwap (|r| = 0.980)
- az_smma_20_dist ≈ dist_sma50 (|r| = 0.978)
- az_mcg_20_dist ≈ dist_sma50 (|r| = 0.975)
- az_keltner_pos ≈ rsi_c (|r| = 0.970)
- az_smi ≈ rsi_c (|r| = 0.964)
- az_garman_klass_20 ≈ atr_rel (|r| = 0.957)
- az_parkinson_20 ≈ atr_rel (|r| = 0.956)
- az_typical_dev ≈ dist_vwap (|r| = 0.955)
- az_sma_20_dist ≈ dist_vwap (|r| = 0.955)

Clusters among the screened set at |r| > 0.90 (union-find): 5 clusters.
The largest is a 62-member trend/momentum mega-cluster — Alligator, ALMA,
Ichimoku lines, most MA-distance variants, ROC family, Stochastic, TSI,
TRIX, Fisher, KST and dozens more are, on hourly BTC, largely the same
signal wearing different names. A second cluster groups the volatility
estimators (ATR %, Garman-Klass, historical vol, Keltner width, Parkinson).
Cluster membership is stored in `results.json` (`analysis.clusters_abs_r_gt_0.90`).

## Exclusions (encyclopedia entries not screened, with reasons)

- **Not computable from 1h OHLCV at all**: order book / Level 2-3 depth,
  tick data and order flow; sentiment / NLP scores (news, social); on-chain
  metrics (hash rate, exchange flows, active addresses); macro series and
  external indices (VIX, rates, yields, gold/oil); market breadth
  (advance/decline, McClellan, TRIN, net new highs) — single-asset BTC has
  no cross-section; put/call ratio; relative currency strength.
- **Volume Profile** (listed in Kass's Drive Sheet / encyclopedia): requires
  the volume-at-price distribution inside each period; 1h OHLCV bars do not
  carry it. Excluded, not approximated.
- **Drawing tools, not features**: Fibonacci retracements/extensions/fans/
  time zones, Gann angles/fans/squares, Andrews' pitchfork — analyst-drawn
  geometry with no unique causal per-candle value.
- **Chart transforms, not per-candle features**: Renko, Kagi, Point & Figure
  (they re-bin price into non-time series; candidates for the sequence
  family later, not for this screen). ZigZag and Williams Fractals in their
  canonical forms use future bars to confirm pivots (non-causal); only the
  causal variants already covered (e.g. swing/price-channel forms) were
  screened.
- **Dropped at build time (documented in `az_indicators.py`)**: Relative
  Vigor Index (az_rvgi) — this pandas_ta fork's `rvi` signature does not
  match the canonical RVI, which requires open; excluded rather than
  screened under a wrong label (the Beyers Relative Volatility Index *is*
  screened, as az_rel_vol_index). QQE line (az_qqe_line) — pandas_ta
  0.4.71b0's QQE long line is NaN on ~50% of rows with interior gaps up to
  34 bars, an implementation defect of the library, not a property of QQE;
  excluded rather than graded on a defective series.
- **Calendar features** (time-of-day / day-of-week cyclical encodings) were
  screened (az_dow_sin/cos, az_hour_sin/cos are in the table) — no signal.

## Integrity notes

- Lab only: the screen read the cached diagnosis frame and wrote only
  under `data/az_screen/`. It never imported the live loop, never opened
  the paper ledger, and registered/promoted nothing.
- Production artifacts untouched: `data/paper.db` and
  `data/live_state.json` mtimes remain 2026-10-21 14:19 UTC (the parked
  snapshot); `models/ppo_champion.zip` mtime 2026-10-21 13:40 UTC — both
  predate the program. Parked state unchanged: $99,232.84, 0 BTC, 28 trades.
- Execution ran in checkpointed phases (`data/az_screen/progress.json`,
  `results.json` rewritten atomically per indicator); the raw/scaled
  indicator frames were computed once and cached as pickles, so the
  repeated runtime restarts during the run could not shift any value
  or window. All 132 screens + correlation + analysis phases completed.
- Environment: sklearn 1.9.1, scipy 1.18.1,
  pandas 3.0.6, numpy 2.2.6; route: pandas_ta 0.4.71b0 wrappers + hand implementations (TA-Lib not installed; see az_indicators docstring).
