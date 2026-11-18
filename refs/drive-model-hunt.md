# Drive Model Hunt — every trained ML model in Kass's Google Drive

**Date:** 2026-10-21 · **Method:** global Drive searches by extension/name (`.pkl`, `.joblib`, `.pt`, `.pth`, `.h5`, `.keras`, `.onnx`, `.cbm`, `.sav`, `.zip`, `.ckpt`, `.bin`, `model`, `checkpoint`, `champion`, `best`, `weights`), fully paginated and deduped by file ID, plus complete recursive walks of "iPhone To Drive G" and targeted listings of every project folder under "Mac Projects". Small text companions (READMEs, configs, metrics CSVs, training logs, registry/metadata JSONs) were downloaded and read; no model binaries or datasets were downloaded. No credential-named files were opened.

**The gate contract (what a challenger must match to enter TradingAgent-Muse directly):**
- (a) Stacker slot: `predict_proba` on exactly **21 specific 1h-BTC features in [−1,1], fixed order**.
- (b) PPO slot: **SB3 zip, obs dim 25, 4 discrete actions** (hold/buy/sell/short).

## Headline verdict

**Zero of the 102 artifacts found can enter the gate directly.** Every PPO brain has the wrong observation size and/or wrong action count and/or wrong framework; every supervised model was trained on a different feature set than the 21-feature contract. Worse, the ones carrying performance claims mostly do not survive contact with their own paperwork:

- VisualML's saved "champions" all record **test accuracy 0.500** in their own metadata files.
- PROJECT 2's headline "91.7% ensemble accuracy" was earned on **120 rows** where 110 of 120 labels are the same class — predicting the majority class scores exactly 91.7%. Its own backtest report shows the ensemble returning **−0.28% vs buy-and-hold +14.35%** (30-day window, 4 trades).
- Trade Bot's four results files contradict each other: one says every model finished at **−100%**, another claims **+338%**. Nothing there is gradeable as-is.
- The lottery models (Lotto_Jup, Lottery_Bot) are the wrong domain — and per the 2026-10-21 Drive audit they were trained on **fake sample-generator draw history**, so their claims are void regardless.

The value in Drive is **recipes, not artifacts** — see the rebuild ranking at the bottom.

---

## Full inventory

Total artifacts in table: **102**

| # | File | Drive ID | Size | Modified | Project / folder |
|---|------|----------|------|----------|------------------|
| 1 | weights.bin (actor) | `1URUN4ZS612AqibmGLrd81sIbjB2iMjDs` | 171 KB | — | Full-Stack — apps/logs/ppo_model/actor |
| 2 | weights.bin (critic) | `1daapeWGSxwtpuUzfUDVbxStpdpo2_kun` | 170 KB | — | Full-Stack — apps/logs/ppo_model/critic |
| 3 | weights.bin (actor, dup) | `1brrh1LWt6Obiq5Iars4HI8kvTVZnNCiQ` | 171 KB | — | Full-Stack — logs/ppo_model/actor |
| 4 | weights.bin (critic, dup) | `1nUoi-fwgg1nGJvY61wmhnP8nsNmCZP2X` | 170 KB | — | Full-Stack — logs/ppo_model/critic |
| 5 | trained_models.joblib | `1_xd1n8hsem3M3quv6Q3OuucrudBq9hnL` | 28.1 MB | 2025-09-22 | Lottery_Bot — data |
| 6 | best_sequence_model.keras | `1p24JsPAMG8RKdIiA3_k8cytogji7-hh0` | 2.2 MB | 2025-09-29 | Lotto_Jup — models |
| 7 | pattern_model.pkl | `1ZLAsX1cCef3KDZWKPKvHI8PQgmKx4woy` | 708 KB | 2025-09-29 | Lotto_Jup — models |
| 8 | sum_model.pkl | `1cAfsEgCG_2p8_NL4zjXqy7Fhz28_Tiqr` | 812 KB | 2025-09-29 | Lotto_Jup — models |
| 9 | best_model.zip | `1NM0Rl1027yiMoA9oabWHGZydxlbyZ100` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models |
| 10 | ppo_trading_model.zip | `1Gf7hPhRORIDRP_vhbG2U-UetJ2xm5HwS` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models |
| 11 | best_model.zip | `1bOgqMhLazigyxO00v-4yK7o2bJZFehGB` | 188 KB | 2025-07-23 | PPO_Trading_Agent — models/best_ppo_model |
| 12 | ppo_trading_checkpoint_100000_steps.zip | `1_sejNFiY02cFR8b0TtEngsq9c1qWx37t` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 13 | ppo_trading_checkpoint_10000_steps.zip | `1b4NWPT3G_JgHFYWIFNbsTdACtnkZMXzK` | 746 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 14 | ppo_trading_checkpoint_105000_steps.zip | `1JSKzo_H9qRg9ZGTCSCIR4_LEmQ8o095H` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 15 | ppo_trading_checkpoint_110000_steps.zip | `1UDXwMv3F9KncFT9aQzTr7wiD4oeFnqXK` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 16 | ppo_trading_checkpoint_115000_steps.zip | `1nbr5kv1wQiUNgdMPy0KpKW122qSf18ja` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 17 | ppo_trading_checkpoint_120000_steps.zip | `1FCoFZlewmtw9DwaZjqt6_CuDWs8-LaN3` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 18 | ppo_trading_checkpoint_125000_steps.zip | `1vsE8mFkw_iWhXYS9F54AfnB3QIOab6Z8` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 19 | ppo_trading_checkpoint_130000_steps.zip | `1CLh3MupvaDOCitS_gvKZIOA4m28FHdq6` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 20 | ppo_trading_checkpoint_135000_steps.zip | `1xPhZjgjVxhvgKc4FB0xs5noO8kCYQhpv` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 21 | ppo_trading_checkpoint_140000_steps.zip | `1GDQIUE3DmFT8dCoMdISzbU_ZPWUosTNP` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 22 | ppo_trading_checkpoint_145000_steps.zip | `13al11Nhxt0wqTif1Uy9NJg5cKEwQL-jO` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 23 | ppo_trading_checkpoint_150000_steps.zip | `1ptOsizipHCbIH94jmKwYsxR-Rj1BT-ll` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 24 | ppo_trading_checkpoint_15000_steps.zip | `1K-yIkQfn6HFfuanPFjKqAMt05otpcCKb` | 747 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 25 | ppo_trading_checkpoint_175000_steps.zip | `1iwjWrp65d2QObOEANddr1g5w1pjJZMVb` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 26 | ppo_trading_checkpoint_200000_steps.zip | `1NUwuZv4TR47LgWPuo0wShUoAvxZy_kGc` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 27 | ppo_trading_checkpoint_20000_steps.zip | `1aCoK1FUZDMZdn_tkqkAXBeBnBZORYYBU` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 28 | ppo_trading_checkpoint_225000_steps.zip | `1s-8uDF59nqET5Dx_Igsd6n0VOPOE0OuY` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 29 | ppo_trading_checkpoint_250000_steps.zip | `1HJFJkzW8s6y-S3gbeC_pOpzGM23fzUFI` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 30 | ppo_trading_checkpoint_25000_steps.zip | `1mU7ZC-4FwOuKUzmLTewiweTDTR9qJ6T_` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 31 | ppo_trading_checkpoint_275000_steps.zip | `1Q2KcMS6FcPUX_AhSluK57BBDFCApeuj2` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 32 | ppo_trading_checkpoint_300000_steps.zip | `1trzJyX2f_u-obYDM0-KHWO8hxkZSCanl` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 33 | ppo_trading_checkpoint_30000_steps.zip | `15j-vPEJdZK3Uhn962ysELkuwiOd2vfC5` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 34 | ppo_trading_checkpoint_325000_steps.zip | `1VKVWeaduOa4P8esM5Hvwh13k3fwJ8ooy` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 35 | ppo_trading_checkpoint_350000_steps.zip | `1-ddJ8PRDtMaLx0XnQQJZJFIzCbZngZhk` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 36 | ppo_trading_checkpoint_35000_steps.zip | `1AwS1SJjnNKQ-KMd6kP1UStDfCCqEUW46` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 37 | ppo_trading_checkpoint_375000_steps.zip | `1a4SZwoTsEbAES06TKm_dTdy38_3bis2J` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 38 | ppo_trading_checkpoint_400000_steps.zip | `1qyT14i5kJ0CnKnhS65KQT1MZlhxw9oYE` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 39 | ppo_trading_checkpoint_40000_steps.zip | `1Gs1UymVwmEZSxVhUoMzJBFV6QdlFqZjr` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 40 | ppo_trading_checkpoint_425000_steps.zip | `1EM7-kISQF8pJddh1zQFX0Vh4MIOT2vbX` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 41 | ppo_trading_checkpoint_450000_steps.zip | `19x-AbpiJ5S-QiJqH_CB0v4fmOlVZunpB` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 42 | ppo_trading_checkpoint_45000_steps.zip | `1S9rTNXzdvMAqvNn4vuJ_jkzXK8EqCZ4Q` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 43 | ppo_trading_checkpoint_475000_steps.zip | `1i90Wnf7Rnq2MYSQMpTiV00-CXg-IpTkV` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 44 | ppo_trading_checkpoint_500000_steps.zip | `1YS2VTXuLGLDVgW4e0NI9N_39nGTzhTmv` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 45 | ppo_trading_checkpoint_50000_steps.zip | `1fCx8lWklR4HMogZk2Zdhl1mcAMujEFNT` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 46 | ppo_trading_checkpoint_5000_steps.zip | `14OsYFR6a0buYvzoI-scbFWluy5YRkY-Q` | 745 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 47 | ppo_trading_checkpoint_55000_steps.zip | `1Sb2qBcQRUtsGvFWu0pZ163csvrYWw2gV` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 48 | ppo_trading_checkpoint_60000_steps.zip | `1HTiP88mPVSZXBOCCtO8ZMHDS0nI3Nq4_` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 49 | ppo_trading_checkpoint_65000_steps.zip | `13-A8P5PGnmd3ijPy5rH06321wgQYnjiT` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 50 | ppo_trading_checkpoint_70000_steps.zip | `1f5kmsIoFFeN2I_QF6AXyE4ns43OTjmLM` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 51 | ppo_trading_checkpoint_75000_steps.zip | `12AhgVyWSpm5MCkgMrPGRj2xa2XAfrLLM` | 3.5 MB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 52 | ppo_trading_checkpoint_80000_steps.zip | `16qjnMc_ZyBXoebg1FMnsdf9wjXPCjuVe` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 53 | ppo_trading_checkpoint_85000_steps.zip | `1wdEST_aTWI0Dm-nMn7TkVqeACMXDg_7j` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 54 | ppo_trading_checkpoint_90000_steps.zip | `1y5HURio0J5OYnkd-F1Oh47jghMlPjOc3` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 55 | ppo_trading_checkpoint_95000_steps.zip | `14uVDNL5-bD6t5F7cdbgsLpGrMjJ-NtZ_` | 748 KB | 2025-07-23 | PPO_Trading_Agent — models/checkpoints |
| 56 | best_model.pt | `1brGHBRaFNA23sWrTNDPzmF7axQMWY38M` | 3.0 MB | 2025-05-22 | PROJECT 2 — models |
| 57 | checkpoint_epoch_0.pt | `1Hdr5Nhz59jQBO8nfAEY2QqSYlsqIxr_4` | 3.0 MB | 2025-05-22 | PROJECT 2 — models |
| 58 | checkpoint_epoch_20.pt | `1OvDlT9CEQUnofkFA4Nye_ZCJ1JZK8SF8` | 3.0 MB | 2025-05-22 | PROJECT 2 — models |
| 59 | checkpoint_epoch_40.pt | `1bJefiNiniEM-OzCJW72R_hXdUqJiG2nL` | 3.0 MB | 2025-05-22 | PROJECT 2 — models |
| 60 | checkpoint_epoch_60.pt | `1wcK2WsWwVh13kU0g6EeVQ46zcXiLOxGm` | 3.0 MB | 2025-05-22 | PROJECT 2 — models |
| 61 | checkpoint_epoch_80.pt | `1mELvGV9Y_i8QpQIewMR4V541H8rsJWXA` | 3.0 MB | 2025-05-22 | PROJECT 2 — models |
| 62 | lightgbm.joblib | `1LOPDCrCD7_npigA5kZwenoI3EpGZRm7u` | 11 KB | 2025-05-22 | PROJECT 2 — models |
| 63 | random_forest.joblib | `1kJ_ll-w457a0nipr4pc7BqVeJ-IU00Wx` | 1.2 MB | 2025-05-22 | PROJECT 2 — models |
| 64 | scaler.joblib | `1lz8DSSuU06-OSAHB3au8OqDW_29ScYx9` | 3 KB | 2025-05-22 | PROJECT 2 — models |
| 65 | xgboost.joblib | `1VPMHfxYsWfwkMfgk_JtpedzJ_nwXYnNR` | 2.2 MB | 2025-05-22 | PROJECT 2 — models |
| 66 | trading_model_20241203_220110.keras | `1TzkC7ozzmV-cj7IoSxih_fftotCeiEj3` | 205 KB | 2024-12-04 | Trade Bot — models/backups |
| 67 | best_model_01.h5 | `1CiZwc8MzhKtGi9m3_vXQi49AAU4R8zI2` | 13.5 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 68 | best_model_04.h5 | `1HSx93YD6LPpva7NtpCy6D8GzfvomAleO` | 13.5 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 69 | best_model_05.h5 | `1eq2EiE4CoVXjScanvdw_1Xle-QBYZ1mp` | 13.5 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 70 | best_model_11.h5 | `1fci4H8-ThbnMsLy4i7rI2JIuHW_6XckD` | 13.5 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 71 | best_model_12.h5 | `18Pfm5AnafBlhnX-CZnywS300o5WhrZNY` | 13.5 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 72 | best_model_13.h5 | `1JdrkkzZcl96kkwbcBuO4cN3QIOzMqKTv` | 13.5 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 73 | best_model_14.h5 | `1PzdkrX_Di3tXOpM7rpMPRoSCEsqVSBxW` | 13.5 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 74 | feature_scaler.pkl | `1mN_VPWLxsjKyUIhyXxKrS4y4S5NvHvBJ` | 2 KB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 75 | final_model.h5 | `1iKWV0mUP5nDJon71XlcR_zOgCJSGygyM` | 14.0 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 76 | fold_1.weights.h5 | `13sxgkf1LDlr2CTUIv285Q5FqAZkotKkB` | 23.2 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 77 | fold_1_final.weights.h5 | `1XB0OcX7uF8vSgveoLLv2dKipddNRjWLP` | 23.2 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 78 | fold_2.weights.h5 | `1ENGtKPbqzLuuiP13WCd4VvacchsZHzw3` | 23.2 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 79 | fold_2_final.weights.h5 | `1NDcmv0vkwME152BnvybAZJ8gTM5Gf9Sd` | 23.2 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 80 | fold_3.weights.h5 | `1gR3M0TFdd-7EKl4nzrpJfxJGKaQzXN2v` | 23.2 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 81 | fold_3_final.weights.h5 | `1TDuf1lHSXVpqlDJ6m8YuAaZ7fhuO3cS6` | 23.2 MB | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 82 | price_scaler.pkl | `15degQlz5kcRnEd70D8CfDce6DJZ7NYCu` | 572 B | 2025-01-28 | Trade Bot — models/heikin_ashi_3 |
| 83 | heikin_ashi_model.keras | `1UeQvanl4ehRO-8ES3-oAl3cCg7vxporl` | 1.8 MB | 2025-01-16 | Trade Bot — models/saved_models |
| 84 | heikin_ashi_model.weights.h5 | `18pHWQPch7HkGiVGQ2065oOCv1qeBYk1H` | 6.6 MB | 2025-01-16 | Trade Bot — models/saved_models |
| 85 | model_improved.keras | `1N3YR-gIKWxzwYJCMaGL3CSSldW8QSYwE` | 12.5 MB | 2025-01-06 | Trade Bot — models/saved_models |
| 86 | model_optimized_20241203_215107.keras | `19uKysDKJPrN9p53a1n4PShsySxYjDFnz` | 205 KB | 2024-12-04 | Trade Bot — models/saved_models |
| 87 | model_optimized_20241203_220106.keras | `1IQGxv_WecHglYM14qhUWYmbQDXP8pM8Q` | 205 KB | 2024-12-04 | Trade Bot — models/saved_models |
| 88 | trading_model.h5 | `1dkhPW9ghbXegL0bdTs1RtlEf0Swjm87M` | 205 KB | 2024-12-04 | Trade Bot — models/saved_models |
| 89 | trading_model.keras | `1OKq9ZrOXqbmJ2yPn6UAtCO5tyTe-LPPE` | 1.6 MB | 2024-12-21 | Trade Bot — models/saved_models |
| 90 | trading_model_20241203_215111.h5 | `1qpUE0edLmsPhBcOSKSohPMy_d0sbtXU8` | 209 KB | 2024-12-04 | Trade Bot — models/saved_models |
| 91 | trading_model_20241203_215250.h5 | `1rNZVIOqEwIrLUfU0jFMVIZsDSS-2pgyG` | 205 KB | 2024-12-04 | Trade Bot — models/saved_models |
| 92 | trading_model_20241203_215351.h5 | `1mZjllOfJn3Wv41tBN6E7vYB9Rrh28BcS` | 205 KB | 2024-12-04 | Trade Bot — models/saved_models |
| 93 | trading_model_20241203_215501.h5 | `1yN9ZNvIRql-wQplrGzXJ9eT0T9b8BRyz` | 205 KB | 2024-12-04 | Trade Bot — models/saved_models |
| 94 | trading_model_20241203_215710.h5 | `1pJ0RJXW0o7skMiriTjI3F7TKAJAdQ672` | 205 KB | 2024-12-04 | Trade Bot — models/saved_models |
| 95 | trading_model_20241203_215802.h5 | `1_b8WiiQn1u0gUX96kXTJwjWzv0SO4sEe` | 205 KB | 2024-12-04 | Trade Bot — models/saved_models |
| 96 | trading_model_new.keras | `1K2Ic-ONMj1ArBDeE4QQM477vMsYexPuV` | 1.6 MB | 2025-01-06 | Trade Bot — models/saved_models |
| 97 | best_model.keras | `16_hN6aoI0v6Lt0dqpiGfa4ucdWpnwtAw` | 5.7 MB | 2025-01-27 | Trade Bot — models/saved_models/heikin_ashi_2 |
| 98 | scaler.pkl | `1GBBQ82h-1Qwl37WpTk_YvbwJpUk8RX20` | 1 KB | 2025-01-27 | Trade Bot — models/saved_models/heikin_ashi_2 |
| 99 | best_model.keras | `1edJ9_SrQyhe5PkGeeZ-62UjvUoLjvw3p` | 51.2 MB | 2025-12-10 | VisualML — models |
| 100 | final_model.keras | `1890ycm9djiQtC7wkvRImkRC6Pahn28Lv` | 51.2 MB | 2025-12-10 | VisualML — models |
| 101 | AI-Test-Model-1_final.keras | `1VgwddhW-cdcN6bwqalzUoES8g6sBpN8E` | 286.6 MB | 2026-01-07 | VisualML — users/<uuid>/models |
| 102 | data_PAVS_1d_90.pkl | `19qZD63Nx8n-OaTp0AH3mrxba_occmhgU` | 6 KB | 2026-03-18 | algos-platform — cache |

*Excluded from the table: 6 Google Takeout export zips at Drive root (`takeout-20251211T…`, 0.4 MB–237 MB, Dec 2025) — bulk archives, not individually trained model artifacts; contents not opened. Also excluded: Python library files inside synced venvs/site-packages (`.pth` path files, `saved_model_cli` scripts, onnx converter scripts, `.pyc` caches) and the TF.js library's own files.*

---

## Per-project notes & verdicts

### 1. PPO_Trading_Agent — 47 artifacts · **Recipe worth rebuilding (PPO slot)** · not directly testable

- **What:** SB3 PPO for BTC, final model (`ppo_trading_model.zip`), EvalCallback best (`models/best_model.zip`), a separate smaller `best_ppo_model/best_model.zip` (188 KB), and 44 checkpoints in two size families (3.54 MB ×20 + 748 KB ×24 — two different network/obs configurations across runs).
- **Recipe (from README + config):** Alpaca **hourly BTC/USD, 17,522 bars**; **~109-dim obs** = 100+ TA-Lib indicators (SMA/EMA/TEMA, RSI, MACD, ROC, CMO, TRIX, BB, ATR, OBV, MFI, A/D, Chaikin, Stochastic, Williams %R, CCI, ADX/Aroon/SAR…) normalized 0–1 + 3 position features; **2 actions only** (Hold / Change-Position toggle); net [256,256,128,64]; 500k steps; 0.05% transaction cost; 10+-component reward (portfolio change + Sharpe + momentum + anti-whipsaw…); chronological 70/15/15 splits claimed.
- **Claimed performance:** training mean reward 54.6 → 101+ over 500k steps. **No out-of-sample P&L or test-set return is stated anywhere** — the README's performance section is qualitative.
- **Why not directly testable:** obs ~109 ≠ 25 and actions 2 ≠ 4; SB3 would refuse the env, and force-loading would be meaningless. Also note its reward-rich recipe rhymes with what TAM's own clean v5.1 grading just failed — a rebuild should simplify, not copy the 10-component reward wholesale.
- **Rebuild value:** highest of the PPO candidates — it is TAM's direct ancestor (hourly BTC, honest split discipline claimed, real cost modeling). The transferable ideas: the broad TA-Lib feature library as a *feature-mining source* for a future feature-set revision, and EvalCallback-best checkpointing.

### 2. PROJECT 2 — 10 artifacts · **Recipe worth rebuilding (stacker slot, with heavy caveats)** · not directly testable

- **What:** ensemble joblibs — `random_forest.joblib`, `xgboost.joblib`, `lightgbm.joblib`, `scaler.joblib` — plus a **custom PyTorch (non-SB3) PPO**: `best_model.pt` + `checkpoint_epoch_{0,20,40,60,80}.pt`.
- **Recipe:** Alpaca crypto; pipeline raw → processed (indicators) → normalized (StandardScaler) → labeled ([−1,0,1] → [0,1,2]); ensemble = RF + XGB + LGBM (the direct ancestor of TAM's RF+XGB+LGBM→LR stacker); PPO: state = rolling **window 10 × num_features**, **3 actions** (Buy/Sell/Hold), actor-critic, position-based reward with cost penalty.
- **Claimed performance — and the catch:** final saved ensemble (2025-05-22 run) reports val accuracy RF 0.9167 / XGB 0.875 / LGBM 0.9167 / ensemble 0.9167 — **on 120 total rows (96 train / 24 val) with class distribution [2, 110, 8]**. A constant majority-class predictor scores 110/120 = 91.7%. Earlier runs on other data scored 0.42–0.58 val accuracy. Its own backtest report: ensemble **−0.28%** vs buy-and-hold **+14.35%** (30d), +0.22% (50d), +485% (100d); the strategy made **4 trades** and its "best threshold" was the one where it never trades.
- **Why not directly testable:** ensemble expects 84 features (per its training log's feature shape) standardized by its own scaler, not TAM's 21 [−1,1] features; the PPO is a raw PyTorch checkpoint with state dim 10×F, not an SB3 zip.
- **Rebuild value:** the *architecture* (tree ensemble over labeled direction) is already TAM's stacker design — nothing new to import. The useful lesson is negative and valuable: tiny-sample + majority-class traps are exactly what TAM's clean gate now guards against.

### 3. Trade Bot — 33 artifacts · **Recipe partially worth rebuilding** · not directly testable, claims not credible

- **What:** three generations of Keras sequence models in `models/`: `saved_models/` (Dec 2024 `trading_model*.h5/.keras` ×7, `heikin_ashi_model.keras/.weights.h5`, `model_improved.keras`, `trading_model_new.keras`, `model_optimized_*` ×2, plus one backup copy), `saved_models/heikin_ashi_2/` (`best_model.keras` + `scaler.pkl`), `heikin_ashi_3/` (3-fold CV: `fold_{1,2,3}.weights.h5` + `_final` variants, `best_model_{01,04,05,11,12,13,14}.h5`, `final_model.h5`, `feature_scaler.pkl`, `price_scaler.pkl`). RL model weights referenced by its results files were **not found** in Drive (only `.pyc` code and TensorBoard logs remain).
- **Recipe (from `config/model_config.py` + `trading_config.py`):** **BTC-USD 1h from Yahoo Finance**, 1 year train / last-30-days test, fees 0.1% + slippage 0.1%. Flagship HA model: **sequence length 90 × 30 features**, CNN (filters 64/128/128) + LSTM(512) + 8-head attention, multi-task (classification weight 2.0 + regression 1.0), **predicts 10 steps ahead**. A second LSTM config uses seq 120 × **102 features**. Custom RL config: state_dim 16, action_dim 3.
- **Claimed performance — internally contradictory:** `results/model_comparison/model_metrics.csv` shows most models at **total return −100%** (blown up); `model_comparison_with_rl.csv` claims +93% to **+338%** on 10–220 trades with near-zero Sharpes; `results/trading_model/metrics.csv` claims +32.9%, PF 1.63, 46 trades; `rl_backtest_metrics.csv` claims +300%, Sharpe 1.09, 1,202 trades. These cannot all describe the same system; at least some accounting is broken (e.g. −100% rows with "best trade +1000%").
- **Why not directly testable:** Keras sequence models consume 90-step windows of their own 30 (or 102) features + their own scalers — a different input contract from both gate slots, and no SB3 anything.
- **Rebuild value:** the **Heikin-Ashi-centric sequence idea at 1h** is the one genuinely different recipe in Drive, and it aligns with Kass's own indicator instincts (HA is one of his five). If a challenger family beyond the current stacker/PPO is ever built, a clean-room HA-sequence classifier graded by the honest gate is the most original candidate. Its Drive-era numbers should be treated as unproven.

### 4. Full-Stack — 4 artifacts · Not directly testable · low rebuild value

- **What:** TF.js PPO **actor + critic** (`weights.bin` + `model.json`), duplicated in `apps/logs/ppo_model/` and `logs/ppo_model/`.
- **Recipe (read from the actor's `model.json`):** input **dim 30**, Dense 256 → BatchNorm → Dropout 0.2 → Dense 128 → BatchNorm → softmax **3 actions**. Part of the Next.js/FastAPI monorepo whose README is partly template-like (per the verified portfolio pass, claim architecture only).
- **Why not:** TF.js runtime, obs 30 ≠ 25, actions 3 ≠ 4, unknown feature definitions and unknown training provenance. Nothing claimed about performance anywhere.

### 5. VisualML — 3 artifacts · **Wrong input modality** · claims refuted by its own metadata

- **What:** `models/best_model.keras` + `final_model.keras` (51 MB each) and user-trained `AI-Test-Model-1_final.keras` (287 MB, ResNet50). Registry also references `Test-Model-2*.keras` (~160 MB) whose files are not present in Drive (metadata only).
- **Recipe (from `training_metadata.json`):** CNN over **chart images** — EfficientNetB0, 256×256 images, 2-class; the user model is ResNet50 at 128×128.
- **Claimed vs recorded:** registry calls AI-Test-Model-1 the **active model**; its own metadata records **best_val_accuracy 0.5, test_accuracy 0.5** (test_loss 916). The flagship's history: train accuracy 0.999 by epoch 5 while **val_accuracy stayed 0.500 every epoch** — textbook memorization, zero generalization. Test-Model-2: same 0.5/0.5.
- **Why not:** image input cannot plug into a feature-vector gate at all; and there is no performance to import even if it could.

### 6. Vision Trading Bot — **0 artifacts** · nothing to test

- `vision_trade_bot/models/saved_models/finvision/runs/` holds 3 run folders (Feb 2026 stamps) and `pretrained/weights/finvision/` exists — **all empty in Drive**. `pattern_classifier/` is empty too. Only code + `model_config.yaml` survive (CNN+ViT over 224×224 chart images, 2 classes: bullish/shorting). Whatever was trained never synced to Drive.

### 7. Lotto_Jup (3 artifacts) + Lottery_Bot (1 artifact) — **Wrong domain; claims void**

- `sum_model.pkl`, `pattern_model.pkl`, `best_sequence_model.keras` (Lotto_Jup) and `trained_models.joblib` (28 MB, Lottery_Bot) predict lottery numbers, not markets. Per the 2026-10-21 Drive audit, both projects' draw history was **fake sample-generator output**, so their models learned noise by construction. Nothing importable, for any purpose.

### 8. algos-platform — 1 artifact · not a model

- `cache/data_PAVS_1d_90.pkl` (6 KB) is a cached data artifact (daily, 90-period) in a cache folder — no trained model files exist anywhere in algos-platform despite its ~449 JSON data files.

### 9. "iPhone To Drive G" / Project_options_bot — **0 artifacts**

- Full recursive walk (55 files): Project_options_bot contains only a README and a metrics table — its code subfolders are empty in Drive. The options decision engine was never accompanied by a trained model here.

### 10. Projects with no trained artifacts anywhere in Drive

CRYPTO_JUMPER, "?????" (full-stack ML Trading Bot), stock-city, trading_ui, heatmap-3D, PROJECT ACE, CrewAi Project, Auto-Post, Trading-League-Game copy — zero model files under any searched extension/name. (Outside Drive, for completeness: GitHub repo `HKassem42/Project-Bot` holds `best_trading_model.pkl`, the original 5m-BTC stacking classifier — not part of this Drive hunt, and its 5m feature set matches neither gate contract either.)

---

## Rebuild ranking — which recipes deserve a clean challenger build

1. **PPO_Trading_Agent's hourly-BTC PPO recipe, simplified** — closest ancestor to TAM; transferable parts are its TA-Lib feature library (as candidate features for a v6 set) and eval-best checkpointing. Do **not** import its 10-component reward or 2-action toggle design; TAM's 4-action env and dense MtM reward are strictly better, and TAM's clean v5.1 result warns that this family shows no edge yet on honest grading.
2. **Trade Bot's Heikin-Ashi sequence classifier (90×30, CNN+LSTM+attention, 10-step horizon)** — the only architecturally distinct idea in Drive and aligned with Kass's own HA indicator. A clean-room rebuild (leak-free labels, TAM's gate, real costs) is the most original challenger family available. Its historical numbers (+33% / +300% / −100%, depending which file you read) are not evidence.
3. **PROJECT 2's tree ensemble** — already TAM's stacker architecture; nothing to rebuild that doesn't already exist. Its real contribution is the cautionary tale baked into its logs (120 rows, 92% majority class).

**Bottom line:** there is no amazing pre-trained brain hiding in Drive. Every performance claim attached to these artifacts is either absent (PPO_Trading_Agent), refuted by the project's own files (VisualML 0.500, PROJECT 2 −0.28% vs +14.35% B&H), self-contradictory (Trade Bot), or void (lottery, fake data). The park stays parked until a challenger passes the clean gate — and if Kass wants a new challenger family, the HA-sequence recipe (#2) is the one worth building cleanly.
