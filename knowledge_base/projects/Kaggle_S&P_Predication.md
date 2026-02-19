---
title: Kaggle S&P Prediction (Hull Tactical — Market Prediction)
category: project
tags: [xgboost, lightgbm, eda, feature-engineering, time-series, leakage-control, wavelet, hurst]
priority: high
last_updated: 2026-02
---

# Kaggle S&P Prediction (Hull Tactical — Market Prediction)

## 1. Introduction
This project is built around the ongoing Kaggle competition **“Hull Tactical — Market Prediction”**. The challenge asks participants to predict the **next-day return** of the S&P 500 index and convert those predictions into a **trading strategy** that seeks positive excess returns while controlling portfolio risk.

The main training file `train.csv` contains several decades of daily U.S. equity-market data. Each row corresponds to a trading day identified by a `date_id`, with **hundreds of anonymized features** grouped into families such as:

- Market-dynamics (M)  
- Macro-economic (E)  
- Interest-rate (I)  
- Valuation (P)  
- Volatility (V)  
- Sentiment (S)  
- Momentum (MOM)  
- Dummy variables (D)

The prediction target is `forward_returns`, defined as the return from buying the S&P 500 and selling it one day later. The file also includes the contemporaneous **federal funds rate** as a proxy for the **risk-free rate**.

## 2. Evaluation Metric and Competition Setup
The competition uses a **custom risk-adjusted performance metric**, implemented in baseline notebooks as an **adjusted-Sharpe-style score** computed from a simulated daily trading strategy (Kaggle “custom metric”).

The competition runs in two phases:
1. **Model-training phase** using historical data  
2. **Forecasting phase** using future, real-time data

After submissions close, a forecasting phase begins in which a second set of roughly **180 trading days** of fresh S&P 500 data is collected after the deadline. Final rankings will be based on models’ performance on this out-of-sample period.

**Project constraint:** the live 180-day test window will finish after the CS680 course deadline, so the official private leaderboard will not be available when this project is due.  
**Our solution:** we reserve the last **180 trading days** of the provided `train.csv` as the project test set, and use all earlier dates exclusively for training.

---

## 3. Background: Tree-Based Modeling Strategy

### 3.1 Prediction Target and Trading Objective
This work focuses on tree-based models for predicting daily S&P 500 **excess returns**. Instead of predicting raw forward returns, we predict the **sign (negative/positive)** of the market forward excess return:

- `market_forward_excess_returns = forward_returns - risk_free_rate`

We then convert the predicted probability into a **0–2 long exposure** trading strategy.

### 3.2 EDA Findings
Key observations:
- The target `market_forward_excess_returns` has **mean close to zero** and **standard deviation around ~1% per day**, with **non-normal tails**.
- Linear correlations between the target and the 90+ input features are all very small (**|ρ| ≤ 0.07**), suggesting that any usable signal is **weak and likely non-linear**.

We also ran a Hurst R/S analysis:
- The **cumulative** excess return shows a Hurst exponent of **≈ 1.01**, indicating persistent long-term trend.
- The **raw** excess returns have **H ≈ 0.54**, close to uncorrelated noise.

**Implication:** a moderately regularized tree model with time-series feature engineering might capture weak non-linear signal while reducing overfitting.

### 3.3 Model Families and Metrics
We use **LightGBM** and **XGBoost** because both are gradient-boosted trees with strong regularization controls (e.g., L1/L2 penalties). We also try two ensemble approaches: **blending** and **stacking**.

We select models using:
- **Primary metric:** log loss (aligned with the training objective: binary logistic loss)
- **Secondary diagnostic:** AUC
- **Ultimate objective:** adjusted Sharpe ratio (competition metric)

---

## 4. Methods: Feature Engineering, Leakage Control, and Training

### 4.1 Pipeline Design
All feature engineering is implemented inside a **scikit-learn Pipeline** to ensure transformations are applied consistently within each time-series cross-validation fold and in the final training.

### 4.2 Preprocessing and Time-Series Features
Steps:
1. **Drop sparse columns:** remove features with >50% missing values to improve stability and avoid heavy imputation on very sparse signals.
2. **Lag and rolling features** based on `market_forward_excess_returns`:
   - Lags (1–5 days) to capture short-term momentum or mean-reversion (roughly one trading week)
   - Rolling means and standard deviations over 5, 21, and 63 days (weekly, monthly, quarterly) to summarize trend and volatility regimes
3. **Fill remaining missing values** with a constant 0 after feature creation.

### 4.3 Wavelet Decomposition Features (with Leakage-Aware Design)
We apply wavelet decomposition to the target and decompose it into 3 components:
- **Low-frequency** sub-band: long-term trend  
- **Mid-frequency** sub-band: medium-term cycles  
- **High-frequency** sub-band: short-term noise  

From these components we derive multi-scale features such as:
- Variances and energy shares (trend-driven vs noise-driven activity)
- Rolling slopes and rolling means (medium-term direction and local reference levels)

**Leakage control:** we avoid standard DWT/MODWT setups that rely on symmetric padding and centered convolutions. We implement **causal wavelet filtering**, and compute features using **backward-looking rolling windows** only.

---

## 5. Model Training and Ensembling

### 5.1 Base Models and Hyperparameter Tuning
We consider two probabilistic classifiers: **LightGBM** and **XGBoost**, trained on the same feature set.

Hyperparameters are tuned via:
- `RandomizedSearchCV`
- `TimeSeriesSplit` (5 folds)
- scoring metric: negative log loss

Search spaces prioritize:
- relatively shallow trees
- moderate learning rates
- non-zero L1/L2 regularization

### 5.2 OOF Predictions for Blending and Stacking
After selecting hyperparameters, we compute **out-of-fold (OOF) predictions** using `TimeSeriesSplit` again. Each fold trains on past data and predicts the next validation block; concatenated predictions yield OOF probability vectors:

- \( p_{LGB} \)
- \( p_{XGB} \)

Ensemble methods:
- **Blending:** grid search weights \( w \in \{0.0, 0.1, \dots, 1.0\} \) for  
  \( p_{blend} = w \cdot p_{LGB} + (1-w)\cdot p_{XGB} \)  
  choose weight by lowest OOF log loss.
- **Stacking:** use \([p_{LGB}, p_{XGB}]\) as 2D meta-features and train a logistic regression meta-model.  
  Tune regularization strength \( C \) by OOF log loss using a second-level time-series split.

---

## 6. Trading Strategy: Mapping Probability to Position
Our final goal is to convert predicted probabilities into a **0–2 exposure** trading strategy and maximize adjusted Sharpe.

We define a position function that maps predicted probability \(p\) into daily leverage:

- exposure increases only when \(p\) exceeds a threshold (“center”)
- exposure is clipped between a minimum floor and a maximum leverage

The function has four tunable parameters:
- **floor:** minimum exposure  
- **center:** probability threshold above which exposure increases  
- **slope:** aggressiveness of exposure growth  
- **max_pos:** maximum allowed leverage  

For each of the four model variants (**LGB**, **XGB**, **blend**, **stack**), we tune the position parameters on the OOF region by grid search, maximizing OOF adjusted Sharpe.

---

## 7. Results (180-Day Held-Out Test Set)

### 7.1 Summary Metrics
Using the tuned models and position functions, we refit each pipeline on the full training period and evaluate on the 180-day test set:

| Model | Test log loss | AUC | Adjusted Sharpe |
|------|---------------:|----:|----------------:|
| LGB + position | 0.6895 | 0.5591 | -0.24 |
| XGB + position | 0.6896 | 0.5440 | 0.30 |
| Blend (best w = 0) + position | 0.6896 | 0.5440 | 0.30 |
| Stack (best C = 0.1) + position | 0.6909 | 0.5584 | 0.22 |

Notes:
- All models achieve test log loss only slightly better than the random-guess baseline (~0.693).
- AUC lies in a narrow range (0.54–0.56), indicating that daily excess returns are extremely hard to predict and any edge is weak.
- After mapping probabilities into positions, **XGBoost + position** is best among the four, with modestly positive adjusted Sharpe.

### 7.2 Strategy Behavior (Qualitative)
For the best strategy (**XGB + position**), we visualize performance on the 180-day test period using:
1. **Cumulative return comparison (market vs strategy):**  
   The market increases from ~1.00 to ~1.03, while the strategy ends around ~1.035. During market drawdowns, the strategy equity curve appears smoother and avoids the deepest losses.
2. **Monthly return bar chart:**  
   The strategy has lower return volatility than the market and a higher fraction of small positive days—suggesting a “wins small but often” profile while avoiding large drawdowns.

---

## 8. Conclusion and Limitations
Our XGB + position strategy delivers **conservative but positive** risk-adjusted performance, serving as a reasonably rigorous quant modeling pipeline. However, there are limitations:

- **Optimistic bias risk:** hyperparameters, ensemble parameters, and position parameters are all tuned using time-series CV and OOF predictions on the training period. Repeatedly selecting best variants on the same OOF region can introduce mild optimistic bias.
- **Potential improvement:** a rolling-window **nested** time-series CV could provide a more robust estimate of performance, but would substantially increase compute and implementation complexity.
- **Fundamental predictability constraint:** raw daily excess returns behave close to noise (consistent with Hurst ≈ 0.54). Therefore, we do not expect dramatically better performance simply by switching model families.
- **Future direction:** a purely rule-based strategy (no learnable parameters) might outperform our model-based pipeline, but exploring this is beyond the scope of CS680.
