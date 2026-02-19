Introduction This project is built around the ongoing Kaggle competition
"Hull Tactical -- Market Prediction". The challenge asks participants to
predict the next-day return of the S&P 500 index and to convert those
predictions into a trading strategy that seeks positive excess returns
while controlling portfolio risk. The main training file, train.csv,
contains several decades of daily U.S. equity-market data. Each row
corresponds to a trading day identified by a 'date_id', with hundreds of
anonymized features grouped into families such as market-dynamics (M),
macro- economic (E), interest-rate (I), valuation (P), volatility (V),
sentiment (S), momentum (MOM), and binary dummy variables (D). The
prediction target is 'forward_returns', defined as the return from
buying the S&P 500 and selling it one day later;\
The file also includes the contemporaneous federal funds rate as a proxy
for the risk- free rate.

The competition uses a custom risk-adjusted performance metric,
implemented in baseline notebooks as an adjusted-Sharpe-style score
computed from a simulated daily trading strategy, and classified on
Kaggle as a "custom metric."

The competition runs in two phases: a model-training phase using
historical data, and a forecasting phase using future, real-time data.
After submissions close, a forecasting phase begins in which a second
set of roughly 180 trading days of fresh S&P 500 data is collected after
the deadline; final rankings will be based on models' performance on
this out-of-sample period.

However, the competition's 180-day live test window will finish after
the CS680 course deadline, so the official private leaderboard will not
be available when this project is due. To obtain an immediately
observable, out-of-sample benchmark for the final project, we therefore
reserve the last 180 trading days of the provided train.csv as our
project test set, and use all earlier dates exclusively for training.

Background Section Tree-based model trial: This part of the project
focuses on tree-based models for predicting daily S&P 500 excess
returns. Instead of predicting raw forward returns, we predict the sign
(negative/positive) of the market forward excess return, which is equal
to raw forward return minus risk free rate, and later convert the
predicted probability into a 0--2 long exposure trading strategy.
Firstly, we conduct exploratory data analysis to gain deeper insights.
The target 'market_forward_excess_returns' has mean close to zero and
standard deviation around 1% per day, with clear non-normal tails.
Linear correlations between the target and the 90+ input features are
all very small (\|ρ\| ≤ 0.07), suggesting that any usable signal must be
non-linear and very weak. A Hurst R/S analysis further supports this
view: the cumulative excess return has a Hurst exponent of ≈1.01,
indicating persistent long-term trend, while the raw excess returns have
H≈0.54, close to uncorrelated noise. Based on the results above, a
moderately regularized tree model with time-series feature engineering
might be a good option to capture non-linear signal without overfitting
noise.\
We use LightGBM and XGBoost, because both are gradient-boosted trees
with strong regularization controls (e.g. both L1/L2 penalties, elastic
net). We also try two ensemble method: stacking and blending.\
We choose log loss as our principal model selection metrics, since it is
exactly the training objective of our tree models (binary logistic
loss). We also treat AUC as a secondary diagnostic to evaluate models'
predicting accuracy. The adjusted Sharpe ratio is the ultimate
evaluation metric, since it is the competition's official metric.

Experiment Section All feature engineering is implemented inside a
scikit-learn Pipeline to ensure that the same transformations are
applied consistently within each time-series cross-validation fold and
in the final training. Firstly, we drop columns with more than 50%
missing values, which improves stability and avoids heavy imputation on
very sparse signals. Secondly, we create lagging & rolling features
based on our target 'market_forward_excess_returns'. We use lagging
features with lag 1\~5 to capture short-term momentum or mean-reversion
over one trading week. Rolling means and standard deviations over
windows of 5, 21, and 63 days, roughly corresponding to weekly, monthly,
and quarterly horizons are used to summarize local trend and volatility
regimes at different time scales. Thirdly, we apply wavelet
decomposition to our target and decompose it into 3 components:
low-frequency sub-band for long- term trend, mid-frequency sub-band for
medium-term cycles, low-frequency sub-band for short-term noise. From
these components we derive multi-scale features such as variances and
energy shares, which quantify how much of the recent activity is trend-
driven versus noise-driven; rolling slope and means, which represent
medium-term direction and local reference levels, and so on. These are
all relatively standard multi- scale features in the financial
time-series. In addition, we made several design efforts to avoid data
leakage when constructing wavelet-based features. We implement causal
wavelet filtering instead of standard DWT/MODWT with symmetric padding
and centered convolutions. All features are computed using
backward-looking rolling windows. After all lagging, rolling, and
wavelet features are created, we fill any remaining missing values with
a constant 0.

As discussed in background section, we consider two tree-based
probabilistic classifiers, LightGBM and XGBoost, both trained on the
same preprocessed feature set. Hyperparameters are tuned via
RandomizedSearchCV with TimeSeriesSplit in 5 folds and scoring metric
negative log loss. To prevent overfitting, our search spaces focus on
relatively shallow trees, moderate learning rates, and non-zero L1/L2
regularization.

After fixing the optimal hyperparameters for LGB and XGB, we compute
out-of-fold (OOF) predictions on the train set using TimeSeriesSplit
again. For each fold, the model is trained on past data and predicts the
next validation block; concatenating these predictions yields OOF
probability vectors, which are 𝑝𝐿𝐺𝐵 and 𝑝𝑋𝐺𝐵. Then two ensemble methods
are tried. For blending, we perform a grid search over weights 𝑤 ∈{0.0,
0.1, ⋯, 1.0} for 𝑝𝐵𝑙𝑒𝑛𝑑= 𝑤∙𝑝𝐿𝐺𝐵+ (1 −𝑤) ∙𝑝𝑋𝐺𝐵 and compute OOF log loss &
AUC for each weight. The optimal weight is selected according to the
lowest OOF log loss. For stacking, we stack the OOF probabilities
\[𝑝𝐿𝐺𝐵, 𝑝𝑋𝐺𝐵\] into a 2-dimension meta-feature input and fit a logistic
regression meta-model. A second level of TimeSeriesSplit on the OOF
region is used to tune the regularization strength 𝐶 by OOF log loss.

As mentioned above, our final goal is to convert model's predicted
probability into a 0--2 exposure trading strategy and maximize the
adjusted Sharpe ratio. Therefore, a position function is defined as
follows to map predicted probabilities into daily leverage: 𝑝𝑜𝑠(𝑝) =
𝑐𝑙𝑖𝑝(𝑓𝑙𝑜𝑜𝑟+ 𝑠𝑙𝑜𝑝𝑒∙max(𝑝−𝑐𝑒𝑛𝑡𝑒𝑟, 0) , 𝑓𝑙𝑜𝑜𝑟, 𝑚𝑎𝑥_𝑝𝑜𝑠). The function has
four tunable parameters: 𝑓𝑙𝑜𝑜𝑟, minimum exposure; 𝑐𝑒𝑛𝑡𝑒𝑟, probability
level above which we start increasing exposure; 𝑠𝑙𝑜𝑝𝑒, how aggressively
exposure grows with confidence; 𝑚𝑎𝑥_𝑝𝑜𝑠, maximum allowed leverage. For
each of the four model variants (LGB, XGB, blend, stack), we tune
parameters on the OOF region by grid search, maximizing the OOF adjusted
Sharpe, and then generate four position functions.

Results Section Using the tuned models and position functions, we refit
each pipeline on the entire training period and generate predicted
probabilities, and strategy returns on the 180- day test set. The
resulting log loss, AUC, and adjusted Sharpe for LGB, XGB, blend, and
stack are summarized as follows:

Model Test log loss AUC Adjusted Sharpe\
LGB + position 0.6895 0.5591 --0.24\
XGB + position 0.6896 0.5440 0.30\
Blend (best w = 0) + position 0.6896 0.5440 0.30 since best w = 0,
equivalent pure XGB Stack (best C = 0.1) + position 0.6909 0.5584 0.22

All models achieve test log loss only slightly better than the
random-guess baseline (\~0.693) and AUC in the narrow range 0.54--0.56,
indicating that daily excess returns are extremely hard to predict and
any probabilistic edge over randomness is very weak. After mapping
probabilities into positions, the adjusted Sharpe on the test set is
negative for LGB, modestly positive (\~0.30) for XGB and its blend
variant (which effectively reduces to pure XGB), and smaller (\~0.22)
for stacking. Hence XGBoost + position is the best of the four, but
still delivers only limited improvement in risk- adjusted performance.
Then we visualize the best strategy, XGBoost + position, on the 180-day
test period by two plots. The first plot is 'Cumulative total return of
the market vs. the best strategy'. The market's cumulative return
increases from 1.00 to about 1.03, while the strategy ends around 1.035.
More importantly, during drawdown episodes where the market falls
sharply, the strategy's equity curve is noticeably smoother and avoids
the deepest losses.

Another plot is 'Monthly Return Bar Chart'. The strategy's returns have
much lower standard deviation than the market and a higher fraction of
small positive days, which indicates that the strategy "wins small but
often" while avoiding large drawdowns.

Conclusion Section Our XGB + position strategy delivers robust,
conservative but positive risk-adjusted performance and serving as a
reasonably rigorous quant modeling. In our current setup, model
hyperparameters, ensemble parameters and position function parameters
are tuned using time-series CV and OOF predictions on the training
period. Repeatedly selecting the best on the same OOF data can introduce
mild optimistic bias. It might be improved by use a rolling-window
nested time-series CV, where in each window we re-run hyperparameter
search and ensemble selection and evaluate only on the forward "future"
slice. Sliding this window over time would yield a more robust estimate
of performance, at the cost of much higher computational and
implementation complexity. Besides, all binary classifiers we built very
slightly outperform random guessing on test data. It aligns perfectly
with the Hurst analysis in our EDA, where raw excess returns behave
essentially like uncorrelated noise with very weak predictability. While
the position function we designed is likely to contribute more to
improving the adjusted Sharpe ratio. We therefore do not expect that
swapping in other model families would dramatically improve performance.
In fact, these findings suggest that a model-free approach, a purely
rule-based trading strategy with no learnable parameters, might perform
better than our current model- based pipeline. However, this goes beyond
the scope of CS680 and even machine learning, so we leave that as an
open direction rather than exploring it here.
