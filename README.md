# Option Pricing with Black-Scholes and Machine Learning

This project investigates how well machine learning models can approximate real-world option prices compared to the classical Black-Scholes formula. The core idea is straightforward: Black-Scholes gives us a clean analytical baseline, but its assumptions (constant volatility, lognormal returns, no skew) break down in practice. By training data-driven models on the same inputs, we can see where the theory falls short and how much accuracy we can recover.

## What the notebook does

Everything lives in a single Jupyter notebook (`Main.ipynb`), designed to run end-to-end in Google Colab. It pulls live option chain data for a handful of major underlyings (AAPL, MSFT, TSLA, SPY) using `yfinance`, cleans it up, and builds out a rich feature set before training and evaluating several pricing models.

The pipeline flows roughly like this:

**Data acquisition and prep** — We download the nearest expiries for each ticker, compute mid-prices from bid/ask quotes, and filter out any rows with missing or nonsensical values. Historical price data is also fetched to calculate realized volatility for each underlying.

**Feature engineering** — On top of the raw option data, we compute the full set of Black-Scholes Greeks (delta, gamma, theta, vega, rho), the theoretical BS price itself, moneyness ratios, and a binary call/put flag. These features encode the structural information that drives option pricing.

**Exploratory analysis with PCA** — Before jumping into modeling, we run PCA on the standardized feature set to understand the variance structure. This helps confirm that the pricing problem is genuinely multivariate and that no single Greek dominates.

**Model training** — We train three models on log-transformed mid-prices:
- A plain linear regression as a baseline to demonstrate the limits of linear approaches on this problem.
- A random forest regressor, tuned via grid search with 3-fold cross-validation over tree depth, leaf size, and feature subsampling.
- An XGBoost regressor, similarly tuned over depth, learning rate, and row/column subsampling.

**Evaluation** — All models (plus the Black-Scholes analytical price) are compared on MAE, RMSE, R-squared, and SMAPE. We generate predicted-vs-actual scatter plots for each model to visualize accuracy, bias, and variance across the price range.

## Key findings

Black-Scholes does respectably well on aggregate metrics (R-squared around 0.996) but shows large errors on deep in-the-money and out-of-the-money contracts, and its SMAPE is high because it systematically misprices cheap options. Linear regression performs poorly, which is expected given the nonlinear relationships between Greeks, implied volatility, and prices. The tree-based models (random forest and XGBoost) both achieve strong test-set performance, with XGBoost edging ahead slightly in MAE and SMAPE.

## Running the notebook

The notebook is self-contained. Open it in Google Colab (or any Jupyter environment), run all cells top to bottom, and it will install dependencies, fetch fresh data, and produce all results. The only external dependency is an internet connection for the `yfinance` API calls.

Required packages (installed automatically in the first cell):
- `yfinance`
- `xgboost`
- `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy` (all included in Colab by default)

## Project structure

```
Main.ipynb    — Full analysis notebook (data, features, models, evaluation, discussion)
README.md     — This file
LICENSE       — MIT license
```