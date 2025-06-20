📈 Context-Aware Return Forecasting via Rolling Adaptation
🧠 Overview
This project implements a rolling, adaptive forecasting model for financial time series, with a focus on learning both from individual stock behavior and from the dynamics of its peer group.

Instead of training on a fixed dataset, the model learns sequentially—predicting one day ahead, comparing its prediction to the true outcome, adjusting accordingly, and then forecasting the next step. This method mimics the way a model might learn in real-world deployment, adapting continuously as new data arrives.

🎯 Objectives
Predict future stock returns one day ahead using rolling, feedback-based training.

Incorporate group-level signals (such as the average return of a stock’s peer group) into each stock’s model.

Evaluate how much each asset benefits from contextual learning within its group.

Explore techniques such as Exponential Weighted Moving Averages (EWMA) and walk-forward validation in a dynamic setting.

🏗️ Methodology
Data Preprocessing

Compute daily returns from historical stock price data.

Organize stocks into meaningful groups (e.g., by industry or correlation clustering).

Calculate group average returns over time.

Sequential Forecasting Strategy

Initialize model parameters on day 1.

For each time step t, predict return at t+1.

Compare prediction to actual value, calculate error.

Update model based on that error (e.g., via gradient descent, updating weights, etc.).

Repeat the process for the entire time series.

Model Features

Past return values (lags, moving averages).

Group average returns.

Optional smoothing with EWMA for noise reduction.

Optionally explore models: Linear Regression, LSTMs, Kalman Filters, etc.