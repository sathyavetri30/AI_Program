Advanced Time Series Forecasting with Transformer and Attention
1. Introduction

This project implements an advanced Transformer-based deep learning model for multivariate time series forecasting. The objective is to predict Global Active Power consumption using historical sensor data indexed by time. The work focuses on capturing long-range temporal dependencies and feature importance through self-attention mechanisms.

2. Dataset Description

The dataset consists of a large-scale multivariate time series indexed by a single time column. It includes power consumption and electrical measurements sampled at regular intervals. Missing values are handled using time-based interpolation.

3. Feature Engineering

To satisfy the minimum feature requirement and enhance predictive power, additional derived features were created:

Hour of day

Day of week

24-step rolling mean

24-step rolling standard deviation
This results in 11 total input features.

4. Preprocessing

All features are normalized using StandardScaler. Sliding window sequences are created with a fixed look-back window of 48 timesteps.

5. Model Architecture

A Transformer Encoder with multi-head self-attention is used. The architecture includes:

Input embedding layer

Multi-head self-attention

Stacked Transformer encoders

Fully connected output layer

6. Training Strategy

Rolling window time-series cross-validation is employed instead of a single train-test split. This ensures robustness and prevents temporal leakage.

7. Hyperparameter Optimization

A structured hyperparameter search is performed over embedding dimension and number of attention heads. The configuration yielding the lowest validation RMSE is selected.

8. Baseline Model

A strong multivariate XGBoost baseline is implemented using lagged values of all features, ensuring a fair comparison.

9. Evaluation Metrics

Models are evaluated using RMSE, MASE, and sMAPE.

10. Attention Weight Analysis

Learned attention weights indicate that recent timesteps receive higher importance, with voltage and global intensity consistently contributing strongly to predictions. This improves interpretability compared to traditional models.

11. Conclusion

The Transformer-based model outperforms the baseline while providing interpretability through attention mechanisms. The approach is suitable for real-world forecasting tasks involving complex temporal dependencies.# AI_Program
