Household Power Consumption Forecasting
Transformer with Explicit Attention & XGBoost Baseline
📌 Project Overview

This project implements an end-to-end multivariate time series forecasting system for predicting Global Active Power using a custom Transformer neural network with explicit multi-head attention. In addition to deep learning, a strong XGBoost lag-based model is implemented as a baseline to ensure fair performance comparison.

A key focus of this project is explainability—the Transformer is designed to expose attention weights, allowing analysis of which historical timesteps influence predictions the most.

🎯 Objectives

Forecast household energy consumption accurately

Capture long-term temporal dependencies using attention

Provide model interpretability through attention weights

Compare deep learning performance against a classical ML baseline

Follow best practices in time series validation

📊 Dataset Description

Dataset: Household Power Consumption

Frequency: Minute-level readings

Target Variable: global_active_power

Preprocessing Steps:

Column normalization and cleaning

Datetime parsing and indexing

Numeric coercion and time-based interpolation

Removal of missing values after feature engineering

🧠 Feature Engineering

The model uses 11 multivariate features, including:

Raw Power Signals

Global active power

Global reactive power

Voltage

Global intensity

Sub-metering (1, 2, 3)

Time-Based Features

Hour of day

Day of week

Statistical Features

24-hour rolling mean

24-hour rolling standard deviation

These features allow the model to learn daily patterns, weekly seasonality, and short-term volatility.

🔄 Data Preparation

Features are standardized using StandardScaler

Sliding windows of 48 timesteps are created for supervised learning

Rolling window cross-validation ensures:

No data leakage

Realistic forecasting conditions

Temporal generalization

🤖 Transformer Model Architecture

The Transformer is implemented from scratch using PyTorch, avoiding internal abstractions to ensure attention weights are accessible.

Architecture Components

Linear embedding layer

Stacked Multi-Head Self-Attention layers

Residual connections

Layer normalization

Feed-forward networks

Final regression head

Why Custom Transformer?

PyTorch’s built-in TransformerEncoder does not reliably expose attention weights. This custom design ensures:

Stable attention extraction

Full transparency

Research-grade explainability

🔍 Attention-Based Explainability

After training:

Attention weights are extracted directly from the model

Importance scores are computed across:

Batch dimension

Attention heads

Query timesteps

The output identifies which historical timesteps most influence the prediction

This provides valuable insights into:

Short-term vs long-term dependency usage

Model decision behavior

Temporal relevance patterns

📈 Model Training

Optimizer: Adam

Loss function: Mean Squared Error (MSE)

Hyperparameter search over:

Embedding dimensions

Number of attention heads

Best model selected using validation RMSE

🧪 Baseline Model (XGBoost)

To benchmark performance, a strong multivariate XGBoost model is trained using:

1-step lag features

24-step lag features

All engineered variables

This ensures the Transformer’s performance is evaluated against a competitive traditional ML approach, not a weak baseline.

📦 Outputs

Best Transformer model saved as:

final_transformer_attention_model.pth


Printed validation RMSE scores

Top influential timesteps from attention analysis

Trained XGBoost baseline model

🛠️ Requirements

Python 3.9+

PyTorch

NumPy

Pandas

Scikit-learn

XGBoost

🚀 Applications

Energy demand forecasting

Smart grid optimization

Load prediction for power utilities

Explainable AI for time series

Research and academic projects

🔮 Future Improvements

Feature-level attention

Temporal Fusion Transformer (TFT)

Probabilistic forecasting

Multi-step forecasting horizon

Attention heatmap visualization

Hyperparameter tuning with Optuna

🧾 Conclusion

This project demonstrates how attention-based deep learning models can outperform traditional approaches while remaining interpretable and robust. The combination of a custom Transformer architecture, strong feature engineering, and rigorous validation makes this solution suitable for real-world forecasting tasks and research use cases.
 providing interpretability through attention mechanisms. The approach is suitable for real-world forecasting tasks involving complex temporal dependencies.
