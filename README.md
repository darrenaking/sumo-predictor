# Sumo Bout Predictor

A LightGBM-based machine learning system for predicting sumo wrestling bout outcomes.

## Overview

This project predicts:
1. **Winner** - Binary classification: P(wrestler A wins)
2. **Kimarite** - Multiclass classification over winning techniques
3. **Bout Duration** - Regression (if duration data available)

## Data Sources

- **Primary**: [sumo-api.com](https://sumo-api.com) - Free API with data from 1958 to present
- **Fallback**: Kaggle dataset for historical bout data (1983-2019)
- **Optional**: sumostats.com for bout duration data

## Project Structure

```
sumo-predictor/
├── notebooks/
│   ├── 01_data_collection.ipynb      # Fetch data from API/Kaggle
│   ├── 02_rating_systems.ipynb       # Compute ELO and Glicko-2
│   ├── 03_feature_engineering.ipynb  # Build feature set
│   ├── 04_model_training.ipynb       # Train LightGBM models
│   ├── 05_evaluation.ipynb           # Metrics, calibration, SHAP
│   └── 06_narrative_generation.ipynb # Qualitative bout previews
├── src/
│   ├── __init__.py
│   ├── data_collection.py            # API client and data fetching
│   ├── rating_systems.py             # ELO and Glicko-2 implementations
│   ├── feature_engineering.py        # Feature computation
│   ├── models.py                     # Model training utilities
│   ├── evaluation.py                 # Evaluation metrics
│   └── narrative.py                  # Narrative generation
└── README.md
```

## Execution Environment

This project runs on **Kaggle Notebooks** with the following paths:
- Input: `/kaggle/input/` (read-only datasets)
- Output: `/kaggle/working/` (persists as notebook output)

Each notebook's output becomes input to the next via Kaggle's dataset chaining.

## Features

The model uses 100+ features including:
- Physical attributes (height, weight, BMI, age)
- Career statistics (win rate, bout count, tournament count)
- Rating systems (ELO, Glicko-2 with uncertainty)
- Current tournament state (wins, losses, streaks)
- Recent form (rolling win rates)
- Style profiles (push/grapple/evasion percentages)
- Head-to-head history
- Pressure situations (kachikoshi/makekoshi, ozeki kadoban, etc.)
- Pairwise differentials (weight diff, rating diff, etc.)

**Critical**: All features use only information available BEFORE the bout to prevent data leakage.

## Rating Systems

Two rating systems are computed from historical bout data:

- **ELO**: Standard implementation with configurable K-factor
- **Glicko-2**: Includes rating deviation (uncertainty) and volatility

## Model Training

- LightGBM gradient boosted trees
- Time-based cross-validation (train on older, validate on newer)
- Metrics: AUC, accuracy, log loss, calibration (classification); RMSE, MAE (regression)
- SHAP for feature importance analysis

## Narrative Generation

The system generates natural language bout previews including:
- Predicted winner and confidence level
- Expected bout type (pushing match, belt battle, evasive bout)
- Most likely winning technique
- Closeness assessment
- Context notes (pressure situations, H2H history)

## Success Criteria

- Beat baseline of ~60% accuracy on winner prediction
- Well-calibrated probabilities
- Interpretable SHAP feature importances
- Sensible kimarite predictions
- Readable bout preview narratives
