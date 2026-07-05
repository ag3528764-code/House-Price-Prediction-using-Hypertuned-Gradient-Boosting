# House Price Prediction using Hypertuned Gradient Boosting

A machine learning project built with Python, Scikit-Learn, and Streamlit to predict residential house prices. The project implements an end-to-end pipeline that includes data cleaning, automated feature engineering, target transformation, hyperparameter grid tuning, and a live web dashboard for model deployment.

---

## Features

Automated Feature Engineering: Computes runtime features like total square footage (`TotalSF`), total bathroom count (`TotalBath`), property age (`HouseAge`), and a remodeling indicator (`IsRemodeled`).
Advanced Preprocessing Pipeline:** Uses a `ColumnTransformer` to handle numerical and categorical data separately, applying median imputation, power transformations (`Yeo-Johnson` scaling), and One-Hot Encoding safely to avoid data leakage.
Target Optimization: Applies a log-transformation (`np.log1p`) to the skewed `SalePrice` target variable to normalize distribution and improves regression stability.
Hyperparameter Tuning:** Utilizes `GridSearchCV` with 5-fold cross-validation to search for optimal Gradient Boosting parameters (learning rate, depth, estimators, and subsample ratios).
Interactive UI:** Features a production-ready Streamlit interface where users can adjust property features via sliders and input boxes to get real-time price predictions.


# Repository Structure

* `fix_data.ipynb` — Handles data splitting by taking the raw master dataset and creating random, unbiased training and testing subsets to eliminate ordering bias.
* `train_and_plot.ipynb` — The primary training pipeline. It engineers features, runs the hyperparameter grid search, plots analytical charts, and serializes the final model weights.
* `predict_test.ipynb` — An independent evaluation script used to load the trained model and score it against an unseen holdout test dataset to generate true evaluation metrics (R², MAE, RMSE).
* `app.py` — The interactive user application that loads the trained pipeline binaries and hosts the web dashboard.
* `requirements.txt` — Lists the external third-party dependencies required to install and deploy the workspace.

---

# Installation & Setup

# 1. Prerequisites
Make sure you have Python 3.8 or higher installed on your computer.

# 2. Install Dependencies
Clone the repository and run the following command in your terminal to install the required external libraries:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
