"""Train and evaluate the sales forecasting models.

Usage:
    python src/train.py --data_dir data
"""

import argparse
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import FEATURES, add_features, clean, load_data, make_model_df

# Replace with the best config found during tuning in the notebook
BEST_PARAMS = {"n_estimators": 1000, "learning_rate": 0.03, "num_leaves": 63}

results = {}


def evaluate(name, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    results[name] = {"MAE": round(mae, 1), "RMSE": round(rmse, 1), "MAPE %": round(mape, 2)}
    print(name, results[name])


def main(data_dir, split_date, out_dir):
    df = add_features(clean(load_data(data_dir)))
    model_df = make_model_df(df)

    train_df = model_df[model_df["Date"] < split_date]
    valid_df = model_df[model_df["Date"] >= split_date]
    X_train, y_train = train_df[FEATURES], train_df["Sales"]
    X_valid, y_valid = valid_df[FEATURES], valid_df["Sales"]
    print("Train:", train_df.shape, "Validation:", valid_df.shape)

    # Naive baselines
    m7 = (valid_df["lag_7"] > 0).values
    evaluate("Last week (lag_7)", y_valid[m7], valid_df.loc[m7, "lag_7"])
    m14 = (valid_df["lag_14"] > 0).values
    evaluate("Two weeks ago (lag_14)", y_valid[m14], valid_df.loc[m14, "lag_14"])
    evaluate("Moving avg (28d)", y_valid, valid_df["roll_mean_28"])

    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    evaluate("Linear Regression", y_valid, lr.predict(X_valid))

    # LightGBM on log(sales)
    model = lgb.LGBMRegressor(
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, **BEST_PARAMS,
    )
    model.fit(X_train, np.log1p(y_train))
    evaluate("LightGBM (tuned, log target)", y_valid, np.expm1(model.predict(X_valid)))

    os.makedirs(out_dir, exist_ok=True)
    table = pd.DataFrame(results).T.sort_values("MAE")
    print(table)
    table.to_csv(os.path.join(out_dir, "results.csv"))
    joblib.dump(model, os.path.join(out_dir, "lgbm_final.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--split_date", default="2015-06-19")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    main(args.data_dir, args.split_date, args.out_dir)
