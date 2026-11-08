# Sales Forecasting with LightGBM (Rossmann Store Sales)

A 7-day-ahead daily sales forecast for 1,115 Rossmann stores, built with calendar, lag and rolling-window features and a LightGBM model. The project compares the model against simple baselines and a Linear Regression using a strict time-based validation.

## Problem

Given a store's sales history up to 7 days ago, plus known information about the target day (promotion, holidays, weekday), predict that store's daily sales.

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) (Kaggle): about 1 million daily records for 1,115 stores, from Jan 2013 to Jul 2015.

- `train.csv`: daily sales, promotions, holidays
- `store.csv`: store type, assortment, competition and Promo2 details

The data is not included in this repository. Download both files from Kaggle and put them in a `data/` folder.

## Approach

1. **Cleaning:** missing `Promo2` fields mean the store is not in the Promo2 programme, so they are filled with 0 / "None". Missing `CompetitionDistance` is filled with the median. Closed days (always zero sales) are removed before modelling.
2. **Features:**
   - Calendar: year, month, day, week of year, day of week
   - Lags: sales 7, 14 and 28 days earlier for the same store
   - Rolling means: 7-day and 28-day averages, shifted by 7 days so no future data is used
   - Promo, holiday, store and competition attributes
   - `Customers` is deliberately excluded because it is not known in advance (data leakage)
3. **Validation:** time-based split. Training uses data before 2015-06-19, and the last ~6 weeks are held out. No random splits.
4. **Models:** naive baselines, Linear Regression, LightGBM, and LightGBM trained on `log(1 + sales)` with light hyperparameter tuning.

## Results

Validation window: 2015-06-19 to 2015-07-31 (open days only).

| Model | MAE | RMSE | MAPE % |
|---|---|---|---|
| **LightGBM (tuned, log target)** | **603.0** | **894.8** | **9.03** |
| LightGBM (log target) | 610.7 | 890.5 | 9.19 |
| LightGBM | 620.1 | 900.4 | 9.45 |
| Linear Regression | 824.1 | 1139.3 | 12.82 |
| Two weeks ago (lag_14) | 813.0 | 1183.2 | 11.77 |
| Moving average (28 days) | 1489.4 | 2106.2 | 20.44 |
| Last week (lag_7) | 2178.0 | 2809.1 | 31.82 |

The tuned LightGBM has about 27% lower MAE than Linear Regression. Time-series cross-validation (3 folds): **[add mean CV MAE here]**.

![Actual vs predicted total sales](images/actual_vs_pred_total.png)
![Store 1: actual vs predicted](images/store1_actual_vs_pred.png)

## Key findings

- **Weekly seasonality:** most stores close on Sundays, so total daily sales drop sharply every week.
- **Yearly seasonality:** average sales peak in December, with a smaller rise in spring.
- **Promotions:** average daily sales are about 38% higher on promo days.
- **Promo cycle:** a store's promo flag matches the one from 14 days ago 85% of the time but the one from 7 days ago only 35% of the time. This is why `lag_14` is the most important feature and "same as last week" is a poor baseline.
- **Feature importance (gain):** `lag_14`, `lag_28`, `roll_mean_28` and `Promo` contribute most. Store-level attributes add little because the lag features already capture each store's sales level.

![Feature importance (gain)](images/feature_importance_gain.png)

## Limitations

- The model under-predicts some sales peaks and over-predicts some dips (see the Store 1 plot), and it over-predicted total sales in early July 2015.
- The validation set is a single 6-week window, and hyperparameters were tuned on it, so the reported score is slightly optimistic.
- Forecast horizon is fixed at 7 days. Longer horizons need different features.

## How to run

```bash
pip install -r requirements.txt
python src/train.py --data_dir data
```

This trains the models, prints the comparison table, and saves `outputs/results.csv` and `outputs/lgbm_final.pkl`.

## Project structure

```
sales-forecasting/
├── data/            (not tracked, download from Kaggle)
├── images/          (plots used in this README)
├── notebooks/       (EDA and experiments)
├── src/
│   ├── features.py  (loading, cleaning, feature engineering)
│   └── train.py     (training and evaluation)
├── requirements.txt
└── README.md
```

## Tools

Python, Pandas, NumPy, scikit-learn, LightGBM, Matplotlib, Google Colab
# sales-forecasting
## Internship Details
- **Intern ID:** CIS69
- **Full Name:** Avni Singh
- **No. of Weeks:** 1 Week
- **Project Name:** Sales Forecasting with LightGBM (Rossmann Store Sales)
- **Project Scope:** 7-day-ahead daily sales forecasting for 1,115 stores, covering data cleaning, EDA and visualization, feature engineering, model comparison and a forecast dashboard.
