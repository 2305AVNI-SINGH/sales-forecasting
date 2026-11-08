"""Data loading, cleaning and feature engineering for the Rossmann sales forecast."""

import pandas as pd

FEATURES = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    "Year", "Month", "Day", "WeekOfYear",
    "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28",
]

LAGS = [7, 14, 28]


def load_data(data_dir):
    """Read train.csv and store.csv and merge them into one sorted DataFrame."""
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(f"{data_dir}/store.csv")
    df = train.merge(store, on="Store", how="left")
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def clean(df):
    """Fill missing store attributes. Blanks in Promo2 columns mean 'not in Promo2'."""
    df = df.copy()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                "Promo2SinceWeek", "Promo2SinceYear"]:
        df[col] = df[col].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")
    return df


def add_features(df):
    """Add calendar, lag and rolling features.

    Lags and rolling means only use sales that are at least 7 days old,
    so the model is a 7-day-ahead forecast and never sees future data.
    """
    df = df.copy()

    # Calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    # Lag features: sales N days earlier for the same store
    for lag in LAGS:
        tmp = df[["Store", "Date", "Sales"]].copy()
        tmp["Date"] = tmp["Date"] + pd.Timedelta(days=lag)
        tmp = tmp.rename(columns={"Sales": f"lag_{lag}"})
        df = df.merge(tmp, on=["Store", "Date"], how="left")

    # Rolling means, shifted by 7 so they only use past data
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    g = df.groupby("Store")["Sales"]
    df["roll_mean_7"] = g.transform(lambda s: s.shift(7).rolling(7).mean())
    df["roll_mean_28"] = g.transform(lambda s: s.shift(7).rolling(28).mean())

    # Encode text columns as integer codes
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    for col in ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]:
        df[col] = df[col].astype("category").cat.codes

    return df


def make_model_df(df):
    """Keep open days that have full lag history."""
    needed = ["lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28"]
    return df[df["Open"] == 1].dropna(subset=needed)
