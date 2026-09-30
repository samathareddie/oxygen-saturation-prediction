import numpy as np
import pandas as pd
import vitaldb
TRACK_NAMES = [
    "HR",
    "PLETH_SPO2",
    "RR_CO2",
    "ART_MBP"
]
COLUMN_NAMES = [
    "heart_rate",
    "spo2",
    "respiratory_rate",
    "mean_arterial_pressure"
]
def load_case_data(caseid):
    values = vitaldb.load_case(
        caseid,
        TRACK_NAMES,
        interval=2
    )

    df = pd.DataFrame(
        values,
        columns=COLUMN_NAMES
    )

    df["time_seconds"] = np.arange(len(df)) * 2

    return df
def create_minute_data(df):
    df = df.copy()

    df["minute"] = (
        df["time_seconds"] // 60
    ).astype(int)

    df_minute = (
        df
        .groupby("minute")
        .agg(
            heart_rate=("heart_rate", "median"),
            spo2=("spo2", "median"),
            respiratory_rate=("respiratory_rate", "median"),
            mean_arterial_pressure=(
                "mean_arterial_pressure",
                "median"
            ),
            spo2_valid_count=("spo2", "count"),
            samples_in_minute=("spo2", "size")
        )
        .reset_index()
    )

    return df_minute
def ensure_continuous_minutes(df):
    df = df.copy()

    full_minutes = range(
        int(df["minute"].min()),
        int(df["minute"].max()) + 1
    )

    df = (
        df
        .set_index("minute")
        .reindex(full_minutes)
        .rename_axis("minute")
        .reset_index()
    )

    return df
def apply_spo2_coverage_rule(df, minimum_coverage=0.50):
    df = df.copy()

    df["spo2_coverage"] = (
        df["spo2_valid_count"] /
        df["samples_in_minute"]
    )

    low_coverage = (
        df["spo2_coverage"] < minimum_coverage
    )

    df.loc[
        low_coverage,
        "spo2"
    ] = np.nan

    return df
def clean_vitals(df):
    df = df.copy()

    df.loc[
        df["heart_rate"] <= 0,
        "heart_rate"
    ] = np.nan

    df.loc[
        (df["spo2"] < 0) |
        (df["spo2"] > 100),
        "spo2"
    ] = np.nan

    df.loc[
        df["respiratory_rate"] < 0,
        "respiratory_rate"
    ] = np.nan

    df.loc[
        df["mean_arterial_pressure"] <= 0,
        "mean_arterial_pressure"
    ] = np.nan

    return df
def create_target(df):
    df = df.copy()

    df["future_spo2_5min"] = (
        df["spo2"].shift(-5)
    )

    return df
def create_features(df):
    df = df.copy()

    # SpO2 lag features
    df["spo2_lag_1"] = df["spo2"].shift(1)
    df["spo2_lag_3"] = df["spo2"].shift(3)
    df["spo2_lag_5"] = df["spo2"].shift(5)

    # SpO2 changes
    df["spo2_change_1min"] = (
        df["spo2"] -
        df["spo2_lag_1"]
    )

    df["spo2_change_3min"] = (
        df["spo2"] -
        df["spo2_lag_3"]
    )

    # SpO2 rolling features
    df["spo2_rolling_mean_5"] = (
        df["spo2"]
        .rolling(
            window=5,
            min_periods=3
        )
        .mean()
    )

    df["spo2_rolling_std_5"] = (
        df["spo2"]
        .rolling(
            window=5,
            min_periods=3
        )
        .std()
    )

    # Heart rate
    df["heart_rate_lag_1"] = (
        df["heart_rate"].shift(1)
    )

    df["heart_rate_rolling_mean_5"] = (
        df["heart_rate"]
        .rolling(
            window=5,
            min_periods=3
        )
        .mean()
    )

    # Respiratory rate
    df["rr_lag_1"] = (
        df["respiratory_rate"].shift(1)
    )

    df["rr_rolling_mean_5"] = (
        df["respiratory_rate"]
        .rolling(
            window=5,
            min_periods=3
        )
        .mean()
    )

    # Mean arterial pressure
    df["map_lag_1"] = (
        df["mean_arterial_pressure"].shift(1)
    )

    df["map_rolling_mean_5"] = (
        df["mean_arterial_pressure"]
        .rolling(
            window=5,
            min_periods=3
        )
        .mean()
    )

    return df
def process_case(caseid):
    df = load_case_data(caseid)

    df = create_minute_data(df)

    df = ensure_continuous_minutes(df)

    df = apply_spo2_coverage_rule(df)

    df = clean_vitals(df)

    df = create_target(df)

    df = create_features(df)

    df["caseid"] = caseid

    return df