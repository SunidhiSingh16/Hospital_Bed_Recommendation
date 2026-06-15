import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta


def generate_synthetic_data(
    start_date=None,
    days=365,
    base_admissions=50,
    weekday_impact=10,
    seasonal_impact=15,
    special_events_count=5,
    special_events_magnitude=30
):
    if start_date is None:
        current_year = datetime.now().year
        start_date = f'{current_year}-01-01'

    dates = pd.date_range(start=start_date, periods=days, freq='D')
    base = np.random.normal(loc=base_admissions, scale=base_admissions * 0.2, size=days)

    weekday_effect = np.where(
        pd.Series(dates).dt.dayofweek < 5,
        weekday_impact,
        -weekday_impact / 2
    )

    month_nums = pd.Series(dates).dt.month
    monthly_effect = np.where(
        (month_nums >= 11) | (month_nums <= 2),
        seasonal_impact,
        np.where(
            (month_nums >= 6) & (month_nums <= 8),
            -seasonal_impact / 2,
            0
        )
    )

    special_events = np.zeros(days)
    if special_events_count > 0:
        event_days = np.random.choice(days, size=special_events_count, replace=False)
        special_events[event_days] = np.random.normal(
            loc=special_events_magnitude,
            scale=special_events_magnitude * 0.2,
            size=special_events_count
        )

    daily_admissions = base + weekday_effect + monthly_effect + special_events
    daily_admissions = np.maximum(daily_admissions, 0)

    return pd.DataFrame({
        'date': dates,
        'daily_admissions': daily_admissions.astype(int)
    })


def generate_scenario_data(scenario='normal'):
    scenarios = {
        'normal': {
            'base_admissions': 50,
            'weekday_impact': 10,
            'seasonal_impact': 15,
            'special_events_count': 5,
            'special_events_magnitude': 30
        },
        'high_variation': {
            'base_admissions': 50,
            'weekday_impact': 20,
            'seasonal_impact': 25,
            'special_events_count': 10,
            'special_events_magnitude': 40
        },
        'pandemic': {
            'base_admissions': 100,
            'weekday_impact': 5,
            'seasonal_impact': 30,
            'special_events_count': 20,
            'special_events_magnitude': 50
        },
        'summer_surge': {
            'base_admissions': 60,
            'weekday_impact': 15,
            'seasonal_impact': -25,
            'special_events_count': 8,
            'special_events_magnitude': 35
        }
    }
    params = scenarios.get(scenario, scenarios['normal'])
    return generate_synthetic_data(**params)


def add_department_breakdown(df, seed=42):
    """
    Decompose daily_admissions into ICU, Emergency, and General Ward streams.
    ICU ~15%, Emergency ~22%, General is the remainder (always >= 0).
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    total = df['daily_admissions'].values

    icu_frac = rng.normal(0.15, 0.02, n).clip(0.10, 0.22)
    emergency_frac = rng.normal(0.22, 0.03, n).clip(0.15, 0.30)
    # Clamp emergency so icu + emergency never exceeds 1.0
    emergency_frac = np.minimum(emergency_frac, 1.0 - icu_frac)
    general_frac = np.maximum(1.0 - icu_frac - emergency_frac, 0.0)

    df = df.copy()
    df['icu_admissions'] = (total * icu_frac).astype(int)
    df['emergency_admissions'] = (total * emergency_frac).astype(int)
    df['general_admissions'] = (total * general_frac).astype(int)
    return df


def load_hospital_data(scenario='normal', custom_params=None):
    if scenario == 'custom' and custom_params is not None:
        df = generate_synthetic_data(**custom_params)
    else:
        df = generate_scenario_data(scenario)

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

    # shift(1) so rolling features at time t use t-7..t-1 only — no current-day leakage
    shifted = df['daily_admissions'].shift(1)
    df['rolling_mean_7d'] = shifted.rolling(window=7, min_periods=1).mean()
    df['rolling_std_7d'] = shifted.rolling(window=7, min_periods=1).std().fillna(0)
    return df


def prepare_ml_features(df):
    features = ['day_of_week', 'month', 'is_weekend', 'rolling_mean_7d', 'rolling_std_7d']
    X = df[features].copy()

    X['rolling_mean_7d'] = X['rolling_mean_7d'].bfill().ffill()
    X['rolling_std_7d'] = X['rolling_std_7d'].fillna(0)

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].mean())

    y = df['daily_admissions']
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    # shuffle=False preserves temporal order — required for time-series evaluation
    return train_test_split(X, y, test_size=test_size, shuffle=False,
                            random_state=random_state)
