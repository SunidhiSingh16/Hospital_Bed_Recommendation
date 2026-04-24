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
    """
    Generate synthetic hospital admissions data with configurable parameters
    Args:
        start_date (str): Start date for the data (defaults to current year start)
        days (int): Number of days to generate
        base_admissions (int): Average daily admissions
        weekday_impact (int): How much weekdays increase admissions
        seasonal_impact (int): How much seasons affect admissions
        special_events_count (int): Number of special events (like disease outbreaks)
        special_events_magnitude (int): How much special events increase admissions
    Returns:
        DataFrame: Synthetic hospital admissions data
    """
    # Use current year as default start date
    if start_date is None:
        current_year = datetime.now().year
        start_date = f'{current_year}-01-01'
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Base number of daily admissions
    base = np.random.normal(loc=base_admissions, scale=base_admissions*0.2, size=days)
    
    # Add weekly pattern (more admissions on weekdays)
    weekday_effect = np.where(pd.Series(dates).dt.dayofweek < 5, 
                             weekday_impact, 
                             -weekday_impact/2)
    
    # Add monthly pattern (more admissions in winter months)
    month_nums = pd.Series(dates).dt.month
    monthly_effect = np.where(
        (month_nums >= 11) | (month_nums <= 2),
        seasonal_impact,  # winter increase
        np.where(
            (month_nums >= 6) & (month_nums <= 8),
            -seasonal_impact/2,  # summer decrease
            0  # spring/fall baseline
        )
    )
    
    # Add special events (like disease outbreaks)
    special_events = np.zeros(days)
    if special_events_count > 0:
        event_days = np.random.choice(days, size=special_events_count, replace=False)
        special_events[event_days] = np.random.normal(
            loc=special_events_magnitude, 
            scale=special_events_magnitude*0.2, 
            size=special_events_count
        )
    
    # Combine all effects
    daily_admissions = base + weekday_effect + monthly_effect + special_events
    
    # Ensure no negative values
    daily_admissions = np.maximum(daily_admissions, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'daily_admissions': daily_admissions.astype(int)
    })
    
    return df

def generate_scenario_data(scenario='normal'):
    """
    Generate data for different scenarios
    Args:
        scenario (str): One of 'normal', 'high_variation', 'pandemic', 'summer_surge'
    Returns:
        DataFrame: Hospital admissions data for the scenario
    """
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
            'seasonal_impact': -25,  # Reverse seasonal pattern
            'special_events_count': 8,
            'special_events_magnitude': 35
        }
    }
    
    params = scenarios.get(scenario, scenarios['normal'])
    return generate_synthetic_data(**params)

def load_hospital_data(scenario='normal', custom_params=None):
    """
    Load and preprocess the hospital admissions dataset
    Args:
        scenario (str): Which scenario to generate data for
        custom_params (dict): Optional custom parameters for data generation
    Returns:
        DataFrame: Processed hospital admissions data
    """
    # Generate synthetic data based on scenario or custom parameters
    if scenario == 'custom' and custom_params is not None:
        df = generate_synthetic_data(**custom_params)
    else:
        df = generate_scenario_data(scenario)
    
    # Extract additional features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Calculate rolling statistics with minimum periods to avoid NaN at start
    df['rolling_mean_7d'] = df['daily_admissions'].rolling(window=7, min_periods=1).mean()
    df['rolling_std_7d'] = df['daily_admissions'].rolling(window=7, min_periods=1).std().fillna(0)
    
    return df

def prepare_ml_features(df):
    """
    Prepare features for machine learning model
    Args:
        df (DataFrame): Input dataframe
    Returns:
        tuple: X (features) and y (target) for ML model
    """
    features = ['day_of_week', 'month', 'is_weekend', 'rolling_mean_7d', 'rolling_std_7d']
    X = df[features].copy()
    
    # Fill any remaining NaN values
    X['rolling_mean_7d'] = X['rolling_mean_7d'].fillna(method='bfill').fillna(method='ffill')
    X['rolling_std_7d'] = X['rolling_std_7d'].fillna(0)  # Fill NaN with 0 for standard deviation
    
    # Ensure all features are numeric and have no NaN values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].mean())
    
    y = df['daily_admissions']
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state) 