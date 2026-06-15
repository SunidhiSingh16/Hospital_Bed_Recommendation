from collections import deque
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.ml_model import HospitalAdmissionPredictor
from src.poisson_model import PoissonPredictor


def make_forward_forecast(predictor, df, n=7):
    """
    Forward-project n calendar days using any model with a predict(DataFrame) method.

    Rolling features are updated after each predicted step so that day k+1 uses
    the prediction from day k — no stale frozen-feature propagation.
    Confidence bands = ±1.96σ of recent 7-day admission volatility.
    """
    last_date = df['date'].iloc[-1]
    rolling_std = max(float(df['rolling_std_7d'].iloc[-1]), 1.0)

    # Seed the rolling window with the last 7 actual observations
    history = deque(df['daily_admissions'].tail(7).tolist(), maxlen=7)

    records = []
    for i in range(1, n + 1):
        future_date = last_date + pd.Timedelta(days=i)
        rolling_mean = float(np.mean(history))

        X_future = pd.DataFrame([{
            'day_of_week': future_date.dayofweek,
            'month': future_date.month,
            'is_weekend': int(future_date.dayofweek >= 5),
            'rolling_mean_7d': rolling_mean,
            'rolling_std_7d': rolling_std,
        }])
        pred = float(predictor.predict(X_future)[0])

        # Feed prediction back into rolling window for next step
        history.append(pred)

        records.append({
            'date': future_date,
            'prediction': pred,
            'ci_lower': max(0.0, pred - 1.96 * rolling_std),
            'ci_upper': pred + 1.96 * rolling_std,
        })

    return pd.DataFrame(records)


class EnsembleForecaster:
    """
    Weighted ensemble combining Poisson, GBM, and XGBoost regressors.

    Default weights: Poisson 25% | GBM 40% | XGBoost 35%
    Caller-supplied weights are auto-normalised to sum to 1.0.
    """

    DEFAULT_WEIGHTS = {'poisson': 0.25, 'gbm': 0.40, 'xgboost': 0.35}

    def __init__(self, weights=None):
        w = weights or dict(self.DEFAULT_WEIGHTS)
        # Normalise so weights always sum to exactly 1.0
        total = sum(w.values())
        if total <= 0:
            raise ValueError("Ensemble weights must be positive.")
        self.weights = {k: v / total for k, v in w.items()}

        self.gbm_model = HospitalAdmissionPredictor(model_type='gbm')
        self.xgb_model = HospitalAdmissionPredictor(model_type='xgboost')
        self.poisson_model = PoissonPredictor()
        self._fitted = False

    def fit(self, X_train, y_train, recent_data):
        self.gbm_model.fit(X_train, y_train)
        self.xgb_model.fit(X_train, y_train)
        self.poisson_model.fit(recent_data)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        gbm_pred = self.gbm_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        poisson_pred = np.full(len(X), self.poisson_model.lambda_)
        return (
            self.weights['poisson'] * poisson_pred
            + self.weights['gbm'] * gbm_pred
            + self.weights['xgboost'] * xgb_pred
        )

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {
            'mae': mean_absolute_error(y_test, preds),
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
        }

    def get_feature_importance(self, feature_names):
        """Averaged feature importance across GBM and XGBoost sub-models."""
        gbm_imp = np.array(list(self.gbm_model.get_feature_importance(feature_names).values()))
        xgb_imp = np.array(list(self.xgb_model.get_feature_importance(feature_names).values()))
        return dict(zip(feature_names, (gbm_imp + xgb_imp) / 2))

    def forecast_next_n_days(self, df, n=7):
        if not self._fitted:
            raise RuntimeError("Call fit() before forecasting")
        return make_forward_forecast(self, df, n)
