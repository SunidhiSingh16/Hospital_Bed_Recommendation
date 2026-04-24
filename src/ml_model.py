import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

class HospitalAdmissionPredictor:
    def __init__(self, model_type='gbm'):
        """
        Initialize the predictor
        Args:
            model_type (str): Type of model to use ('gbm' or 'xgboost')
        """
        self.model_type = model_type
        if model_type == 'gbm':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
        else:
            raise ValueError("Invalid model type. Choose 'gbm' or 'xgboost'")
    
    def fit(self, X_train, y_train):
        """
        Train the model
        """
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """
        Make predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        Returns:
            dict: Dictionary containing MAE and RMSE
        """
        predictions = self.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        return {
            'mae': mae,
            'rmse': rmse
        }
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores
        Args:
            feature_names (list): List of feature names
        Returns:
            dict: Dictionary mapping features to their importance scores
        """
        if self.model_type == 'gbm':
            importance = self.model.feature_importances_
        else:  # xgboost
            importance = self.model.feature_importances_
            
        return dict(zip(feature_names, importance)) 