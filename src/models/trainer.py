"""
Model training module for NBA game predictions.
"""
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from joblib import dump

from ..utils.config import (
    RANDOM_SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODELS_DIR
)
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelTrainer:
    """Class for training NBA game prediction models."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            features_df: DataFrame containing features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and target arrays
        """
        # Define features and target
        feature_cols = [col for col in features_df.columns 
                       if col not in ['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'WIN']]
        target_col = 'WIN'
        
        # Split features and target
        X = features_df[feature_cols]
        y = features_df[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple containing train, validation, and test sets
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED
        )
        
        # Second split: separate validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_SEED
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        # Initialize model
        self.model = xgb.XGBClassifier(
            n_estimators=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=RANDOM_SEED
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING_PATIENCE,
            verbose=True
        )
        
        logger.info("Model training completed")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        log_loss = -np.mean(y_test * np.log(y_pred_proba + 1e-15) + 
                          (1 - y_test) * np.log(1 - y_pred_proba + 1e-15))
        
        metrics = {
            'accuracy': accuracy,
            'log_loss': log_loss
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, model_name: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            model_name: Name of the model
        """
        # Save model
        model_path = MODELS_DIR / f"{model_name}.joblib"
        dump(self.model, model_path)
        
        # Save scaler
        scaler_path = MODELS_DIR / f"{model_name}_scaler.joblib"
        dump(self.scaler, scaler_path)
        
        logger.info(f"Saved model and scaler to {MODELS_DIR}")

def main():
    """Main function to train the NBA prediction model."""
    trainer = ModelTrainer()
    
    try:
        # Load features
        features_df = pd.read_parquet("game_features_2023-24.parquet")
        
        # Prepare data
        X, y = trainer.prepare_data(features_df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
        
        # Train model
        trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        metrics = trainer.evaluate_model(X_test, y_test)
        
        # Save model
        trainer.save_model("nba_prediction_model_2023-24")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    main() 