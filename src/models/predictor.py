"""
Prediction module for NBA game predictions.
"""
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from joblib import load

from ..utils.config import MODELS_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class GamePredictor:
    """Class for making NBA game predictions."""
    
    def __init__(self, model_name: str):
        """
        Initialize the game predictor.
        
        Args:
            model_name: Name of the model to load
        """
        self.model = None
        self.scaler = None
        self.load_model(model_name)
    
    def load_model(self, model_name: str) -> None:
        """
        Load the trained model and scaler.
        
        Args:
            model_name: Name of the model
        """
        try:
            # Load model
            model_path = MODELS_DIR / f"{model_name}.joblib"
            self.model = load(model_path)
            
            # Load scaler
            scaler_path = MODELS_DIR / f"{model_name}_scaler.joblib"
            self.scaler = load(scaler_path)
            
            logger.info(f"Loaded model and scaler from {MODELS_DIR}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            features_df: DataFrame containing features
            
        Returns:
            np.ndarray: Scaled features
        """
        # Define feature columns
        feature_cols = [col for col in features_df.columns 
                       if col not in ['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'WIN']]
        
        # Extract features
        X = features_df[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict_games(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for games.
        
        Args:
            features_df: DataFrame containing features
            
        Returns:
            pd.DataFrame: Predictions with probabilities
        """
        # Prepare features
        X = self.prepare_features(features_df)
        
        # Make predictions
        win_probs = self.model.predict_proba(X)[:, 1]
        
        # Create predictions DataFrame
        predictions_df = features_df[['GAME_ID', 'TEAM_ID', 'GAME_DATE']].copy()
        predictions_df['WIN_PROBABILITY'] = win_probs
        
        return predictions_df
    
    def get_game_predictions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get predictions for each game with both teams.
        
        Args:
            features_df: DataFrame containing features
            
        Returns:
            pd.DataFrame: Game predictions with both teams
        """
        # Get predictions
        predictions_df = self.predict_games(features_df)
        
        # Pivot to get both teams for each game
        game_predictions = predictions_df.pivot(
            index='GAME_ID',
            columns='TEAM_ID',
            values='WIN_PROBABILITY'
        ).reset_index()
        
        # Rename columns
        game_predictions.columns = ['GAME_ID', 'HOME_TEAM_PROB', 'AWAY_TEAM_PROB']
        
        # Add game date
        game_predictions = pd.merge(
            game_predictions,
            features_df[['GAME_ID', 'GAME_DATE']].drop_duplicates(),
            on='GAME_ID'
        )
        
        return game_predictions

def main():
    """Main function to make NBA game predictions."""
    try:
        # Load features
        features_df = pd.read_parquet("game_features_2023-24.parquet")
        
        # Initialize predictor
        predictor = GamePredictor("nba_prediction_model_2023-24")
        
        # Get predictions
        game_predictions = predictor.get_game_predictions(features_df)
        
        # Save predictions
        predictions_path = MODELS_DIR / "game_predictions_2023-24.parquet"
        game_predictions.to_parquet(predictions_path)
        
        logger.info(f"Saved predictions to {predictions_path}")
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise

if __name__ == "__main__":
    main() 