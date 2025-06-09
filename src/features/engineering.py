"""
Feature engineering module for NBA game data.
"""
from typing import List, Optional

import pandas as pd
import numpy as np

from ..utils.config import LOOKBACK_WINDOW, MIN_GAMES_PLAYED, PROCESSED_DATA_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    """Class for engineering features from NBA game data."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.lookback_window = LOOKBACK_WINDOW
        self.min_games_played = MIN_GAMES_PLAYED
    
    def calculate_team_stats(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team-level statistics from game data.
        
        Args:
            games_df: DataFrame containing game data
            
        Returns:
            pd.DataFrame: Team statistics
        """
        # Group by team and calculate basic stats
        team_stats = games_df.groupby('TEAM_ID').agg({
            'PTS': ['mean', 'std'],
            'AST': ['mean', 'std'],
            'REB': ['mean', 'std'],
            'FG_PCT': ['mean', 'std'],
            'FG3_PCT': ['mean', 'std'],
            'FT_PCT': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        team_stats.columns = ['_'.join(col).strip() for col in team_stats.columns.values]
        
        return team_stats
    
    def calculate_rolling_stats(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling statistics for teams.
        
        Args:
            games_df: DataFrame containing game data
            
        Returns:
            pd.DataFrame: Rolling statistics
        """
        # Sort by date
        games_df = games_df.sort_values('GAME_DATE')
        
        # Calculate rolling averages for each team
        rolling_stats = games_df.groupby('TEAM_ID').rolling(
            window=self.lookback_window,
            min_periods=1
        ).agg({
            'PTS': 'mean',
            'AST': 'mean',
            'REB': 'mean',
            'FG_PCT': 'mean',
            'FG3_PCT': 'mean',
            'FT_PCT': 'mean'
        }).reset_index()
        
        # Add suffix to indicate rolling stats
        rolling_stats.columns = [f"{col}_rolling" if col != 'TEAM_ID' else col 
                               for col in rolling_stats.columns]
        
        return rolling_stats
    
    def create_game_features(self, games_df: pd.DataFrame, 
                           team_stats_df: pd.DataFrame,
                           rolling_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for each game.
        
        Args:
            games_df: DataFrame containing game data
            team_stats_df: DataFrame containing team statistics
            rolling_stats_df: DataFrame containing rolling statistics
            
        Returns:
            pd.DataFrame: Game features
        """
        # Merge team stats
        features_df = pd.merge(
            games_df,
            team_stats_df,
            on='TEAM_ID',
            how='left'
        )
        
        # Merge rolling stats
        features_df = pd.merge(
            features_df,
            rolling_stats_df,
            on=['TEAM_ID', 'GAME_DATE'],
            how='left'
        )
        
        # Calculate additional features
        features_df['HOME_GAME'] = features_df['MATCHUP'].str.contains('vs.').astype(int)
        features_df['REST_DAYS'] = features_df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
        
        return features_df
    
    def save_features(self, features_df: pd.DataFrame, filename: str) -> None:
        """
        Save engineered features to a parquet file.
        
        Args:
            features_df: DataFrame containing features
            filename: Name of the file
        """
        filepath = PROCESSED_DATA_DIR / filename
        features_df.to_parquet(filepath)
        logger.info(f"Saved features to {filepath}")

def main():
    """Main function to engineer features from NBA data."""
    engineer = FeatureEngineer()
    
    try:
        # Load raw data
        games_df = pd.read_parquet(PROCESSED_DATA_DIR / "games_2023-24.parquet")
        
        # Calculate features
        team_stats = engineer.calculate_team_stats(games_df)
        rolling_stats = engineer.calculate_rolling_stats(games_df)
        game_features = engineer.create_game_features(games_df, team_stats, rolling_stats)
        
        # Save features
        engineer.save_features(game_features, "game_features_2023-24.parquet")
        
    except Exception as e:
        logger.error(f"Error engineering features: {str(e)}")
        raise

if __name__ == "__main__":
    main() 