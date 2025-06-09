"""
Data processing module for NBA game data.
"""
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataProcessor:
    """Class for processing NBA game data."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.player_box_df = None
        self.team_game_df = None
    
    def load_data(self) -> None:
        """Load the raw data files."""
        try:
            # Load player box scores
            self.player_box_df = pd.read_parquet(RAW_DATA_DIR / "box_totals_combined.parquet")
            logger.info(f"Loaded player box score data with shape: {self.player_box_df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self) -> None:
        """
        Validate the loaded data for quality and consistency.
        """
        # Check for missing values
        missing_values = self.player_box_df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
        
        # Check for duplicate rows
        duplicates = self.player_box_df.duplicated(subset=['gameId', 'teamId', 'personId'])
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate player-game entries")
        
        # Validate date ranges
        min_date = self.player_box_df['game_date'].min()
        max_date = self.player_box_df['game_date'].max()
        logger.info(f"Data covers period from {min_date} to {max_date}")
        
        # Validate team consistency
        unique_teams = self.player_box_df['teamId'].nunique()
        logger.info(f"Found {unique_teams} unique teams in the dataset")
    
    def clean_data(self) -> None:
        """
        Clean and preprocess the data.
        """
        # Convert date column to datetime
        self.player_box_df['game_date'] = pd.to_datetime(self.player_box_df['game_date'])
        
        # Sort by date
        self.player_box_df = self.player_box_df.sort_values('game_date')
        
        # Remove any duplicate entries
        self.player_box_df = self.player_box_df.drop_duplicates(subset=['gameId', 'teamId', 'personId'])
        
        # Fill missing values with appropriate defaults
        numeric_columns = self.player_box_df.select_dtypes(include=[np.number]).columns
        self.player_box_df[numeric_columns] = self.player_box_df[numeric_columns].fillna(0)
        
        logger.info("Data cleaning completed")
    
    def aggregate_to_team_game(self) -> pd.DataFrame:
        """
        Aggregate player-level box scores to team-level per game.
        Returns:
            pd.DataFrame: Team-game level features
        """
        # List of columns to aggregate
        agg_dict = {
            'minutes': 'sum',
            'fieldGoalsMade': 'sum',
            'fieldGoalsAttempted': 'sum',
            'fieldGoalsPercentage': 'mean',
            'threePointersMade': 'sum',
            'threePointersAttempted': 'sum',
            'threePointersPercentage': 'mean',
            'freeThrowsMade': 'sum',
            'freeThrowsAttempted': 'sum',
            'freeThrowsPercentage': 'mean',
            'reboundsOffensive': 'sum',
            'reboundsDefensive': 'sum',
            'reboundsTotal': 'sum',
            'assists': 'sum',
            'steals': 'sum',
            'blocks': 'sum',
            'turnovers': 'sum',
            'foulsPersonal': 'sum',
            'points': 'sum',
            'plusMinusPoints': 'sum',
        }
        # Group by game and team
        team_game_df = self.player_box_df.groupby([
            'season_year', 'game_date', 'gameId', 'matchup', 'teamId', 'teamCity', 'teamName', 'teamTricode', 'teamSlug'
        ]).agg(agg_dict).reset_index()
        logger.info(f"Aggregated to team-game level with shape: {team_game_df.shape}")
        self.team_game_df = team_game_df
        return team_game_df
    
    def save_processed_data(self, features_df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to a parquet file.
        
        Args:
            features_df: DataFrame containing processed features
            filename: Name of the output file
        """
        filepath = PROCESSED_DATA_DIR / filename
        features_df.to_parquet(filepath)
        logger.info(f"Saved processed data to {filepath}")

def main():
    """Main function to process NBA data."""
    processor = DataProcessor()
    
    try:
        # Load data
        processor.load_data()
        
        # Validate data
        processor.validate_data()
        
        # Clean data
        processor.clean_data()
        
        # Aggregate to team-game level
        team_game_df = processor.aggregate_to_team_game()
        
        # Save processed data
        processor.save_processed_data(team_game_df, "team_game_features.parquet")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 