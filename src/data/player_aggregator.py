"""
Player data aggregation module for NBA game data.
This module aggregates individual player statistics from the raw parquet file to create a player-level dataset for future reference in the prediction model.
"""
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class PlayerAggregator:
    """Class for aggregating individual player data from NBA game data."""

    def __init__(self):
        """Initialize the player aggregator."""
        self.player_box_df = None
        self.player_agg_df = None

    def load_data(self) -> None:
        """Load the raw player box score data."""
        try:
            self.player_box_df = pd.read_parquet(RAW_DATA_DIR / "box_totals_combined.parquet")
            logger.info(f"Loaded player box score data with shape: {self.player_box_df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self) -> None:
        """
        Validate the loaded player data for quality and consistency.
        """
        # Check for missing values
        missing_values = self.player_box_df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")

        # Check for duplicate player-game entries
        duplicates = self.player_box_df.duplicated(subset=['gameId', 'teamId', 'personId'])
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate player-game entries")

        # Validate date ranges
        min_date = self.player_box_df['game_date'].min()
        max_date = self.player_box_df['game_date'].max()
        logger.info(f"Data covers period from {min_date} to {max_date}")

        # Validate player consistency
        unique_players = self.player_box_df['personId'].nunique()
        logger.info(f"Found {unique_players} unique players in the dataset")

    def clean_data(self) -> None:
        """
        Clean and preprocess the player data.
        """
        # Convert date column to datetime
        self.player_box_df['game_date'] = pd.to_datetime(self.player_box_df['game_date'])

        # Sort by date
        self.player_box_df = self.player_box_df.sort_values('game_date')

        # Remove any duplicate entries
        self.player_box_df = self.player_box_df.drop_duplicates(subset=['gameId', 'teamId', 'personId'])

        # Parse 'minutes' column to handle formats like 'MM:SS' or 'MM:SS:MS'
        def parse_minutes(min_str):
            if pd.isna(min_str) or min_str == '':
                return 0
            parts = min_str.split(':')
            if len(parts) == 2:
                return float(parts[0]) + float(parts[1]) / 60
            elif len(parts) == 3:
                return float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600
            return 0

        self.player_box_df['minutes'] = self.player_box_df['minutes'].apply(parse_minutes)

        # Fill missing values with appropriate defaults
        numeric_columns = self.player_box_df.select_dtypes(include=[np.number]).columns
        self.player_box_df[numeric_columns] = self.player_box_df[numeric_columns].fillna(0)

        logger.info("Player data cleaning completed")

    def aggregate_player_data(self) -> pd.DataFrame:
        """
        Aggregate individual player data to create a player-level dataset, grouped by year.
        Returns:
            pd.DataFrame: Aggregated player-level data, grouped by year
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

        # Group by player and year
        player_agg_df = self.player_box_df.groupby([
            'personId', 'personName', 'position', 'teamId', 'teamName', 'teamTricode', 'season_year'
        ]).agg(agg_dict).reset_index()

        # Convert columns to numeric
        for col in agg_dict.keys():
            player_agg_df[col] = pd.to_numeric(player_agg_df[col], errors='coerce')

        # Ensure 'minutes' is numeric and handle NaN values
        player_agg_df['minutes'] = pd.to_numeric(player_agg_df['minutes'], errors='coerce').fillna(0)

        # Calculate per-game averages
        player_agg_df['games_played'] = self.player_box_df.groupby(['personId', 'season_year']).size().reset_index(name='games_played')['games_played']
        for col in agg_dict.keys():
            if col not in ['fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage']:
                player_agg_df[f'{col}_per_game'] = player_agg_df[col] / player_agg_df['games_played']

        # Calculate per-48-minute estimates
        for col in agg_dict.keys():
            if col not in ['fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage']:
                player_agg_df[f'{col}_per_48'] = (player_agg_df[col] / player_agg_df['minutes'].replace(0, np.nan)) * 48

        logger.info(f"Aggregated player data by year with shape: {player_agg_df.shape}")
        self.player_agg_df = player_agg_df
        return player_agg_df

    def save_aggregated_data(self, filename: str) -> None:
        """
        Save aggregated player data to a parquet file.
        Args:
            filename: Name of the output file
        """
        filepath = PROCESSED_DATA_DIR / filename
        self.player_agg_df.to_parquet(filepath)
        logger.info(f"Saved aggregated player data to {filepath}")

def main():
    """Main function to aggregate player data."""
    aggregator = PlayerAggregator()

    try:
        # Load data
        aggregator.load_data()

        # Validate data
        aggregator.validate_data()

        # Clean data
        aggregator.clean_data()

        # Aggregate player data
        player_agg_df = aggregator.aggregate_player_data()

        # Save aggregated data
        aggregator.save_aggregated_data("player_aggregated_data.parquet")

    except Exception as e:
        logger.error(f"Error aggregating player data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 