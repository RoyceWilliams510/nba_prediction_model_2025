"""
Data collection module for NBA game data.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from ..utils.config import NBA_API_BASE_URL, API_TIMEOUT, RAW_DATA_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class NBADataCollector:
    """Class for collecting NBA game data from the NBA API."""
    
    def __init__(self):
        """Initialize the NBA data collector."""
        self.base_url = NBA_API_BASE_URL
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make a request to the NBA API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Dict: API response
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise
    
    def get_game_data(self, season: str, game_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get game data for a specific season or game.
        
        Args:
            season: NBA season (e.g., '2023-24')
            game_id: Optional specific game ID
            
        Returns:
            pd.DataFrame: Game data
        """
        endpoint = "leaguegamefinder"
        params = {
            'LeagueID': '00',
            'Season': season,
            'GameID': game_id if game_id else ''
        }
        
        data = self._make_request(endpoint, params)
        return pd.DataFrame(data['resultSets'][0]['rowSet'])
    
    def get_player_stats(self, season: str, player_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get player statistics for a specific season or player.
        
        Args:
            season: NBA season (e.g., '2023-24')
            player_id: Optional specific player ID
            
        Returns:
            pd.DataFrame: Player statistics
        """
        endpoint = "leaguedashplayerstats"
        params = {
            'LeagueID': '00',
            'Season': season,
            'PlayerID': player_id if player_id else '',
            'PerMode': 'PerGame'
        }
        
        data = self._make_request(endpoint, params)
        return pd.DataFrame(data['resultSets'][0]['rowSet'])
    
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save data to a parquet file.
        
        Args:
            data: DataFrame to save
            filename: Name of the file
        """
        filepath = RAW_DATA_DIR / filename
        data.to_parquet(filepath)
        logger.info(f"Saved data to {filepath}")

def main():
    """Main function to collect and save NBA data."""
    collector = NBADataCollector()
    
    # Collect current season data
    current_season = "2023-24"
    
    try:
        # Get game data
        game_data = collector.get_game_data(current_season)
        collector.save_data(game_data, f"games_{current_season}.parquet")
        
        # Get player stats
        player_stats = collector.get_player_stats(current_season)
        collector.save_data(player_stats, f"player_stats_{current_season}.parquet")
        
    except Exception as e:
        logger.error(f"Error collecting data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 