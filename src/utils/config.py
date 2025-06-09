"""
Configuration settings for the NBA prediction model.
"""
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering parameters
LOOKBACK_WINDOW = 10  # Number of previous games to consider
MIN_GAMES_PLAYED = 20  # Minimum games played for player statistics

# API settings
NBA_API_BASE_URL = "https://stats.nba.com/stats"
API_TIMEOUT = 30  # seconds

# Model training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 