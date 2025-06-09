# NBA Game Prediction Model 2025

A machine learning model for predicting NBA game outcomes using historical data and advanced statistics.

## Project Structure

```
nba_prediction_model_2025/
├── src/
│   ├── data/           # Data collection and processing scripts
│   ├── features/       # Feature engineering code
│   ├── models/         # Model training and evaluation code
│   └── utils/          # Utility functions and helpers
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks for analysis
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Setup

1. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Development

- Use `black` for code formatting
- Use `flake8` for linting
- Write tests for new functionality
- Document code with docstrings

## Data Sources

- NBA API for current season data
- Historical game data from various sources
- Player statistics and team performance metrics

## Model Details

The model uses various features including:

- Team statistics
- Player performance metrics
- Historical matchup data
- Home/away performance
- Rest days and travel distance

## License

MIT License
