import pandas as pd
import glob

# Path to your CSV files (adjust as needed)
csv_files = glob.glob("data/regular_season_box_scores_2010_2024_part_*.csv")


# Load and concatenate all CSVs
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Convert to Parquet (compressed by default with snappy)
df.to_parquet("box_totals_combined.parquet", index=False)

print("Combined Parquet file written successfully!")