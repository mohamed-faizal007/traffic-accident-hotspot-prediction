from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw/collisions_raw.csv")


def load_raw_data():
    print("=" * 60)
    print("LOADING RAW COLLISION DATA")
    print("=" * 60)

    df = pd.read_csv(
        RAW_DATA_PATH,
        low_memory=False
    )

    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    load_raw_data()