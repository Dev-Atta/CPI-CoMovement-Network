import pandas as pd
from pathlib import Path
from config import DATA_FILE, OUTPUT_DIR

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df.columns = ["year", "month", "city", "item_name", "category", "price_index"]
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["price_index"] = pd.to_numeric(df["price_index"], errors="coerce")
    df = df.sort_values(["year", "month", "item_name", "city"]).reset_index(drop=True)
    return df


def get_items(df: pd.DataFrame) -> list:
    return sorted(df["item_name"].unique().tolist())


def get_cities(df: pd.DataFrame) -> list:
    return sorted(df["city"].unique().tolist())


def save_master(df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "master_data.csv"
    df.to_csv(path, index=False)
    return path


def summarize(df: pd.DataFrame) -> str:
    years = sorted(df["year"].unique().tolist())
    items = df["item_name"].nunique()
    cities = df["city"].nunique()
    months = df.groupby("year")["month"].nunique().to_dict()
    missing = int(df["price_index"].isna().sum())
    return (
        f"Years: {years} | Items: {items} | Cities: {cities} | "
        f"Months per year: {months} | Missing values: {missing}"
    )
