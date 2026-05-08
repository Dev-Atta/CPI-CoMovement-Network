# vectors.py
# For each item in each city and year, we compute a price-change vector.
# The vector has 11 values — one for each month transition (Jan->Feb, Feb->Mar, ... Nov->Dec).
# Each value is the percentage change in price from one month to the next.
import pandas as pd
from config import PROJECT_YEARS

def compute_price_vectors(df):
    """
    Returns a nested dictionary: vectors[year][city][item] = list of 11 price changes.

    A price-change vector captures HOW the price moved each month, not what the price was.
    This lets us compare movement patterns between items fairly, regardless of their price level.
    """
    vectors = {}

    for year in PROJECT_YEARS:
        vectors[year] = {}
        year_df = df[df["year"] == year]

        cities = sorted(year_df["city"].unique())
        items  = sorted(year_df["item_name"].unique())

        for city in cities:
            vectors[year][city] = {}
            city_df = year_df[year_df["city"] == city]

            for item in items:
                item_df = city_df[city_df["item_name"] == item]

                # Build a simple lookup: month number -> price
                monthly_prices = {}
                for _, row in item_df.iterrows():
                    monthly_prices[int(row["month"])] = float(row["price_index"])

                # Compute percentage change from month m-1 to month m, for m = 2 to 12
                # This gives us 11 values (months 2,3,4,...,12)
                price_changes = []
                for m in range(2, 13):
                    prev_price = monthly_prices.get(m - 1, None)
                    curr_price = monthly_prices.get(m,     None)

                    if prev_price is None or curr_price is None or prev_price == 0:
                        # If data is missing or previous price is zero, record no change
                        price_changes.append(0.0)
                    else:
                        pct_change = (curr_price - prev_price) / prev_price
                        price_changes.append(pct_change)

                vectors[year][city][item] = price_changes

    return vectors