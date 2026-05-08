from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "cpi_raw.csv"
OUTPUT_DIR = BASE_DIR / "output"

SIMILARITY_THRESHOLD = 0.75   
MIN_CITIES = 5
PROJECT_YEARS = [2023, 2024, 2025]

CITIES = [
    "Bahawalpur", "Faisalabad", "Gujranwala", "Hyderabad",
    "Islamabad", "Karachi", "Lahore", "Larkana", "Multan",
    "Peshawar", "Quetta", "Rawalpindi", "Sargodha", "Sialkot", "Sukkur",
]

CATEGORIES = [
    "Food & Non-Alcoholic Beverages",
    "Alcoholic Beverages, Tobacco",
    "Clothing and Footwear",
    "Housing, Water, Electricity",
    "Furnishing & HH Equipment",
    "Health",
    "Transport",
]

TAU_VALUES = [0.65, 0.70, 0.75, 0.80, 0.85]
K_VALUES = [3, 5, 7, 10]
WEIGHT_SCHEMES = ["unweighted", "weighted_city_count", "weighted_avg_similarity"]
