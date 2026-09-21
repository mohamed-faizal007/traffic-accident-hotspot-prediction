from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
# ============================================================
# EDA OUTPUT PATHS
# ============================================================

EDA_FIGURES_DIR = FIGURES_DIR / "eda"


# ============================================================
# DATA FILE PATHS
# ============================================================

RAW_COLLISIONS_PATH = RAW_DATA_DIR / "collisions_raw.csv"

CLEAN_COLLISIONS_PATH = (
    INTERIM_DATA_DIR / "collisions_clean.csv"
)

QUALITY_REPORT_PATH = (
    INTERIM_DATA_DIR / "data_quality_report.csv"
)


# ============================================================
# REQUIRED PROJECT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [

    # Unique identifier
    "collision_index",

    # Year
    "collision_year",

    # Spatial
    "longitude",
    "latitude",

    # Temporal
    "date",
    "time",
    "day_of_week",

    # Collision information
    "collision_severity",
    "number_of_vehicles",
    "number_of_casualties",

    # Road information
    "road_type",
    "speed_limit",
    "junction_detail",
    "junction_control",

    # Environmental information
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",

    # Area information
    "urban_or_rural_area"
]


# ============================================================
# CRS SETTINGS
# ============================================================

WGS84_CRS = "EPSG:4326"

UK_PROJECTED_CRS = "EPSG:27700"


# ============================================================
# UK COORDINATE VALIDATION
# ============================================================

UK_LAT_MIN = 49
UK_LAT_MAX = 61

UK_LON_MIN = -8
UK_LON_MAX = 2


# ============================================================
# SPATIAL SETTINGS
# ============================================================
# Grid cells are 500 m x 500 m squares in the British National Grid
# (EPSG:27700). DBSCAN parameters are chosen from training data in
# hotspot_detection.py (results/dbscan_selection.json), not set here.

GRID_SIZE_METERS = 500


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# EVALUATION PROTOCOL (single source of truth)
# ============================================================
# Splits are defined by the TARGET month (the month being predicted).
#   train: target month <= 2023-12
#   val  : target month in 2024   (threshold, calibration, hyperparameters)
#   test : target month in 2025   (evaluated once, with a frozen config)

FIRST_MONTH = "2021-01"
LAST_MONTH = "2025-12"
TRAIN_END_MONTH = "2023-12"
VAL_YEAR = 2024
TEST_YEAR = 2025

# A grid is "active" if it has at least this many collisions in the
# training period (2021-2023) only.
ACTIVE_MIN_COLLISIONS = 3

# Target: >= this many collisions in the next month.
HOTSPOT_MIN_COLLISIONS = 2

# Months at the start of each grid history dropped as feature warm-up.
WARMUP_MONTHS = 3

# Rank-based risk tiers: share of grids (per month) in each tier.
TIER_TOP_FRACTIONS = {"Critical": 0.01, "High": 0.05, "Medium": 0.10}

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
FROZEN_CONFIG_PATH = RESULTS_DIR / "frozen_config.json"
FEATURES_PATH = PROCESSED_DATA_DIR / "ml_features.parquet"


def ensure_output_dirs(root=PROJECT_ROOT):
    """Create the git-ignored output folders (a fresh clone has none of them)."""
    for relative in ("data/raw", "data/interim", "data/processed", "models", "outputs", "results"):
        (Path(root) / relative).mkdir(parents=True, exist_ok=True)
