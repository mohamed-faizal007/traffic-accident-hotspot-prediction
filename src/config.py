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
MAPS_DIR = OUTPUTS_DIR / "maps"


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

GRID_SIZE_METERS = 500

DBSCAN_EPS_METERS = 500

DBSCAN_MIN_SAMPLES = 15


# ============================================================
# TEMPORAL SETTINGS
# ============================================================

TIME_PERIOD = "M"

TEST_PERIODS = 6


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42