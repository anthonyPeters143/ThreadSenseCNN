"""
constants.py
Defines project-wide constants, file paths, column labels, and menu options.

Dependencies:
- None (used by all other modules)

Required Imports:
- None
"""
####################################################################################################
# SYSTEM CONSTANCES
####################################################################################################

# Paths
INPUT = "input"
PREDICTION_INPUT = "prediction_input"

INPUT_CSV = "input_csv.csv"
INPUT_PREPROCESSED_CSV = "input_csv_preprocessed.csv"
IMAGES = "input_images"
ORGINAL_STATS_CSV = "orginal_input_csv_stats.csv"

MODEL_HISTORY = "model_history.csv"

COMPARSION_TABLE_CSV = "comparsion_table.csv"

COMPARSION_GRAPH = "comparsion_graph.png"

TRAINING_LOGS = "training_logs"

MODEL_CHCEKPOINTS = "model_checkpoints"
BEST_MODEL = "best_model.keras"

# Labes for columns
IMAGE_PATH = "image path"

THREAD_COUNT = "thread count"
WIDTH = "width (centimeter)"
HEIGHT = "height (centimeter)"

WIDTH_NORM = "width_norm"
HEIGHT_NORM = "height_norm"
THREAD_COUNT_NORM = "thread_count_norm"

WIDTH_STD = "width_std"
HEIGHT_STD = "height_std"
THREAD_COUNT_STD = "thread_count_std"

WIDTH_MEAN = "width_mean"
HEIGHT_MEAN = "height_mean"
THREAD_COUNT_MEAN = "thread_count_mean"

WIDTH_ACTUAL = "width_actual"
HEIGHT_ACTUAL = "height_actual"
THREAD_COUNT_ACTUAL = "thread_count_actual"

WIDTH_PREDICTED = "width_predicted"
HEIGHT_PREDICTED = "height_predicted"
THREAD_COUNT_PREDICTED = "thread_count_predicted"

THREAD_COUNT_ABS_ERROR = "thread_count_abs_error"
WIDTH_ABS_ERROR = "width_count_abs_error"
HEIGHT_ABS_ERROR = "height_count_abs_error"

# Menu labels
EXIT = "Exit system"
SCRAPE = "Scrape data"
PROCESSING = "Process data"
TRAINING = "Train using data"
WHOLE = "Data pipeline"
PREDICTING = "Predicting using data"
DISPLAY = "Display data"

# Set max dimenision for images
MAX_DIMENSION = 1024