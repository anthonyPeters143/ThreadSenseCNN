"""
menu.py
Main command-line entry point for the thread estimation pipeline.
Provides user menu for scraping, processing, training, and prediction steps.

Dependencies:
- constants.py
- utility.py
- scraping.py
- processing.py
- models.py
- displaying.py

Required Imports:
- os
- sys
- pathlib.Path
"""
####################################################################################################
# IMPORTS
####################################################################################################

import os
import sys
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import constants
import utility
from scraping import scrape_data_from
from processing import data_preprocessing
from models import train_model, model_prediction, create_model_v2
from displaying import display_prediction

####################################################################################################
# MENU
####################################################################################################

# Menu loop, will repeat till exit code is entered
# Initlized exit and valid flag to false
exit_flag = False
valid_flag = False

# Determine project name or else use default name
try:
    project_name = current_file = Path(__file__).parent.resolve().name

except:
    project_name = "ThreadSenseCNN"

# Output system start up
print(f"\nStarting  {project_name}...")

while not exit_flag:
    menu_input = input(f"\t0 - {constants.EXIT}\n\t1 - {constants.SCRAPE}\n\t2 - {constants.PROCESSING}\n\t3 - {constants.TRAINING}\n\t4 - {constants.WHOLE}\n\t5 - {constants.PREDICTING}\nEnter menu selection : ")

    if (menu_input == "0"):
        # Flip exit flag
        exit_flag = True

    elif (menu_input == "1"):
        # Generate infomation from database of DST files and output validation flag
        print(utility.outcome_to_message(scrape_data_from(constants.INPUT, constants.INPUT_CSV, constants.IMAGES), constants.SCRAPE))

    elif (menu_input == "2"):
        # Preprocess infomation
        print(utility.outcome_to_message(data_preprocessing(constants.INPUT_CSV, constants.INPUT_PREPROCESSED_CSV, constants.ORGINAL_STATS_CSV, training=True), constants.PROCESSING))

    elif (menu_input == "3"):
        # Train model from generated infomation
        print(utility.outcome_to_message(train_model(constants.INPUT_PREPROCESSED_CSV, create_model_v2(), constants.MODEL_HISTORY), constants.TRAINING))

    elif (menu_input == "4"):
        # Run whole system, checking if processes are valid between each step and if not aborting
        # Generate infomation from database of DST files
        valid_flag = scrape_data_from(constants.INPUT, constants.INPUT_CSV, constants.IMAGES)
        print(utility.outcome_to_message(valid_flag, constants.SCRAPE))

        # Check if process was valid 
        if valid_flag:
            # Preprocess infomation
            valid_flag = data_preprocessing(constants.INPUT_CSV, constants.INPUT_PREPROCESSED_CSV, constants.ORGINAL_STATS_CSV, training=True)
            print(utility.outcome_to_message(valid_flag, constants.PROCESSING))

        # Check if process was valid 
        if valid_flag:
            # Train model from generated infomation
            valid_flag = train_model(constants.INPUT_PREPROCESSED_CSV, create_model_v2(), constants.MODEL_HISTORY)
            print(utility.outcome_to_message(valid_flag, constants.TRAINING))


    elif (menu_input == "5"):
        # Run prediction, checking if processes are valid between each step and if not aborting
        # Determine which model checkpoint to use and verify is it valid
        model_selected = utility.determine_model_checkpoint()
        valid_flag = model_selected != ""

        # Check if process was valid
        if valid_flag:
            # Generate infomation from testing database of DST files
            valid_flag = scrape_data_from(constants.PREDICTION_INPUT, constants.INPUT_CSV, constants.IMAGES, training=True)
            print(utility.outcome_to_message(valid_flag, constants.SCRAPE))

        # Check if process was valid 
        if valid_flag:
            # Preprocess infomation
            valid_flag = data_preprocessing(constants.INPUT_CSV, constants.INPUT_PREPROCESSED_CSV, constants.ORGINAL_STATS_CSV)
            print(utility.outcome_to_message(valid_flag, constants.PROCESSING))

        # Check if process was valid 
        if valid_flag:
            # Predict using best model checkpoint
            valid_flag = model_prediction(model_selected, constants.INPUT_PREPROCESSED_CSV, constants.ORGINAL_STATS_CSV, constants.COMPARSION_TABLE_CSV)
            print(utility.outcome_to_message(valid_flag, constants.PREDICTING))

        # Check if process was valid 
        if valid_flag:
            # Display prediction
            valid_flag = display_prediction(constants.COMPARSION_TABLE_CSV, constants.COMPARSION_GRAPH)
            print(utility.outcome_to_message(valid_flag, constants.DISPLAY))
            
# Output system shut down
print(f"\n...Ending  {project_name}")

# Exit system
sys.exit()