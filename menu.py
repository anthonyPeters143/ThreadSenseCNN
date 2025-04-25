####################################################################################################
# IMPORTS
####################################################################################################

import constants
from scraping import scrape_data_from

####################################################################################################
# GLOBLE VARIABLES
####################################################################################################

# Initlize counter for files skipped during verfication
FILES_SKIPPED = 0
FILES_VALID = 0
FILES_TOTAL = 0

####################################################################################################
# MENU
####################################################################################################

# Menu loop, will repeat till exit code is entered
# Exit flag is initlized to false
exit_flag = False

# Output system start up
print("\nStarting thread_est_o_mater_v2.py ...")

while not exit_flag:
    menu_input = input("\t0 - exit\n\t1 - process data from database\n\t2 - preprocess data\n\t3 - train model\n\t4 - whole system\n\t5 - predict images\nEnter menu selection : ")

    if (menu_input == "0"):
        # Flip exit flag
        exit_flag = True

    elif (menu_input == "1"):
        # Generate infomation from database of dst files
        scrape_data_from(constants.TRAINING_CSV, constants.DATABASE_PATH, constants.TRAINING_IMAGES)

    elif (menu_input == "2"):
        # Preprocess infomation
        data_preprocessing(constants.TRAINING_CSV, constants.TRAINING_PREPROCESSED_CSV, constants.TRAINING_ORGINAL_VALUES_CSV)

    elif (menu_input == "3"):
        # Train model from generated infomation
        train_model(constants.TRAINING_PREPROCESSED_CSV, create_model_v2())

    elif (menu_input == "4"):
        # Run whole system
        # Generate infomation from database of dst files 
        print("process data from database")
        scrape_data_from(constants.TRAINING_CSV, constants.DATABASE_PATH, constants.TRAINING_IMAGES)

        # Preprocess infomation
        print("preprocess data")
        data_preprocessing(constants.TRAINING_CSV, constants.TRAINING_PREPROCESSED_CSV, constants.TRAINING_ORGINAL_VALUES_CSV)

        # Train model from generated infomation
        print("train model")
        train_model(constants.TRAINING_PREPROCESSED_CSV, create_model_v2())



    elif (menu_input == "5"):
        # Generate infomation from testing database of dst files
        print("process data from testing database")
        scrape_data_from(constants.TESTING_CSV, constants.TESTING_DATABASE_PATH, constants.TESTING_IMAGES)

        # Preprocess infomation
        print("preprocess data")
        data_preprocessing(constants.TESTING_CSV, constants.TESTING_PREPROCESSED_CSV, constants.TESTING_ORGINAL_VALUES_CSV)

        # Predict using best model using test infomation
        print("test model")
        model_prediction(constants.MODEL_CHCEKPOINTS + "\\" + constants.BEST_MODEL, constants.TESTING_PREPROCESSED_CSV, constants.TESTING_ORGINAL_VALUES_CSV, constants.COMPARSION_TABLE_CSV)

# Output system shut down
print("... Ending thread_est_o_mater_v2.py\n")

# Exit system
sys.exit()