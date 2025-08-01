"""
utility.py
General-purpose helper functions for path management, prompts, file selection, and result formatting.

Dependencies:
- constants.py

Required Imports:
- os
"""
# Will take model training name and return the model path
# Purpose: Generate the full model checkpoint path from a given training name.
# Inputs: model_training_name (str): Name used for training/model.
# Output: str: Full path to checkpoint directory.
def generate_checkpoint_name_from_training_name(model_training_name):
    # Import constants
    import constants

    return constants.MODEL_CHCEKPOINTS + "\\" + model_training_name + "\\" + constants.BEST_MODEL

# Will take passed flag value and section and return outcome message
# Purpose: Format a standardized message for success/failure based on a flag.
# Inputs: flag_value (bool): Operation result. section_name (str): Section description.
# Output: str: Formatted message.
def outcome_to_message(flag_value, section_name):
    return f"{"Successful" if flag_value else "Failed"} - {section_name}\n"

# Prompt user to input a name for model training, input will be formatted and the user can check and reinput, but inputs can't be an empty string and will be asked to reprompt if it is, if input fails the returned string will be an empty string
# Purpose: Prompt user to enter a valid model training name, formatted for filenames.
# Inputs: None (uses input())
# Output: str: Training name, or '' if input failed.
def prompt_user_name_training():
    # Iniltze input valid flag to false
    input_valid = False

    try:
        # Loop while input check is invalid
        while not input_valid:
            # Prompt user to set training name, then remove outside whitespace, set characters to lowerreplace, and replace internal space with underscore characters
            training_name = input("Input name for training = ").strip().lower().replace(" ", "_")

            # Check if input is empty
            if training_name == "":
                # Output name can't be an empty string
                print("Training name can't be an empty string, please enter another input")
            
            else:
                # Check if user validates the training name with formatting
                user_input = input(f"{training_name} - Input Y/y = set, N/n = reinput").strip().lower()
                if user_input == "y":
                    # Set input flag to true
                    input_valid = True
    except:
        # Invalid name input
         training_name = ""

    # Return training name
    return training_name

# Will output a list of model checkpoints from the model_checkpoints directory and prompt the user to select a file wihtin it, will return an empty string if there are no files within directory else will cycle prompting the user to select a file index then return the path to selected file
# Purpose: List available model checkpoints and prompt user to select one.
# Inputs: None (uses input() and lists directory)
# Output: str: Selected model checkpoint path, or '' if none found.
def determine_model_checkpoint():
    # Import libaries
    import os, constants

    # Initlize input valid input
    input_valid = False

    # Set the allowed file extensions
    extensions = (".h5", ".keras")

    # List all files within the model checkpoint directory that are files and end with a valid extension
    files = [file for file in os.listdir(constants.MODEL_CHCEKPOINTS) 
             if os.path.isfile(os.path.join(constants.MODEL_CHCEKPOINTS, file)) and file.endswith(extensions)]
    
    # Check if there are no files
    if not files:
        # Set the file path to emptry sense no files where found
        model_path = ""

    # There are files
    else:
        # Display files to user
        for index, file in enumerate(files):
            print(f"{index} - {file}")

        # Loop till user makes valid input
        while not input_valid:
            try:
                # Prompt user for file choice 
                user_input = int(input("Enter the index of the file you want to select: "))

                # Check if file choice is between 0 and the total of files
                if 0 <= user_input < len(files):
                    # Store files path of selected model
                    model_path = os.path.join(constants.MODEL_CHCEKPOINTS, files[user_input])

                    # Set input valid flag to true
                    input_valid = True

                else:
                    # Input is invalid
                    print("Inputed selection is invalid")

            except ValueError:
                # File choice input is not an integer
                print("Input is not an integer please reenter a valid integer")

    # Return model path
    return model_path