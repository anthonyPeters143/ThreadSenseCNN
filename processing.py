"""
processing.py
Handles dataset cleaning, normalization, IQR outlier removal, image padding, stats calculation, and preprocessing pipeline.

Dependencies:
- constants.py

Required Imports:
- pandas
- numpy
- PIL
- pathlib.Path
"""
####################################################################################################
# PROCESSING
####################################################################################################

# Will determines and removes data outliers from the passed dataframe using the IQR method
# Purpose: Removes outliers from the dataframe using the Interquartile Range (IQR) method.
# Inputs: dataframe_input (pd.DataFrame): DataFrame with data to be cleaned.
# Output: pd.DataFrame: Cleaned DataFrame with outliers removed.
def iqr_data_cleaning(dataframe_input):
    # Import library
    import constants
    
    # Summarize dataframe preprocessing
    print("before IQR - ")
    print(dataframe_input.describe())

    # Determine and store features for IQR range calculation
    iqr_features = [features for features in dataframe_input.columns if features != constants.IMAGE_PATH]

    # Determine quartiles 1, 2, and IQR
    q1 = dataframe_input[iqr_features].quantile(0.25)
    q3 = dataframe_input[iqr_features].quantile(0.75)
    iqr = q3 - q1

    # Creates mask of dataframe for non outliers using the IQR method, to remove points outside of the range
    non_outlier_mask = ~(((dataframe_input[iqr_features] < (q1 - 1.5 * iqr)) | (dataframe_input[iqr_features] > (q3 + 1.5 * iqr))).any(axis=1))

    # Apply mask on dataframe
    updated_dataframe = dataframe_input[non_outlier_mask].copy()

    # Summarize dataframe after preprocessing
    print("after IQR - ")
    print(updated_dataframe.describe())

    # Return the IQR cleaned data by using the non outlier mask
    return updated_dataframe

# Will pad and save images at the image path passed to the passed target size
# Purpose: Pads an image to the target size, maintaining aspect ratio, and saves it.
# Inputs: image_path (str): Path to the image. target_size (int): Desired size for both dimensions.
# Output: bool: True if successful, False otherwise.
def pad_image_to_size(image_path, target_size):
    # Import libaries
    from PIL import Image, ImageOps

    # Initilize valid flag to false
    valid_flag = False

    try:
        # Open image
        image = Image.open(image_path).convert('RGB')
        
        # Keeps aspect ratio of orginal image
        image.thumbnail((target_size, target_size))

        # Calculate padding needed for image
        padding_width = target_size - image.size[0]
        padding_height = target_size - image.size[1]
        
        # Pad and fill image with white
        padded_image = ImageOps.expand(image, 
                        (padding_width // 2, 
                        padding_height // 2, 
                        padding_width - padding_width // 2, 
                        padding_height - padding_height // 2),
                        fill="white")
        
        # Save updated image
        padded_image.save(image_path)

        # Set valid flag to true
        valid_flag = True

    except Exception as e:
        # Output exception message
        print(f"Invalid image padding {image_path}")

    # Return valid flag
    return valid_flag

# Checks if passed image path is within the max dimenision, will return true if image dimenision is within max dimenision
# Purpose: Checks if image dimensions are within the specified maximum.
# Inputs: image_path (str): Path to the image. max_dimenision (int): Maximum allowed width/height.
# Output: bool: True if in range, False otherwise.
def check_image_in_range(image_path, max_dimenision):
    # Import libaries
    from PIL import Image

    # Initilize valid flag to false
    valid_flag = False

    try:
        # Open each image and set its size
        width, height = Image.open(image_path).size

        # Check if width and height are within the max dimenision
        if width <= max_dimenision and height <= max_dimenision:
            # Set valid flag to true
            valid_flag = True

    except Exception as e:
        # Image not able to opened
        # Output exception message
        print(e)

    # Return valid flag value
    return valid_flag

# Will clean and returm passed dataframe based on dataframes's images paths image, checking if the images are within passed max diemension and padding them to it if needed, else remove images
# Purpose: Pads or removes images in a DataFrame based on size, ensuring all images meet required dimensions.
# Inputs: dataframe_input (pd.DataFrame): DataFrame with image paths. target_size (int): Target size for images.
# Output: pd.DataFrame: DataFrame with invalid images removed.
def image_cleaning(dataframe_input, target_size):
    # Initlize remove list
    remove_list = []

    # Initlizee valid exection flag to False
    valid_flag = False

    # Pre image cleaning summary
    print("Pre image cleaning - ")
    print(dataframe_input.describe())

    # Loop through dataframe image paths
    for image_path in dataframe_input["image path"]:
        # Check if image is within max dimenision range
        if check_image_in_range(image_path, target_size):
           # Set output progess
            loop_progress_message = image_path

            # Within range
            # Pad image to max max_dimension and store valid exection flag
            valid_flag = pad_image_to_size(image_path, target_size)

            # Check if exectuion is not valid
            if not valid_flag:
                # Output invalid exection message
                print("Invalid, quiting exection")

                # break for loop
                break
        
        else:
            # Set output progess
            loop_progress_message = image_path + " - Not valid"

            # Outside range 
            # Add image to remove set
            remove_list.append(image_path)

        # Output loop progess messsage
        print(loop_progress_message)

    # Check valid exection flag
    if valid_flag:
        # Remove elements using the remove list
        dataframe_input = dataframe_input[~dataframe_input["image path"].isin(remove_list)]

    # Post image clening summary
    print("Post image cleaning - ")
    print(dataframe_input.describe())

    # Return cleaned dataframe
    return dataframe_input

# Will normialzes and return dataframe using Z-score method, will add width, height, and thread count normailzed values columns
# Purpose: Normalizes width, height, and thread count using Z-score based on training stats.
# Inputs: dataframe_input (pd.DataFrame): DataFrame to normalize. training_stats_dataframe (pd.DataFrame): Stats for normalization.
# Output: pd.DataFrame: Normalized DataFrame.
def data_normalization(dataframe_input, training_stats_dataframe):
    # Import constants
    import constants
    
    # dataframe_input = dataframe_input.copy()

    # Z-score normalization of width, height, and thread count of designs using the orginal training data's stats
    dataframe_input[constants.WIDTH_NORM] = (dataframe_input[constants.WIDTH] - training_stats_dataframe[constants.WIDTH_MEAN].iloc[0]) / training_stats_dataframe[constants.WIDTH_STD].iloc[0]
    dataframe_input[constants.HEIGHT_NORM] = (dataframe_input[constants.HEIGHT] - training_stats_dataframe[constants.HEIGHT_MEAN].iloc[0]) / training_stats_dataframe[constants.HEIGHT_STD].iloc[0]
    dataframe_input[constants.THREAD_COUNT_NORM] = (dataframe_input[constants.THREAD_COUNT] - training_stats_dataframe[constants.THREAD_COUNT_MEAN].iloc[0]) / training_stats_dataframe[constants.THREAD_COUNT_STD].iloc[0]

    # Return normailized dataframe
    return dataframe_input

# Will calcuate and store stats for normilzing data (mean and std for thread count, width, and height) at passed CSV training stats from passed dataframe input, will return flag based on valid execution
# Purpose: Calculates and stores normalization statistics (mean, std) for thread count, width, and height.
# Inputs: dataframe_input (pd.DataFrame): Data to analyze. csv_normalizing_stats (str): Output path for stats CSV.
# Output: bool: True if stats were calculated and saved, False otherwise.
def calculate_data_stats(dataframe_input, csv_normalizing_stats):
    # Import libaries
    import pandas as pd

    # Import constants
    import constants

    try:
        # Calculate and store stats as a CSV file stored at passed training stats path
        stats = {
            constants.THREAD_COUNT_MEAN: dataframe_input[constants.THREAD_COUNT].mean(),
            constants.THREAD_COUNT_STD: dataframe_input[constants.THREAD_COUNT].std(),
            constants.WIDTH_MEAN: dataframe_input[constants.WIDTH].mean(),
            constants.WIDTH_STD: dataframe_input[constants.WIDTH].std(),
            constants.HEIGHT_MEAN: dataframe_input[constants.HEIGHT].mean(),
            constants.HEIGHT_STD: dataframe_input[constants.HEIGHT].std(),
        }
        pd.DataFrame([stats]).to_csv(csv_normalizing_stats, index=False)

        # Set exection valid flag to true
        execution_valid_flag = True
    
    except Exception as e:
        # Set exection valid flag to False
        execution_valid_flag = False

        # Print exception 
        print(e)

    # Return exectuion valid flag
    return execution_valid_flag

# Will create a comparsion dataframe using the passed predictions, input values, and the orginal training stats. Will create comparison dataframe with a image path, thread count actual and predicted and a absoulte value error of the difference, then will return the dataframe
# Purpose: Creates a comparison DataFrame of actual vs predicted thread counts and calculates absolute errors.
# Inputs: predictions (array-like), csv_input_file (str), csv_orginal_values (str)
# Output: pd.DataFrame: DataFrame with image path, actual, predicted, and error columns.
def create_comparison_dataframe(predictions, csv_input_file, csv_orginal_values):
        # Import libaries
        import pandas as pd
        import numpy as np
        
        import constants

        # Create dataframe from input csv~
        input_dataframe = pd.read_csv(csv_input_file)

        # Load orginal stats from passed orginal values csv path
        orginal_values = pd.read_csv(csv_orginal_values)

        # Flatten priections to an array
        flatten_predictions = np.array(predictions).flatten()

        # Denormalize using orginal stats 
        denormalize_predictions = (flatten_predictions * float(orginal_values[constants.THREAD_COUNT_STD].iloc[0])) + float(orginal_values[constants.THREAD_COUNT_MEAN].iloc[0])

        # Create comparison dataframe
        comparison = pd.DataFrame({
            constants.IMAGE_PATH: input_dataframe[constants.IMAGE_PATH],
            constants.THREAD_COUNT_ACTUAL: input_dataframe[constants.THREAD_COUNT],
            constants.THREAD_COUNT_PREDICTED: denormalize_predictions,
        })

        # Calculate error metrics, using absoulte value of the thread counter subtracted from the predicted amount
        comparison[constants.THREAD_COUNT_ABS_ERROR] = abs(comparison[constants.THREAD_COUNT_ACTUAL] - comparison[constants.THREAD_COUNT_PREDICTED])

        # Return comparison dataframe
        return comparison

# Will preprocess CSV inputs into CSV output by cleaning the data using IQR, resizing, and normializing, if a training mode flag is set to true will stored training stats for preprocessing at the passed training stats
# Purpose: Full preprocessing pipeline: IQR cleaning, image validation, stats calculation, normalization, CSV export.
# Inputs: csv_input (str), csv_output (str), csv_normalizing_stats (str), training (bool)
# Output: bool: True if preprocessing succeeded, False otherwise.
def data_preprocessing(csv_input, csv_output, csv_normalizing_stats, training=False):
    # Import libaries
    import pandas as pd
    from pathlib import Path

    # Import constants
    import constants

    # Iniilize exection valid flag to False
    execution_valid_flag = False

    try:
        # Convert CSV file into panda dataframe
        dataframe = pd.read_csv(csv_input)

        # Cleaning data using IQR method to remove outliers
        dataframe = iqr_data_cleaning(dataframe)

        # Resize and remove images outside of max dimenision
        dataframe = image_cleaning(dataframe, constants.MAX_DIMENSION)

        # Check if in training mode and created a valid data stats CSV file or is not in training mode and the referenced data stats is a CSV file 
        if (training and calculate_data_stats(dataframe, csv_normalizing_stats)) or (not training and (Path(csv_normalizing_stats).is_file() and Path(csv_normalizing_stats).suffix == ".csv")):
            # Convert orginal training stats CSV into dataframe
            training_stats_dataframe = pd.read_csv(csv_normalizing_stats)

            # Normalize data using Z-score
            dataframe = data_normalization(dataframe, training_stats_dataframe)

            # Save preprocessed dataframe into csv file
            dataframe.to_csv(csv_output)

            # Set exection valid flag to true
            execution_valid_flag = True

    except Exception as e:
        # Set exection valid flag to False
        execution_valid_flag = False

        # Print exception 
        print(e)

    finally:
        # Return exectuion valid flag
        return execution_valid_flag