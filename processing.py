####################################################################################################
# PROCESSING
####################################################################################################

# Will determines and removes data outliers from the passed dataframe using the IQR method
def iqr_data_cleaning(dataframe_input):
    print("IQR DATA CLEANING")

    # Summarize dataframe preprocessing
    print("before - ")
    print(dataframe_input.describe())

    # Determine and store features for IQR range calculation
    iqr_features = [features for features in dataframe_input.columns if features != "image path"]

    # Determine quartiles 1, 2, and IQR
    q1 = dataframe_input[iqr_features].quantile(0.25)
    q3 = dataframe_input[iqr_features].quantile(0.75)
    iqr = q3 - q1

    # Creates mask of dataframe for non outliers using the IQR
    non_outlier_mask = ~(((dataframe_input[iqr_features] < (q1 - 1.5 * iqr)) | (dataframe_input[iqr_features] > (q3 + 1.5 * iqr))).any(axis=1))

    # Summarize dataframe after preprocessing
    print("after - ")
    print(dataframe_input[non_outlier_mask].describe())

    # Return the IQR cleaned data by using the non outlier mask
    return dataframe_input[non_outlier_mask].copy()

# Will pad and save images at the image path passed to the passed target size
def pad_image_to_size(image_path, target_size):
    # Import libaries
    from PIL import Image, ImageOps

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

# Checks if passed image path is within the max dimenision, will return true if image dimenision is within max dimenision
def check_image_in_range(image_path, max_dimenision):
    # Import libaries
    from PIL import Image

    try:
        # Open each image and set its size
        width, height = Image.open(image_path).size

        # Check if width and height are within the max dimenision
        if width <= max_dimenision and height <= max_dimenision:
            # Set valid flag to true
            valid_flag = True
        
        else:
            # Set valid flag to false
            valid_flag + False

    except:
        # Image not able to opened
        # Set valid flag to false
        valid_flag = False

    # Return valid flag value
    return valid_flag

# Will clean and returm passed dataframe based on dataframes's images paths image, checking if the images are within passed max diemension and padding them to it if needed, else remove images
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
            loop_progress_message = image_path + " - Removed"

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
def data_normalization(dataframe_input):
    # Import constants
    import constants
    
    # dataframe_input = dataframe_input.copy()

    # Z-score normalization of width, height, and thread count of designs
    dataframe_input[constants.WIDTH_NORM] = (dataframe_input[constants.WIDTH] - dataframe_input[constants.WIDTH].mean()) / dataframe_input[constants.WIDTH].std()
    dataframe_input[constants.HEIGHT_NORM] = (dataframe_input[constants.HEIGHT] - dataframe_input[constants.HEIGHT].mean()) / dataframe_input[constants.HEIGHT].std()
    dataframe_input[constants.THREAD_COUNT_NORM] = (dataframe_input[constants.THREAD_COUNT] - dataframe_input[constants.THREAD_COUNT].mean()) / dataframe_input[constants.THREAD_COUNT].std()

    # Return normailized dataframe
    return dataframe_input

# Will calcuate and store stats for normilzing data (mean and std for thread count, width, and height) at passed CSV training stats from passed dataframe input, will return flag based on valid execution
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
    
    except:
        # Set exection valid flag to False
        execution_valid_flag = False

    # Return exectuion valid flag
    return execution_valid_flag

# Will preprocess CSV inputs into CSV output by cleaning the data using IQR, resizing, and normializing, if a training mode flag is set to true will stored training stats for preprocessing at the passed training stats
def data_preprocessing(csv_input, csv_output, csv_normalizing_stats, training=False):
    # Import libaries
    import pandas as pd
    from pathlib import Path

    # Import constants
    import constants

    # Iniilize exection valid flag to False
    execution_valid_flag = False

    try:
        # Convert csv file into panda dataframe
        dataframe = pd.read_csv(csv_input)

        # Cleaning data using IQR method to remove outliers
        dataframe = iqr_data_cleaning(dataframe)

        # Resize and remove images outside of max dimenision
        dataframe = image_cleaning(dataframe, constants.MAX_DIMENSION)

        # Check if in training mode and created a valid data stats CSV file or is not in training mode and the referenced data stats are a CSV file 
        if (training and calculate_data_stats(dataframe, csv_normalizing_stats)) or (not training and (Path(csv_normalizing_stats).is_file() and Path(csv_normalizing_stats).suffix == ".csv")):

            # Normalize data using Z-score
            dataframe = data_normalization(dataframe, csv_normalizing_stats)

            # Save preprocessed dataframe into csv file
            dataframe.to_csv(csv_output)

            # Set exection valid flag to true
            execution_valid_flag = True

    except:
        # Set exection valid flag to False
        execution_valid_flag = False

    finally:
        # Return exectuion valid flag
        return execution_valid_flag