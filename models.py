"""
models.py
Defines and manages the CNN model architecture, training routines, dataset preparation, and prediction logic.

Dependencies:
- constants.py
- utility.py
- processing.py

Required Imports:
- tensorflow
- pandas
- numpy
"""
import constants
import utility

import tensorflow as tf

# Will randomly process and return the passed image to cause cosmetic differences but keeping the thread count, width, and height uneffected
# Purpose: Apply random augmentations to an image tensor for data augmentation during training.
# Inputs: image (tf.Tensor): Image tensor to augment.
# Output: tf.Tensor: Augmented image tensor.
def augment_image(image):
    image = tf.image.random_flip_left_right(image)            # horizontal flip
    image = tf.image.random_flip_up_down(image)               # vertical flip
    image = tf.image.random_brightness(image, max_delta=0.1)  # random brightness
    image = tf.image.random_contrast(image, 0.9, 1.1)         # random contrast
    image = tf.image.random_saturation(image, 0.9, 1.1)       # random saturation
    image = tf.image.random_hue(image, 0.05)                  # small hue shift

    # Return images
    return image

# Will return fresh callback methods for model training 
# Purpose: Create callback methods for model training (checkpointing, early stopping, logging, learning rate scheduling).
# Inputs: model_training_name (str): Name for identifying checkpoints/logs.
# Output: list: List of TensorFlow Keras callbacks.
def get_callback_methods(model_training_name):
    # Import libaries
    import datetime
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

    # from constants import generate_checkpoint_name_from_training_name

    # Will check and save the best checkpoint model based loss valee stat and display the stats during training
    checkpoint_callback = ModelCheckpoint(
            filepath = utility.generate_checkpoint_name_from_training_name(model_training_name),   
                                                                # Where callback will be stored
            save_best_only = True,                              # Set to only update callback if better stats
            monitor = "val_loss",                               # What stats to monitor
            mode = "min",                                       # How to validate stats monitored
            verbose = 1                                         # Output operations
        )

    # Will stop the training does improve its loss value stat doesn't improve within 5 epochs
    early_stop_callback = EarlyStopping(
            monitor = "val_loss",                               # What stats to monitor
            patience = 5,                                       # How many epochs to wait for improvment before stopping
            restore_best_weights=True                           # Reverts best model weights at end of an epoch
        )

    # Will adjust the learning rate dynamically during training
    reduce_learning_rate = ReduceLROnPlateau(
        monitor='val_loss',                                     # What stats to monitor
        factor=0.5,                                             # Reduce the learning rate by factor when it's plateauing
        patience=5,                                             # How many epochs to wait for no improvement before reducing LR
        min_lr=1e-6,                                            # Minimum learning rate
        verbose=1                                               # Output learning rate change info
    )

    # Will log and save the training data created during the training in a file inlcuding training name, date, and time
    csv_logger_callback = CSVLogger(constants.TRAINING_LOGS + f"training_log_{utility.generate_checkpoint_name_from_training_name(model_training_name)}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", append=True)

    # Will return checkpoint, early stop, and csv loging, dynamic learning rate methods
    return [checkpoint_callback, early_stop_callback, csv_logger_callback, reduce_learning_rate]

# Will load images into ram and preprocess them into tensor to be fed into the model
# Purpose: Load image from path, normalize, optionally augment, and combine with metadata for model input.
# Inputs: image_path (str), metadata (np.ndarray or tf.Tensor), label (optional, numeric), training (bool).
# Output: tuple: Tensor(s) ready for input into the model.
def load_and_preprocess(image_path, metadata, label = None, training=False):
    # Load and process image
    image = tf.io.read_file(image_path)             # Opens file and stores as an png image 
    image = tf.image.decode_png(image, channels=3)  # Converts the png image into TensorFlow tensor
    image = tf.cast(image, tf.float32) / 255.0      # Converts into Tensorflow tensor values into float in the range of 0-1

    # Cast meta data as float32
    metadata = tf.cast(metadata, tf.float32)

    # Pads or crops image to max dimension
    image = tf.image.resize_with_crop_or_pad(image, target_height=constants.MAX_DIMENSION, target_width=constants.MAX_DIMENSION)

    # If training will augment image 
    if training:
        image = augment_image(image)

    # Pads or crops image to max dimension after augmenatation
    image = tf.image.resize_with_crop_or_pad(image, target_height=constants.MAX_DIMENSION, target_width=constants.MAX_DIMENSION)
    image.set_shape([constants.MAX_DIMENSION, constants.MAX_DIMENSION, 3])

    # Set return tensor based on if in training mode
    if training:
        return_tensor = (image, metadata), label
    
    else:
        return_tensor = ((image, metadata),)

    # Return tensor
    return return_tensor

# Will process and return prediction datasets from passed csv file into dataframes 
# Purpose: Prepare a TensorFlow dataset for model prediction from a CSV file.
# Inputs: csv_file (str): Path to preprocessed CSV file.
# Output: tf.data.Dataset: Batched dataset for prediction.
def process_predicting_dataset(csv_file):
    # Import libaries
    import pandas as pd
    # import numpy as np

    # Convert preprocessed csv file into dataframe
    prediction_dataframe = pd.read_csv(csv_file)

    # Create arrays from dataframe rows
    predict_image_paths = prediction_dataframe[constants.IMAGE_PATH].values
    predict_meta = prediction_dataframe[[constants.WIDTH_NORM, constants.HEIGHT_NORM]].values

    # # Dummy data to fill in for mapping later
    # predict_labels = np.zeros(len(predict_image_paths), dtype=np.float32)

    # Create datasets from image path and metadata
    predict_dataset = tf.data.Dataset.from_tensor_slices((predict_image_paths, predict_meta))

    # Map datasets keeping only image and meta and set parallel calls using autotuning
    predict_dataset = predict_dataset.map(
        lambda predict_image_paths, predict_meta: load_and_preprocess(predict_image_paths, predict_meta, training=False), num_parallel_calls=tf.data.AUTOTUNE).batch(8)

    # Return dataset
    return predict_dataset

# Will process and return training and test datasets from passed csv file into dataframes 
# Purpose: Split data into train/test, load images/metadata, and return TensorFlow datasets for training.
# Inputs: csv_file (str): Path to preprocessed CSV file.
# Output: tuple: (train_dataset, test_dataset) as TensorFlow datasets.
def process_training_dataset(csv_file):
    # Import libaries
    import pandas as pd

    # Convert preprocessed csv file into dataframe
    dataframe = pd.read_csv(csv_file)

    # Split dataframe for training and testing
    train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.2, random_state=42)

    # Create arrays from dataframe rows
    train_image_paths = train_dataframe[constants.IMAGE_PATH].values
    train_meta = train_dataframe[[constants.WIDTH_NORM, constants.HEIGHT_NORM]].values
    train_labes = train_dataframe[constants.THREAD_COUNT_NORM].values

    test_image_paths = test_dataframe[constants.IMAGE_PATH].values
    test_meta = test_dataframe[[constants.WIDTH_NORM, constants.HEIGHT_NORM]].values
    test_labels = test_dataframe[constants.THREAD_COUNT_NORM].values

    # Create datasets from arrays
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_meta, train_labes))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_meta, test_labels))

    # Map datasets and set parallel calls using autotuning
    train_dataset = train_dataset.map(
        lambda image_path, metadata, label: load_and_preprocess(image_path, metadata, label, training=True), num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(
        lambda image_path, metadata, label: load_and_preprocess(image_path, metadata, label, training=False), num_parallel_calls=tf.data.AUTOTUNE)

    # Randomly organizes datasets in batchs using autotuning
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(8).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

    # Return datasets
    return train_dataset, test_dataset

# Will create a CNN MLM and return it, v-1
# Purpose: Build and compile a hybrid CNN model (image + metadata) version 1.
# Inputs: None.
# Output: tf.keras.Model: Compiled Keras model.
def create_model_v1():
    # Import libaries
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Concatenate, GlobalAveragePooling2D
    from tensorflow.keras.models import Model, load_model

    # Image Input Branch
    image_input = Input(shape=(1024, 1024, 3), name='image_input')
    x = Conv2D(32, (3,3), activation='relu')(image_input)
    x = MaxPooling2D(2)(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)

    # Metadata Input Branch
    meta_input = Input(shape=(2,), name='meta_input')  # width and height
    y = Dense(32, activation='relu')(meta_input)

    # Combine Branches
    combined = Concatenate()([x, y])
    z = Dense(64, activation='relu')(combined)
    output = Dense(1)(z)  # no activation for regression

    # Create model
    model = Model(inputs=[image_input, meta_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Return model
    return model

# Will create a CNN MLM and return it, v-2
# Running notes of model is overfitting 
# Purpose: Build and compile an improved hybrid CNN model (image + metadata) version 2 with dropout.
# Inputs: None.
# Output: tf.keras.Model: Compiled Keras model.
def create_model_v2():
    # Import libaries
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Concatenate, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam


    # Image Input Branch
    image_input = Input(shape=(1024, 1024, 3), name='image_input')  # Image input,                          [batch_size, 1024, 1024, 3]
    x = Conv2D(32, (3,3), activation='relu')(image_input)           # Low-level pattern detection,          [batch_size, 1022, 1022, 32]
    x = MaxPooling2D(2)(x)                                          # Downsampling,                         [batch_size, 511, 511, 32]
    x = Conv2D(64, (3,3), activation='relu')(x)                     # Mid-level feature detection,          [batch_size, 509, 509, 64]
    x = MaxPooling2D(2)(x)                                          # Downsampling,                         [batch_size, 254, 254, 64]
    x = Conv2D(128, (3,3), activation='relu')(x)                    # Higher-level feature detection,       [batch_size, 252, 252, 128]
    x = MaxPooling2D(2)(x)                                          # Downsampling,                         [batch_size, 126, 126, 128]
    x = Conv2D(256, (3,3), activation='relu')(x)                    # Advanced design feature extraction,   [batch_size, 124, 124, 256]
    x = MaxPooling2D(2)(x)                                          # Downsampling,                         [batch_size, 62, 62, 256]
    x = Conv2D(512, (3,3), activation='relu')(x)                    # Global pattern recognition,           [batch_size, 60, 60, 512]
    x = GlobalAveragePooling2D()(x)                                 # Feature summarization,                [batch_size, 512]

    # Metadata Input Branch
    meta_input = Input(shape=(2,), name='meta_input')               # Metadata input for width and height,  [batch_size, 2]
    y = Dense(32, activation='relu')(meta_input)                    # Transform metadata,                   [batch_size, 32]

    # Combine Branches
    combined = Concatenate()([x, y])                                # Combine image + metadata,             [batch_size, 544]
    z = Dense(64, activation='relu')(combined)                      # Merge and compress combined features,	[batch_size, 64]
    z = Dropout(0.3)(z)                                             # Prevent overfitting,          	    [batch_size, 64]
    output = Dense(1)(z)                                            # Thread count prediction (regression),	[batch_size, 1]

    # Create model using input and out tensors
    model = Model(inputs=[image_input, meta_input], outputs=output) 

    # Complie mode using the Adam optizier with a learning rate of 1e-3, loss fucntion of mean squared error, and eveualting using mean absolute error
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])

    # Return model
    return model

# Will train CNN MLM model using passed preprocessed data, will prompt user for a training name that will be used for bookkeeping, will callback methods and evaluate the MSE and MAE using the test part of the dataset split, will store the history of the model at the passed path
# Purpose: Train the CNN model on preprocessed data, logging metrics and saving best checkpoint.
# Inputs: preprocessed_csv_file (str), model (tf.keras.Model), model_history (str).
# Output: bool: True if training was successful, False otherwise.
def train_model(preprocessed_csv_file, model, model_history):
    # Import libaries
    import pandas as pd
    from tensorflow.keras.models import load_model

    # Initialize valid flag to false
    valid_flag = False

    # Prompt user to name model training
    model_training_name = utility.prompt_user_name_training()

    # Check if user inputed name is not an empty string
    if model_training_name != "":
        try:
            # Create training and testing datasets from passed csv file
            train_dataset, test_dataset = process_training_dataset(preprocessed_csv_file)

            # Train model
            history = model.fit(
                train_dataset,                                              # Data that model will train on
                validation_data = test_dataset,                             # Data that model will validate on
                epochs = 30,                                                # Epoch size for processing
                callbacks = get_callback_methods(model_training_name),      # Callbacks to activate while training
                verbose = 1                                                 # Sets live progress bar during training
                )
            
            # Load best model from checkpoints
            best_model = load_model(utility.generate_checkpoint_name_from_training_name(model_training_name))

            # # Create graph showing Loss on left and MAE on right
            # plot_training_history(history)

            # Store history as a CSV at passed history path
            pd.DataFrame(history.history).to_csv(model_history, index=False)

            # Evaluate model using test dataset
            loss, mae = best_model.evaluate(test_dataset)

            # Set output message using model evaluation
            output_message = f"{model_training_name} trained - Test MSE Loss: {loss:.4f}\nTest MAE: {mae:.4f}"

            # Set valid flag to true
            valid_flag = True

        except Exception as e:
            # Error during training, set output message to error
            output_message = f"Training error,\n{e}"

    # Training name genreation was invalid
    else:
        # Set output message to invalid inputs
        output_message = "User input invalid training name, training aborted"

    # Output process message
    print(output_message)

    # Return valid flag
    return valid_flag

# Will predict and produce a comparsion table to be stored as the passed comparsion table path, using the passed model, prediciton CSV file, and the orginal values for denormailzing from the input CSV file
# Purpose: Use trained model to predict on new data, produce a comparison table for evaluation.
# Inputs: model_path (str), csv_input_file (str), csv_orginal_values (str), comparsion_table_csv (str).
# Output: bool: True if prediction was successful, False otherwise.
def model_prediction(model_path, csv_input_file, csv_orginal_values, comparsion_table_csv):
    # Import libaries
    # import pandas as pd
    # import numpy as np
    from tensorflow.keras.models import load_model

    from processing import create_comparison_dataframe

    # Initialize valid flag to false
    valid_flag = False

    try:
        # Load model from passed model path
        model = load_model(model_path)

        # Create prediction dataset from passed CSV file
        prediciton_dataset = process_predicting_dataset(csv_input_file)

        # Predict and return thread count from prediciton dataset
        predictions = model.predict(prediciton_dataset)

        # Create comparison dataframe from predictions, orginal inputs, and the training data stats
        comparsion_table = create_comparison_dataframe(predictions, csv_input_file, csv_orginal_values)

        # Store comaprsion tabel as a CSV file
        comparsion_table.to_csv(comparsion_table_csv)

        # Set valid flag to true
        valid_flag = True

    except Exception as e:
        # Error during prediction, set output message to error
        print(f"Training error,\n{e}")

    # Return valid flag
    return valid_flag