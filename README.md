# ThreadSenseCNN

ThreadSenseCNN is a complete machine learning pipeline designed to estimate thread count for embroidery design files (DST) from images of the desgin (PNG). It uses a hybrid Convolutional Neural Network (CNN) that combines image analysis with design metadata (width and height), trained on over 100,000 real-world embroidery files.

This project demonstrates scalable ML workflows for visual feature extraction, large dataset preprocessing, and production-ready predictions.

## Features

- Full CLI-based pipeline (scraping → processing → training → prediction)
- Handles DST embroidery files and common image formats
- Extracts thread count, width, height metadata
- Converts DST files to PNG with formatting adjustments
- Cleans data using IQR and image dimension validation
- Normalizes values using Z-score
- Hybrid CNN: image and metadata as dual inputs
- Model saving, checkpointing, and metric logging
- Visual comparison of predicted vs actual results

## Pipeline Overview

1. Scrape Input Data  
    - Converts embroidery DST/images → PNG  
    - Extracts thread count, width, height  
    - Saves to CSV and image folder  

2. Preprocess Dataset  
    - Removes outliers via IQR  
    - Pads/validates images to 1024×1024  
    - Normalizes data with Z-score  
    - Outputs a preprocessed CSV  

3. Train CNN Model  
    - Uses hybrid CNN (image + metadata)  
    - Applies data augmentation  
    - Logs metrics to CSV  
    - Saves best model checkpoint  

4. Predict and Display  
    - Uses saved model to predict on new data  
    - Outputs comparison graph and MAE  

## CLI Usage

To run the pipeline, open a terminal and execute:

```bash
python menu.py
```

You will see this menu:

```
0 - Exit system
1 - Scrape data
2 - Process data
3 - Train using data
4 - Full data pipeline (1→3)
5 - Predict using data
```

### Menu Breakdown

| Option | Description |
|--------|-------------|
| 0 | Exit the system |
| 1 | Scrapes embroidery files, converts to PNG, extracts thread count / width / height, and saves images and `input_csv.csv` |
| 2 | Removes outliers, resizes/crops images, normalizes metadata using saved stats, and writes to `input_csv_preprocessed.csv` |
| 3 | Trains the CNN using the preprocessed CSV. Saves best model to `model_checkpoints/` and logs metrics to `training_logs/` |
| 4 | Runs steps 1–3 as a full pipeline |
| 5 | Prompts for a saved model, scrapes and processes new data, runs predictions, and displays a bar chart and MAE |

## Directory Structure

```
.
├── constants.py               # Central config (file paths, labels, etc.)
├── displaying.py              # Training history plots and prediction visualizations
├── menu.py                    # CLI entry point
├── models.py                  # CNN model architecture, training logic, prediction
├── processing.py              # Data normalization, cleaning, IQR outlier removal
├── scraping.py                # DST/image scraping, metadata extraction, conversion
├── utility.py                 # Helper functions (naming, prompts, selection)

├── model_checkpoints/         # Stores best model checkpoints (*.keras)
├── training_logs/             # CSV logs from training
├── input/                     # Folder for raw DST/image files
├── input_images/              # Converted PNGs for training and prediction
├── prediction_input/          # Input directory when running predictions
```

## Model Architecture

ThreadSenseCNN uses a hybrid model architecture:

- Image Input (CNN): Extracts visual patterns from 1024×1024 PNGs
- Metadata Input (Dense): Processes width and height values
- Concatenation Layer: Merges visual and dimensional signals
- Output: Predicts a single thread count value (regression)

Training includes:

- Mean squared error loss and mean absolute error metric
- Early stopping based on validation loss
- Learning rate decay on plateau
- Best model saved using checkpointing

## Dataset

- Input formats: PNG
- Initial dataset: Over 107,000 DST files
- Outliers removed using the IQR method
- Extracted fields: thread count, width (cm), height (cm)
- Normalized using Z-score based on training statistics

## Requirements

- Python 3.8+
- TensorFlow
- pandas
- matplotlib
- pyembroidery
- cairosvg
- Pillow

Install with:

```bash
pip install -r requirements.txt
```

## Running the Full Pipeline

If you're starting with a folder of raw embroidery files:

```bash
python menu.py
# Select option 4 - Full pipeline
```

If you already have a trained model and want to run predictions:

```bash
python menu.py
# Select option 5 - Predict using data
```

## License

MIT License — free to use, modify, and integrate.

## Author

Anthony Peters - Software Engineer  
This project was created to demonstrate my ability to design and implement scalable machine learning pipelines, apply deep learning to unstructured visual and numeric data, and work with large-scale real-world datasets across multiple stages of preprocessing, model training, and evaluation.
