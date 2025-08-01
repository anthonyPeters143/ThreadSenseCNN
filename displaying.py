"""
displaying.py
Visualizes model training history and prediction comparisons.

Dependencies:
- Requires 'constants.py' for column labels and file paths
- Requires trained model history and CSV results for prediction comparison

Required Imports:
- matplotlib
- pandas
- numpy
- os
"""

# Purpose: Visualizes model training and validation loss and mean absolute error across epochs.
# Inputs:
#   - history: Keras History object from model.fit(), containing loss/mae metrics.
# Output:
#   - None (shows plots using matplotlib)
def plot_training_history(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE', marker='o')
    plt.plot(history.history['val_mae'], label='Val MAE', marker='o')
    plt.title('Mean Absolute Error Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Purpose: Visualizes prediction results by comparing actual and predicted thread counts, including error bars and input images.
# Inputs:
#   - comparsion_table_path (str): Path to CSV file containing actual/predicted counts and image paths.
#   - comparison_graph_path (str): Output path for saving the generated PNG graph.
# Output:
#   - valid_flag (bool): True if the process completes successfully, otherwise False.
def display_prediction(comparsion_table_path, comparison_graph_path):
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import constants

    valid_flag = False

    try:
        comparsion_dataframe = pd.read_csv(comparsion_table_path)
        image_names = comparsion_dataframe[constants.IMAGE_PATH]
        actual = comparsion_dataframe[constants.THREAD_COUNT_ACTUAL]
        predicted = comparsion_dataframe[constants.THREAD_COUNT_PREDICTED]
        abs_error = comparsion_dataframe[constants.THREAD_COUNT_ABS_ERROR]

        x = range(len(image_names))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual and predicted bars
        ax.bar([i - bar_width/2 for i in x], actual, width=bar_width, label='Actual', color='skyblue')
        ax.bar([i + bar_width/2 for i in x], predicted, width=bar_width, label='Predicted', color='salmon')

        # Add bar labels and lines to show difference
        for i, (a, p, e) in enumerate(zip(actual, predicted, abs_error)):
            ax.text(i - bar_width / 2, a + max(actual) * 0.02, f'{int(a)}', ha='center', va='bottom', fontsize=9)
            ax.text(i + bar_width / 2, p + max(predicted) * 0.02, f'{int(p)}', ha='center', va='bottom', fontsize=9)
            ax.plot([i - bar_width / 2, i + bar_width / 2], [a, p], 'k--', linewidth=1)
            ax.text(i, (a + p) / 2 + max(actual) * 0.02, f'{e:.0f}', ha='center', va='bottom', fontsize=10, color='red')

        ax.set_ylabel('Thread Count')
        ax.set_title('Actual vs Predicted Thread Count')
        ax.set_xticks(x)
        ax.set_xticklabels([os.path.basename(name) for name in image_names], rotation=45, ha='right')
        ax.legend()
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.3)

        # Draw images under the bars
        for i, image_path in enumerate(image_names):
            try:
                img = mpimg.imread(image_path)
                imagebox = ax.inset_axes([i / len(image_names) + 0.08, -0.26, 0.08, 0.15])
                imagebox.imshow(img)
                imagebox.axis('off')
            except FileNotFoundError:
                print(f"Image not found: {image_path}")

        plt.tight_layout()
        plt.savefig(comparison_graph_path, dpi=300, bbox_inches='tight')
        plt.show()

        mae_thread = np.mean(comparsion_dataframe[constants.THREAD_COUNT_ABS_ERROR])
        execution_message = f"Mean Absolute Error - Thread count: {mae_thread:.2f}"
        valid_flag = True

    except Exception as e:
        execution_message = f"Error displaying prediction:\n{e}"

    print(execution_message)
    return valid_flag