


## Overview
This Python script implements a machine learning pipeline to classify gamma-ray data using several classification algorithms, including K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression, Support Vector Machines (SVM), and a Neural Network.

The dataset used is the MAGIC Gamma Telescope dataset (`magic04.data`), which distinguishes between gamma rays (signal) and hadrons (background). Preprocessing techniques and oversampling are applied to handle class imbalance and improve model performance.



## Requirements
The following libraries are required to run this script:

- Python 3.x
- NumPy
- pandas
- Matplotlib
- scikit-learn
- imbalanced-learn
- TensorFlow

Install the required libraries using:
```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow
```


## Features
1. **Data Preprocessing**
   - Reads the dataset and assigns column names.
   - Converts the target class into binary format: gamma (1) and hadron (0).
   - Splits the data into training, validation, and test sets.
   - Scales the features and applies oversampling to the training set.

2. **Data Visualization**
   - Plots histograms for feature distributions, separated by class.

3. **Classification Models**
   - **K-Nearest Neighbors (KNN):** Trains and evaluates a KNN classifier.
   - **Naive Bayes:** Implements Gaussian Naive Bayes for classification.
   - **Logistic Regression:** Trains a logistic regression model.
   - **Support Vector Machine (SVM):** Fits an SVM with default hyperparameters.
   - **Neural Network:** Trains a neural network with varying hyperparameters, including nodes, dropout rates, learning rates, and batch sizes. Selects the model with the least validation loss.

4. **Performance Metrics**
   - Uses `classification_report` from scikit-learn to evaluate models on the test set.


## Usage
1. Place the dataset file (`magic04.data`) in the same directory as the script.
2. Run the script:
   ```bash
   python script_name.py
   ```
3. Outputs include feature histograms, model performance metrics, and visualizations of loss and accuracy for neural networks.



## Customization
- **Hyperparameter Tuning:** Modify the range of nodes, dropout probabilities, learning rates, batch sizes, and epochs in the neural network training loop.
- **Model Selection:** Comment or uncomment specific model training sections to include/exclude them.
- **Dataset:** Replace `magic04.data` with a different dataset by updating file paths and column names.



## Limitations
- The script assumes binary classification. Adjustments are needed for multiclass datasets.
- Neural network training may require GPU acceleration for large datasets or extensive hyperparameter tuning.
