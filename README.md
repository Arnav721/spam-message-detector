# Spam Message Detection Using Machine Learning

## Overview

This project classifies SMS messages as spam or not spam using Natural Language Processing (NLP) and machine learning. It also uses simple rule-based checks to identify potential phishing indicators.

The application provides an interactive interface built with Streamlit, allowing users to enter a message and view the detection results.

## Objectives

- Process text messages into numerical features.
- Train a machine learning model for spam classification.
- Evaluate the model using unseen test data.
- Identify potential phishing indicators using rule-based checks.
- Provide an interactive interface for message analysis.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Matplotlib
- Joblib
- Streamlit
- Regular Expressions (Regex)

## Dataset

This project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

Dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The training script downloads the dataset automatically if it is not already available locally. An internet connection is required for the initial download.

## Model Workflow

1. Download and load the SMS Spam Collection dataset.
2. Clean the data by removing missing values and duplicate messages.
3. Split the dataset into training and testing sets.
4. Convert text messages into numerical features using TF-IDF.
5. Train a Logistic Regression classifier.
6. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Save the trained model for use in the application.
8. Use Streamlit to provide an interactive message detection interface.
9. Apply rule-based checks to identify potential phishing indicators.

## Features

- Classifies messages as spam or not spam using machine learning.
- Displays the model prediction and its confidence score.
- Checks for selected suspicious phrases, links, and prize-related language.
- Combines machine learning predictions with rule-based warnings.
- Generates a confusion matrix for model evaluation.
- Provides safety guidance when potential phishing indicators are detected.

## Project Structure

    spam-message-detector/
    ├── app.py
    ├── train_model.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    └── confusion_matrix.png

The `data/` and `models/` directories are generated locally when the training script runs and are excluded from version control.

## Setup and Installation

### 1. Download the Repository

Download the project from GitHub and open its folder in VS Code or a terminal.

### 2. Create a Virtual Environment

    python -m venv .venv

### 3. Activate the Virtual Environment

Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

If PowerShell blocks activation, use Command Prompt instead:

    .venv\Scripts\activate.bat

### 4. Install Dependencies

    pip install -r requirements.txt

### 5. Train the Model

    python train_model.py

This downloads the dataset if necessary, trains the classifier, prints evaluation metrics, generates the confusion matrix, and saves the trained model.

An internet connection is required for the initial dataset download.

### 6. Run the Application

    streamlit run app.py

Open the local URL displayed in the terminal to use the application.

Important: Train the model before launching the application because the saved model is not included in the repository.

## Evaluation

The training script evaluates the classifier on a held-out test set and reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The confusion matrix is saved as `confusion_matrix.png`.

Actual evaluation results depend on the dataset and model configuration. Run `train_model.py` to obtain the current results.

## Limitations

- The model may incorrectly classify legitimate or spam messages.
- Unseen words, slang, new scam patterns, and other languages may affect performance.
- Rule-based checks may produce false positives or miss suspicious messages.
- The model confidence score does not guarantee a correct prediction.
- A message classified as not spam is not necessarily safe.

Users should verify suspicious messages through official channels and avoid sharing sensitive information through unexpected links.

## Future Improvements

- Expand the dataset with additional message examples.
- Improve phishing detection rules.
- Compare different machine learning algorithms.
- Improve the user interface and evaluation results display.
- Test the system on additional datasets.

## Author

Name: Arnav Bhardwaj

Course: B.Tech in Artificial Intelligence and Machine Learning

University: COER University, Roorkee

## Disclaimer

This project is developed for educational purposes. It demonstrates machine learning-based text classification and basic phishing indicator detection. It should not be treated as a complete cybersecurity solution.