# Loan Default Prediction App

This Streamlit application predicts the likelihood of a loan applicant defaulting on their loan based on various financial and demographic factors. The model leverages historical loan data to provide insights that can assist lenders in making informed decisions.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Machine Learning Model](#machine-learning-model)
- [How It Works](#how-it-works)
- [Data Preprocessing](#data-preprocessing)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Files Included](#files-included)
- [Contributing](#contributing)
- [License](#license)

## Overview

The primary goal of this app is to assess the risk associated with loan applications. Users can input applicant details, and the model will predict whether the applicant is likely to default. This can aid financial institutions in streamlining their lending processes.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Imbalanced-learn
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Pickle

## Machine Learning Model

The app utilizes a **Random Forest Classifier**, which is an ensemble method that builds multiple decision trees and merges them to obtain a more accurate and stable prediction. 

### Model Training Process
1. **Data Collection**: A sample of loan data is read from a CSV file.
2. **Data Preprocessing**: The features and target variable are separated, and categorical columns are one-hot encoded.
3. **Train-Test Split**: The dataset is divided into training and testing sets using an 80-20 split.
4. **Scaling**: The features are standardized using a pre-trained scaler to ensure they are on the same scale.
5. **Handling Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the classes in the training set.
6. **Model Training**: The model is trained on the processed training data.

## How It Works

1. The user inputs applicant details such as age, income, loan amount, credit score, and employment history.
2. The app calculates the Debt-to-Income (DTI) ratio to assess the applicant's financial health.
3. Upon clicking the "Check Loan Eligibility" button, the app processes the input data and uses the trained Random Forest model to predict the risk of loan default.
4. The result is displayed to the user, indicating whether to proceed with or deny the loan application.

## Data Preprocessing

The dataset is preprocessed to ensure the model is trained on clean and relevant data:
- A subset of 3000 records is randomly sampled from the original dataset.
- Features such as age, income, loan amount, and credit score are selected for model training.
- Categorical features are transformed using one-hot encoding to convert them into numerical representations.

## Installation and Setup

To run this application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/loan-default-prediction.git
   cd loan-default-prediction
