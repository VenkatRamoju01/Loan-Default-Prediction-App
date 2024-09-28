# Loan Default Prediction App

This Streamlit application predicts the likelihood of a loan applicant defaulting on their loan based on various financial and demographic factors. The model leverages historical loan data to provide insights that can assist lenders in making informed decisions.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Machine Learning Model](#machine-learning-model)
- [Model Construction and Prediction](#model-construction-and-prediction)

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

The app utilizes a **Random Forest Classifier**, an ensemble method that builds multiple decision trees and merges them to obtain a more accurate and stable prediction. 

### Data Preprocessing

Before constructing the model, the data is preprocessed to ensure it is clean and relevant. The dataset is sampled, features are selected, and categorical columns are one-hot encoded.

## Model Construction and Prediction

After data preprocessing, the following steps were taken to construct the model and make predictions:

1. **Train-Test Split**: The dataset was divided into training and testing sets using an 80-20 split to ensure that the model is evaluated on unseen data.

2. **Feature Scaling**: The features were standardized using a pre-trained scaler (saved as `scaler.pkl`) to ensure that all input features are on the same scale, improving model performance.

3. **Handling Class Imbalance**: To address class imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training set. This technique generates synthetic examples to balance the distribution of classes, ensuring that the model learns from an equal number of positive and negative instances.

4. **Model Training**: The Random Forest model was trained on the preprocessed training data. The trained model was then saved using Pickle to allow for quick loading and reuse in the application.

5. **Running the Model**: When the user inputs applicant details into the Streamlit app and clicks "Check Loan Eligibility":
   - The input data is transformed and scaled using the previously saved scaler.
   - The model is loaded from the saved `random_forest_model_pickle.pkl` file.
   - The model predicts the risk of loan default based on the scaled input data.
   
6. **Output Results**: The prediction result is displayed to the user, indicating whether the applicant is likely to default on the loan or if they should proceed with the loan sanctioning.

### Notes
- **The model is trained on historical loan data from the USA. It aims to convert Indian salaries and loan amounts to USD based on Purchasing Power Parity (PPP) to provide more accurate predictions. However, users should consider the context and limitations of the model's training data when making lending decisions.**
