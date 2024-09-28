import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import time
import warnings
from sklearn.metrics import (accuracy_score, 
                             f1_score, 
                             precision_score, 
                             recall_score, 
                             RocCurveDisplay,
                             precision_recall_curve, 
                             average_precision_score,
                             roc_auc_score, 
                             roc_curve, auc)
warnings.filterwarnings('ignore')


df = pd.read_csv('Loan_default.csv')


# Step 1: Data Preprocessing and Feature Selection

# Separate features (X) and target variable (y)
df_subset = df.sample(n=3000, random_state=42).copy()
X=df_subset.drop(['Default','LoanID'],axis=1)
y=df_subset['Default']

# One-hot encode categorical columns
X_encoded= X[['Age', 'Income', 'LoanAmount', 'CreditScore', 
                     'MonthsEmployed', 'NumCreditLines', 
                     'InterestRate', 'LoanTerm', 'DTIRatio']]

# Split data into train and test sets
X_train,X_test,y_train,y_test= train_test_split(X_encoded,y,stratify=y,test_size=0.2,random_state=42)

# Load your previously saved scaler model
scaler_model = pickle.load(open('scaler.pkl', 'rb'))

# Transform both training and test data
X_train_scaled = scaler_model.transform(X_train)

# Transform test data
X_test_scaled = scaler_model.transform(X_test)

# Filter selected features in train and test datasets
X_train_selected = X_train_scaled
X_test_selected = X_test_scaled

# Apply SMOTE to balance the classes on the training set only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

# Load your trained model (you can save it using pickle)
model = pickle.load(open('random_forest_model_pickle.pkl', 'rb'))

# Streamlit App
st.title("Loan Risk Assessment")

st.write("""Please enter the applicant's details to check if they are likely to default on their loan
""")

# Get user inputs
age = st.number_input("Age [range:(18,100)]", min_value=18, max_value=100)
income = st.number_input("Yearly income-INR (numeric only, eg: 2450000)]", min_value=0)
loan_amount = st.number_input("Requested loan amount-INR (numeric only, eg:3000000)", min_value=0)
credit_score = st.number_input("Credit score [range:(300,850)]", min_value=300, max_value=850)
months_employed = st.number_input("Number of months employed at your current job", min_value=0)
num_credit_lines = st.number_input("Number of open credit lines (no. credit cards + loans etc.)", min_value=0)
interest_rate = st.number_input("Loan interest rate in % [range:(0,30), eg:11.50] ", min_value=0.01, max_value=30.00, format="%.2f")
loan_term = st.number_input("Loan term in months [range:(12,360)]", min_value=12, max_value=360)

current_debt = st.number_input("Current monthly EMI-INR (numeric only, eg:45000)", min_value=0)
current_income = st.number_input("Current monthly income-INR (numeric only, eg:145000)")

# Check for division by zero before calculating DTI ratio
if current_income > 0:
    dti_ratio = current_debt / current_income
else:
    dti_ratio = 0

# Predict loan eligibility
if st.button("Check Loan Eligibility"):
    input_data = [[
        age,            # Age
        income*0.04547,         # Yearly Income
        loan_amount,    # Loan Amount
        credit_score,   # Credit Score
        months_employed, # Months Employed
        num_credit_lines, # Number of Credit Lines
        interest_rate,  # Interest Rate
        loan_term,      # Loan Term
        dti_ratio       # DTI Ratio
    ]]
    
    selected_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 
                         'MonthsEmployed', 'NumCreditLines', 
                         'InterestRate', 'LoanTerm', 'DTIRatio']
    
    # Convert the input into a DataFrame
    input_df = pd.DataFrame(input_data, columns=selected_features)

    # Apply scaling to input data
    input_scaled = scaler_model.transform(input_df)

    # Predict the risk using the trained model
    risk = model.predict(input_scaled)

    # Output the result
    if risk[0] == 0:
        st.write("#### Please proceed with sanctioning the loan")
        st.markdown("<small><i>The data used for training the model is based in the USA, so the results may not be accurate.</i></small>", unsafe_allow_html=True)
    else:
        st.write("#### Hey, the applicant is likely to default. Please do not sanction the loan")
        st.markdown("<small><i>Note: The data used for training the model is based in the USA, so the results may not be accurate.</i></small>", unsafe_allow_html=True)
