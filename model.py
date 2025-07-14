"""Group project"""

'''
1. Data exploration: a complete review and analysis of the dataset including:

Load and describe data elements (columns), provide descriptions & types, ranges and values of elements as appropriate. – use pandas, numpy and any other python packages.
Statistical assessments including means, averages, correlations
Missing data evaluations – use pandas, numpy and any other python packages
Graphs and visualizations – use pandas, matplotlib, seaborn, numpy and any other python packages, you also can use power BI desktop.


2. Data modelling:

Data transformations – includes handling missing data, categorical data management, data normalization and standardizations as needed.
Feature selection – use pandas and sci-kit learn. (The group needs to justify each feature used and any data columns discarded)
Train, Test data splitting – use numpy, sci-kit learn.
Managing imbalanced classes if needed. Check here for info: https://elitedatascience.com/imbalanced-classes
Use pipelines class to streamline all the pre-processing transformations.
'''


# 1. Data exploration

import os
import urllib.request
import tarfile
import pandas as pd

# creating dataframe from the csv file
def load_data():
    filename = r"ksi-dataset.csv"

    if  os.path.exists(filename):
        df = pd.read_csv(filename)
        return df

KSI_dataset = load_data()


print(KSI_dataset.shape)  # Display the shape of the dataset

print(KSI_dataset.columns)  # Display the column names


# Display the first few rows of the dataset
def display_column_details(df):
    for col in df.columns:
        print(f"\nColumn: {col}")
        print("Data type:", df[col].dtype)
        print("First 5 values:\n", df[col].head())
        print("Missing values:", df[col].isnull().sum())
        if pd.api.types.is_numeric_dtype(df[col]):
            print("Statistical summary:\n", df[col].describe())
            print("\n\n\n")
        else:
            print("Value counts:\n", df[col].value_counts().head())
            print("\n\n\n")

display_column_details(KSI_dataset)




