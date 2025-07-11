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

print(KSI_dataset.isnull().sum())  # Check for null column names


# Display the first few rows of the dataset
def display_data(df):
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nData types and descriptions:")
    print(df.info())
    print("\nStatistical summary:")
    print(df.describe())

display_data(df=KSI_dataset)

# Historical data visualization for every column in the dataset

import matplotlib.pyplot as plt

def plot_historical_data(df):
    df.hist(bins=50, figsize=(20, 15))
    plt.show()
    plt.savefig(r'Graphs\ksi-histogram.png')  # Save the histogram plot as an image file

plot_historical_data(df=KSI_dataset)


