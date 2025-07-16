"""Group project"""
import seaborn as sns

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


print(KSI_dataset.isnull().sum())  # Check for missing values in each column



#Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# --- Data Preparation for Temporal Plots ---
# Convert 'DATE' to datetime objects
KSI_dataset['DATE'] = pd.to_datetime(KSI_dataset['DATE'])

# Extract Year, Month, Day of Week, and Hour for easier plotting
KSI_dataset['YEAR'] = KSI_dataset['DATE'].dt.year
KSI_dataset['MONTH'] = KSI_dataset['DATE'].dt.month_name()
KSI_dataset['DAY_OF_WEEK'] = KSI_dataset['DATE'].dt.day_name()
# Convert integer time (e.g., 1450) to just the hour (e.g., 14)
KSI_dataset['HOUR'] = KSI_dataset['TIME'] // 100


# --- Graph 1: Accident Severity Breakdown ---
plt.figure(figsize=(8, 6))
sns.countplot(data=KSI_dataset, x='ACCLASS', palette='viridis')
plt.title('Distribution of Accident Severity')
plt.xlabel('Accident Class')
plt.ylabel('Number of Persons Involved')
plt.savefig(r'graphs/1_countplot_acclass.png')
plt.close()

# --- Graph 2: Collisions by Light Condition ---
plt.figure(figsize=(10, 6))
sns.countplot(data=KSI_dataset, y='LIGHT', order=KSI_dataset['LIGHT'].value_counts().index, palette='magma')
plt.title('Number of Collisions by Light Condition')
plt.xlabel('Number of Persons Involved')
plt.ylabel('Light Condition')
plt.tight_layout()
plt.savefig(r'graphs/2_countplot_light.png')
plt.close()

# --- Graph 3: Collisions by Impact Type ---
plt.figure(figsize=(12, 8))
sns.countplot(data=KSI_dataset, y='IMPACTYPE', order=KSI_dataset['IMPACTYPE'].value_counts().index, palette='plasma')
plt.title('Number of Collisions by Impact Type')
plt.xlabel('Number of Persons Involved')
plt.ylabel('Impact Type')
plt.tight_layout()
plt.savefig(r'graphs/3_countplot_impactype.png')
plt.close()

# --- Graph 4: Severity by Road Class (Percentage) ---
crosstab_roadclass = pd.crosstab(KSI_dataset['ROAD_CLASS'], KSI_dataset['ACCLASS'], normalize='index') * 100
crosstab_roadclass.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='coolwarm')
plt.title('Proportion of Accident Severity by Road Class')
plt.xlabel('Road Class')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(title='Accident Class')
plt.tight_layout()
plt.savefig(r'graphs/4_stacked_severity_by_roadclass.png')
plt.close()

# --- Graph 5: Severity by Contributing Factors (Alcohol) ---
plt.figure(figsize=(10, 7))
sns.countplot(data=KSI_dataset[KSI_dataset['ALCOHOL'] == 'Yes'], x='ALCOHOL', hue='ACCLASS', palette='Reds_r')
plt.title('Accident Severity in Alcohol-Related Collisions')
plt.xlabel('Alcohol Involved')
plt.ylabel('Number of Persons Involved')
plt.legend(title='Accident Class')
plt.tight_layout()
plt.savefig(r'graphs/5_grouped_severity_alcohol.png')
plt.close()

# --- Graph 6: Severity vs. Aggressive/Distracted Driving ---
plt.figure(figsize=(10, 7))
sns.countplot(data=KSI_dataset[KSI_dataset['AG_DRIV'] == 'Yes'], x='AG_DRIV', hue='ACCLASS', palette='Oranges_r')
plt.title('Accident Severity in Aggressive/Distracted Driving Collisions')
plt.xlabel('Aggressive/Distracted Driving Involved')
plt.ylabel('Number of Persons Involved')
plt.legend(title='Accident Class')
plt.tight_layout()
plt.savefig(r'graphs/6_grouped_severity_ag_driv.png')
plt.close()


# --- Graph 7: Collisions Over the Years ---
plt.figure(figsize=(14, 7))
sns.countplot(data=KSI_dataset, x='YEAR', hue='ACCLASS', palette='viridis')
plt.title('KSI Collisions by Year')
plt.xlabel('Year')
plt.ylabel('Number of Persons Involved')
plt.xticks(rotation=45)
plt.legend(title='Accident Class')
plt.tight_layout()
plt.savefig(r'graphs/7_line_collisions_by_year.png')
plt.close()

# --- Graph 8: Collisions by Hour of Day ---
plt.figure(figsize=(14, 7))
sns.countplot(data=KSI_dataset, x='HOUR', palette='twilight_shifted')
plt.title('Number of Collisions by Hour of Day')
plt.xlabel('Hour of Day (24-hour format)')
plt.ylabel('Number of Persons Involved')
plt.tight_layout()
plt.savefig(r'graphs/8_bar_collisions_by_hour.png')
plt.close()

# --- Graph 9: Collisions by Day of Week ---
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(12, 6))
sns.countplot(data=KSI_dataset, x='DAY_OF_WEEK', order=days_order, palette='cubehelix')
plt.title('Number of Collisions by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Persons Involved')
plt.tight_layout()
plt.savefig(r'graphs/9_bar_collisions_by_day.png')
plt.close()

# --- Graph 10: Collision Hotspot Map ---
plt.figure(figsize=(10, 10))
sns.kdeplot(
    data=KSI_dataset,
    x='LONGITUDE',
    y='LATITUDE',
    fill=True,
    cmap='Reds',
    alpha=0.6,
    levels=20 # Increase levels for more detail
)
plt.title('Collision Hotspot Density in Toronto')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig(r'graphs/10_kdeplot_hotspots.png')
plt.close()