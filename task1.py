import pandas as pd # type: ignore
import numpy as np # type: ignore

def clean_data(file_path):
    df = pd.read_csv(file_path)

    print("Original Data:\n", df)

    df = df.drop_duplicates()

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['salary'] = pd.to_numeric(df['salary'], errors='coerce')

    df['age'].fillna(df['age'].median(), inplace=True)
    df['salary'].fillna(df['salary'].median(), inplace=True)


    df['name'].fillna(df['name'].mode()[0], inplace=True)
    df['gender'].fillna(df['gender'].mode()[0], inplace=True)

    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    
    df['name'] = df['name'].str.strip().str.title()
    df['gender'] = df['gender'].str.upper()

    
    df.reset_index(drop=True, inplace=True)

    print("\nCleaned Data:\n", df)

    return df


cleaned_df = clean_data("data.csv")