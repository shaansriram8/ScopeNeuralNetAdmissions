import os
import pandas as pd

def clean_yes_no_to_binary(value):
    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower.startswith('y'): 
            return 1
        else:  
            return 0
    return 0  



def combine_and_preprocess(folder_path, output_file):
    #these are the specific column names to preprocess 
    columns_to_clean = [
        "Have you ever applied to Scope before? If so, which semester(s)?",
        "Are you willing to commit at least 4 hours a week to the club (including general meetings and weekly project team meetings)?",
        "Do you have any conflicts on Tuesdays from 6-7 pm?",
        "Do you plan to be on campus this semester?"
    ]
    
    dataframes = []

    #read and combine CSV files
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_path)
                dataframes.append(df)
                print(f"Processed: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    combined_data = pd.concat(dataframes, ignore_index=True)

    #preprocess cols
    for col in columns_to_clean:
        if col in combined_data.columns:
            #clean to binary values (1 for 'y', 0 for 'n')
            combined_data[col] = combined_data[col].apply(clean_yes_no_to_binary)

    #removes rows without 0 or 1
    binary_cols = [
        col for col in columns_to_clean
        if col in combined_data.columns and combined_data[col].dropna().isin([0, 1]).all()
    ]
    combined_data = combined_data[binary_cols]

    #drop rows taht are missing vals
    combined_data = combined_data.dropna()

    combined_data.to_csv(output_file, index=False)
