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

def combine_csv_files(folder_path):
    """Combine all CSV files in a folder into a single DataFrame."""
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

    #combine all DataFrames into one
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data

def clean_combined_data(dataframe, columns_to_clean):
    """Clean specified columns in a DataFrame."""
    #preprocess all da columns
    for col in columns_to_clean:
        if col in dataframe.columns:
            #clean to binary values (1 -> 'y', 0 -> 'n')
            dataframe[col] = dataframe[col].apply(clean_yes_no_to_binary)

    #columns with valid binary values 
    binary_cols = [
        col for col in columns_to_clean
        if col in dataframe.columns and dataframe[col].dropna().isin([0, 1]).all()
    ]
    
    #keep only the binary-valid columns
    dataframe = dataframe[binary_cols]

    #drop rows with missing values
    dataframe = dataframe.dropna()
    return dataframe


   


