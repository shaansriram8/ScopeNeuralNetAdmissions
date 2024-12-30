import os
import pandas as pd

def combine_csv_to_csv(folder_path, output_file):
    """
    Combines all CSV files in a folder into a single CSV file.

    Parameters:
    - folder_path (str): Path to the folder containing CSV files.
    - output_file (str): Name of the output CSV file.
    """
    dataframes = []

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

    combined_data.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
