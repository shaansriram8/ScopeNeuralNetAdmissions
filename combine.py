import pandas as pd
import os

folder_path = './csv_files'  # Replace with the path to your folder
output_file = 'combined_data.csv'    # Name of the output CSV file

# Initialize an empty list to hold dataframes
dataframes = []

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):  # Ensure the file is an Excel file
        file_path = os.path.join(folder_path, file_name)
        # Read the Excel file into a dataframe
        df = pd.read_excel(file_path)
        dataframes.append(df)

# Combine all dataframes into one
combined_data = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a CSV file
combined_data.to_csv(output_file, index=False)