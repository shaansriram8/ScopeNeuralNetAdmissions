import os
import pandas as pd

def combine_tokenized_binary(output1, output2, combined_output):
    # Load the two CSV files into DataFrames
    df1 = pd.read_csv(output1)
    df2 = pd.read_csv(output2)
    
    # Check if the number of rows is the same in both DataFrames
    if df1.shape[0] != df2.shape[0]:
        raise ValueError("The two CSV files must have the same number of rows to combine horizontally.")
    
    # Combine the DataFrames horizontally
    combined_data = pd.concat([df1, df2], axis=1)
    
    # Save the combined DataFrame to a new CSV file
    combined_data.to_csv(combined_output, index=False)
    
    print(f"Combined CSV saved as: {combined_output}")
