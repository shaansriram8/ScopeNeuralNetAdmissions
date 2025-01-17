import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv_horizontally(input_csv, training_csv, testing_csv, test_size=0.2, random_state=None):
    """
    Splits a CSV file into training and testing CSVs horizontally with the same column names.
    
    Parameters:
    - input_csv: Path to the input CSV file to split.
    - training_csv: Path to save the training CSV file.
    - testing_csv: Path to save the testing CSV file.
    - test_size: Proportion of the data to allocate to the testing set (default 0.2).
    - random_state: Random state for reproducibility (default None).
    """
    # Load the combined CSV file into a DataFrame
    data = pd.read_csv(input_csv)
    
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )
    
    # Save the training and testing DataFrames to CSV files
    train_data.to_csv(training_csv, index=False)
    test_data.to_csv(testing_csv, index=False)
    

