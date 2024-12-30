from combine import combine_csv_to_csv  # Import the function from your script

if __name__ == "__main__":
    # Define the folder containing the CSV files and the output file name
    folder_path = './csv_files'  # Replace with the path to your folder
    output_file = 'combined_data.csv'  # Desired output CSV file name

    # Call the function to combine CSV files
    combine_csv_to_csv(folder_path, output_file)
