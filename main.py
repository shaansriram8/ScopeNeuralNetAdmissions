from combine import combine_csv_to_csv
from load_data import load_data

if __name__ == "__main__":
    folder_path = './csv_files'  
    output_file = 'combined_data.csv'  

    combine_csv_to_csv(folder_path, output_file)
    load_data(output_file)

