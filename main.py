from combine import combine_csv_files, clean_combined_data
from load_data import load_data

if __name__ == "__main__":
    folder_path = './csv_files'  
    output_file1 = 'binary_csv.csv'  

    columns_to_clean = [
        "Have you ever applied to Scope before? If so, which semester(s)?",
        "Are you willing to commit at least 4 hours a week to the club (including general meetings and weekly project team meetings)?",
        "Do you have any conflicts on Tuesdays from 6-7 pm?",
        "Do you plan to be on campus this semester?"
    ]

    combined_data = combine_csv_files(folder_path)

    #yes/no to binary: anything below this is for shaan's script

    binary_csv = combined_data.copy()

    cleaned_data = clean_combined_data(binary_csv, columns_to_clean)
    
    cleaned_data.to_csv(output_file1, index=False)

    load_data(output_file1)
    #------------------------------------------------------------
