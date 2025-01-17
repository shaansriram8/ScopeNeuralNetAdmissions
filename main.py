from combine import combine_csv_files, clean_combined_data
from unused.load_data import load_data
from tokenize_unstructured import clean_unstructured_data
from mastercsv import combine_tokenized_binary

if __name__ == "__main__":
    folder_path = './csv_files'  
    output_file1 = 'binary_csv.csv'
    output_file2 = 'tokenized_csv.csv'  

    columns_to_clean = [
        "Timestamp",
        "On campus",
        "Conflicts",
        "Commit hours",
        "Applied before"
    ]
    

    combined_data = combine_csv_files(folder_path)

    

    #yes/no to binary: anything below this is for shaan's script

    binary_csv = combined_data.copy()

    cleaned_data = clean_combined_data(binary_csv, columns_to_clean)
    
    cleaned_data.to_csv(output_file1, index=False)

    #load_data(output_file1)
    #------------------------------------------------------------


    # tokenizing unstructured data: Srushti

    unstructured_csv = combined_data.copy()

    tokenized_data = clean_unstructured_data(unstructured_csv)

    tokenized_data.to_csv(output_file2, index=False)

    #------------------------------------------------------------

    # combining tokenized and binary data: Shaan

    combined_output = 'master_csv.csv'

    combine_tokenized_binary(output_file1, output_file2, combined_output)


    
