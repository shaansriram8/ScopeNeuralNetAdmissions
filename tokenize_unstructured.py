import nltk
import string
import re
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize  import word_tokenize, sent_tokenize

# set of stop words to remove (noise words)
stop_words = set(stopwords.words('english'))


def tokenize_data(response):
#     response = word_tokenize(response)
    if isinstance(response, str): 

        # response.translate(str.maketrans("", "", string.punctuation))

        response_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9 ]', '', response))
        response_tokens = [x.lower() for x in response_tokens if x.lower() not in stop_words and x.lower() != 'i' and x.lower() != '’']
        return response_tokens



def clean_unstructured_data(dataframe):
    """Clean specified columns in a DataFrame."""

    columns_to_clean = [
    "Why are you interested in joining SCOPE Consulting? (100 - 200 words)",
    "What skills, expertise and knowledge will you bring to SCOPE Consulting and your team?",
    "What do you hope to gain from this experience?",
    "Is there anything else you would like SCOPE board to know?"
    ]

    # set dataframe to only be the relavent columns
    dataframe = dataframe[columns_to_clean]

    # tokenize each response in every column
    for col in columns_to_clean:
    #     if col in dataframe.columns:


            # print(dataframe[col])

            dataframe[col] = dataframe[col].apply(tokenize_data)

            # print(dataframe[col])
            # print("--------------------------------------------------------")

    #columns with valid binary values 
    # binary_cols = [
    #     col for col in columns_to_clean
    #     if col in dataframe.columns and dataframe[col].dropna().isin([0, 1]).all()
    # ]
    
    # #keep only the binary-valid columns
    # dataframe = dataframe[binary_cols]

    # #drop rows with missing values
    return dataframe