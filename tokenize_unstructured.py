import nltk
import re
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize  import word_tokenize, sent_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# set of stop words to remove (noise words)
stop_words = set(stopwords.words('english'))


def tokenize_data(response):
#     response = word_tokenize(response)
    if isinstance(response, str): 

        # tokenizes all of the data to be just the core words (reduces noise and unnecessary characters)

        response_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9 ]', '', response))
        response_tokens = [x.lower() for x in response_tokens if x.lower() not in stop_words and x.lower() != 'i' and x.lower() != '’']
        response_tokens = " ".join(response_tokens)


        return response_tokens



def tf_idf_calc(df):
    
    tfidf_vectorizer = TfidfVectorizer()  # For TF-IDF

    # Vectorize each column
    for column in df.columns:

        # fix na values to be lowercase
        df[column] = df[column].fillna("")

        # Use TF-IDF
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])
        print(f"TF-IDF Matrix for {column}:\n", tfidf_matrix.toarray())

        df[column] = tfidf_matrix.toarray().tolist()

    return df


def clean_unstructured_data(dataframe):
    #Clean specified columns in a DataFrame.

    columns_to_clean = [
    "Why scope",
    "What skills",
    "Gain",
    "Anything else",
    "Major",
    "Minor",
    "How did you hear"
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


    dataframe = tf_idf_calc(dataframe)

    #columns with valid binary values 
    # binary_cols = [
    #     col for col in columns_to_clean
    #     if col in dataframe.columns and dataframe[col].dropna().isin([0, 1]).all()
    # ]
    
    # #keep only the binary-valid columns
    # dataframe = dataframe[binary_cols]

    # #drop rows with missing values

    return dataframe




