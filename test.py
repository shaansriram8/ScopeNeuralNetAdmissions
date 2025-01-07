import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample DataFrame
data = {'text': ['This is a sample', 'Another example text', 'Yet another sample']}
df = pd.DataFrame(data)

# Step 1: Compute the TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['text'])  # Compute TF-IDF

# Step 2: Convert TF-IDF rows into lists
tfidf_vectors = tfidf_matrix.toarray().tolist()  # Each row is now a list

# Step 3: Add TF-IDF vectors as a new column
df['tfidf'] = tfidf_vectors

# Optional: If you need the vectors as numpy arrays instead of lists
# import numpy as np
# df['tfidf'] = [np.array(v) for v in tfidf_vectors]

print(df)
