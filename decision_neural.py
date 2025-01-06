import tensorflow
from tensorflow.python import keras
from tensorflow.python.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Embedding, LSTM, Dense, Dropout

#Loading in new applicant data
tstfile = pd.read_csv(input("Enter file path of applicant data"))

X_struc = tstfile.iloc[ : , :-4]
X_unstr = tstfile.iloc[: , -3:-1]
Y_val = tstfile.iloc[: , -1]

model_unstr = keras.Sequential([

    #Embedding takes in a vector of numbers based on the tokenized sentence. It returns a complex array for each word
    layers.Embedding(input_dim = 5000, output_dim = 128, input_length =100) ,# Shrusti - input dim and input length (diff vocab, and max sequence length)
    
    #LSTM for learning common patterns between words, asking model for 64 patterns
    layers.LSTM(units = 64, return_sequences= False, dropout = 0.3), #dropout for overfitting issues

    layers.Dense(units = 64, activation = 'relu'),
    layers.Dropout(rate = 0.3),

    # OUTPUT LAYER
    layers.Dense(units = 1, activation = 'sigmoid')
])
model_struc = keras.Sequential([

    # HIDDEN LAYERS
    layers.Dense(units = 100, activation = 'relu', use_bias = True,input_shape = [50]), #input shape based on feature count quantitative
    layers.Dropout(rate = .3),
    layers.Dense(units = 100, activation = 'relu', use_bias = True),
    layers.Dropout(rate = .3),
    layers.Dense(units = 100, activation = 'relu', use_bias = True),

    # OUTPUT LAYER
    layers.Dense(units = 1, activation = 'sigmoid')
])

#Reload all weights
pred_struc = model_struc.load_weights('model_struc_weights.h5')
pred_unstr = model_unstr.load_weights('model_unstr_weights.h5')

#Get predictions
model_struc.predict(X_struc)
model_unstr.predict(X_unstr)

#Final predicions
final  = (pred_struc + pred_unstr)/2

#Applicant score must be over 85 percent if applicant should be admitted
final = [1 if x > .85 else 0 for x in final]


print("The following appl;icants are eligble for admission: \n")

for x,y in enumerate(final):
    if final[y] == 1: print(f"Applicant {x}")