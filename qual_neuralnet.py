import tensorflow
from tensorflow.python import keras
from tensorflow.python.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Embedding, LSTM, Dense, Dropout

#initialize data
trnfile = pd.read_csv('clean_train_data.csv') # update both with actual files
tstfile = pd.read_csv('clean_test_data.csv') #ONLY USED FOR FINAL TESTING

Xtrain, Xval= train_test_split(trnfile, test_size = 0.2, random_state = 42)

# -- training dataset --
X_train_struc = Xtrain.iloc[: , :-4] # yes/no
X_train_unstr = Xtrain.iloc[: , -3:-1] # essays
Y_train = Xtrain.iloc[:, -1] #accepted/denied

# -- testing dataset --
X_val_struc = Xval.iloc[: , :-4]
X_val_unstr = Xval.iloc[:, -3:-1]
Y_val = Xval.iloc[:, -1]


# -- Testing dataset (FINAL TESTING ONLY) --
X_test_struc = tstfile.iloc[: , :-4]
X_test_unstr = tstfile.iloc[: , -3:-1]
Y_test = tstfile.iloc[: , -1]


#TOKENIZATION OF WORDS
# -- replaced with Shrusti's work --


#Creating the model
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

# prevent overfitting/underfitting + save time
early_stopping = keras.callbacks.EarlyStopping(
    patience = 5, # will train for 5 epochs of consecutive unimprovement
    min_improv = .001, # mimimum improvement to be considered effective
    restore_best_weights = True # put back the best weights
)

#setting up the model by specifications
model_struc.compile(
    optimizer = 'adam', # chose adam over SGT (Stochastic Gradient Descent) since it has ability to adaptively change learning rates
    loss = 'binary_crossentropy', #using this loss for quantitative. It compares rate of yes to actual
    metrics = ['accuracy'] # track accuracy
)
model_unstr.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

#training the model
training_unstruc = model_unstr.fit(
    X_train_unstr, Y_train,
    validation_data = (X_val_unstr, Y_val),
    epochs = 100, # represents how many times we train over the entire dataset
    batch_size = 32, # how many values to look at before updating weights
    callbacks = [early_stopping]
)

training_struc = model_struc.fit(
    X_train_struc, Y_train,
    validation_data=(X_val_struc, Y_val),
    epochs = 100,
    batch_size = 32,
    callbacks = [early_stopping]
)

#SAVE THE WEIGHTS
model_struc.save_weights('model_struc_weights.h5')
model_unstr.save_weights('model_unstr_weights.h5')

#plot the loss vs val_loss data 
plt.title('Structured - Loss on Model Training (Training vs Validation)')
plt.plot(training_struc.history['loss'], label = 'Training Loss')
plt.plot(training_struc.history['val_loss'], label = 'Validation Loss')
plt.legend()
plt.grid()
plt.show()

plt.title('Unstructured - Accuracy on Model Training (Training vs Validation)')
plt.plot(training_unstruc.history['loss'], label = 'Training Loss')
plt.plot(training_unstruc.history['val_loss'], label = 'Validation Loss')
plt.legend()
plt.grid()
plt.show()

#plot the accuracy vs val_accuracy data
plt.title('Structured - Accuracy on Model Training (Training vs Validation)')
plt.plot(training_struc.history['accuracy'], label = 'Training Accuracy')
plt.plot(training_struc.history['val_accuracy'], label = 'Validation Accuracy')
plt.legend()
plt.grid()
plt.show

plt.title('Unstructured - Accuracy on Model Training (Training vs Validation)')
plt.plot(training_unstruc.history['accuracy'], label = 'Training Accuracy')
plt.plot(training_unstruc.history['val_accuracy'], label = 'Validation Accuracy')
plt.legend()
plt.grid()
plt.show

#Final Test of the model to evaluate
tst_loss_struc, tst_accuracy_struc = model_struc.evaluate(X_test_struc, Y_test, verbose = 2)
tst_loss_unstr, tst_accuracy_unstr = model_unstr.evaluate(X_test_unstr, Y_test, verbose = 2)

#Final Test of the model to predict
predict_struc = model_struc.predict(X_test_struc, Y_test, verbose = 2)
predict_unstr = model_unstr.predict(X_test_unstr, Y_test, verbose = 2)

predictions = (predict_unstr + predict_struc)/2

plt.title('Structured - Loss and Accuracy on Model Testing (Training vs Validation)')
plt.plot(tst_loss_struc, label = "Test Loss")
plt.plot(tst_accuracy_struc, label = 'Test Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.title('Unstructured - Loss and Accuracy on Model Testing (Training vs Validation)')
plt.plot(tst_loss_unstr, label = "Test Loss")
plt.plot(tst_accuracy_unstr, label = 'Test Accuracy')
plt.legend()
plt.grid()
plt.show()