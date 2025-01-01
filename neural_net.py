from tensorflow import keras
from tensorflow.python.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#initialize data
trnfile = open('clean_train_data.csv', 'r') # update both with actual files
tstfile = open('clean_test_data.csv', 'r')

X_train = trnfile.iloc[: , :-1]
Y_train = trnfile.iloc[: , -1]

X_test = tstfile.iloc[: , :-1] #For final check
Y_test = tstfile.iloc[: , -1]

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state = 42) # (80% train, 20% validation), random state to get same results for debugging

#Creating the model
model = keras.Sequential([

    # HIDDEN LAYERS
    layers.Dense(units = 50, activation = 'relu', use_bias = True,input_shape = [50]), #input shape based on feature count quantitative
    layers.Dense(units = 50, activation = 'relu', use_bias = True),
    layers.Dense(units = 50, activation = 'relu', use_bias = True),

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
model.compile(
    optimizer = 'adam', # chose adam over SGT (Stochastic Gradient Descent) since it has ability to adaptively change learning rates
    loss = 'binary_crossentropy', #using this loss for quantitative. It compares rate of yes to actual
    metrics = ['accuracy'] # track accuracy
)

#training the model
training = model.fit(
    X_train, Y_train,
    validation_data = (X_val, Y_val),
    epochs = 100, # represents how many times we train over the entire dataset
    batch_size = 32, # how many values to look at before updating weights
    callbacks = [early_stopping]
)

#plot the loss vs val_loss data 
plt.title('Loss on Model Training (Training vs Validation)')
plt.plot(training.history['loss'], label = 'Training Loss')
plt.plot(training.history['val_loss'], label = 'Validation Loss')
plt.legend()
plt.grid()
plt.show()

plt.title('Accuracy on Model Training (Training vs Validation)')
#plot the accuracy vs val_accuracy data
plt.plot(training.history['accuracy'], label = 'Training Accuracy')
plt.plot(training.history['val_accuracy'], label = 'Validation Accuracy')
plt.legend()
plt.grid()
plt.show()

#Final Test of the model
tst_loss, tst_accuracy = model.evaluate(X_test, Y_test, verbose = 2)

plt.title('Loss and Accuracy on Model Testing (Training vs Validation)')
plt.plot(tst_loss, label = "Test Loss")
plt.plot(tst_accuracy, label = 'Test Accuracy')
plt.legend()
plt.grid()
plt.show()